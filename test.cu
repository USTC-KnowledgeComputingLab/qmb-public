#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <torch/extension.h>

template<typename T, int size>
struct array_less {
    __host__ __device__ bool operator()(const std::array<T, size>& lhs, const std::array<T, size>& rhs) const {
        for (auto i = 0; i < size; ++i) {
            if (lhs[i] < rhs[i]) {
                return true;
            }
            if (lhs[i] > rhs[i]) {
                return false;
            }
        }
        return false;
    }
};

std::int64_t int_sqrt(std::int64_t x) {
    for (std::int64_t i = 0;; ++i) {
        if (1 << (2 * i) > x) {
            return 1 << (i - 1);
        }
    }
}

__device__ bool get_bit(std::uint8_t* data, int index) {
    return ((*data) & (1 << index)) >> index;
}

__device__ bool set_bit(std::uint8_t* data, int index, bool value) {
    if (value) {
        *data |= (1 << index);
    } else {
        *data &= ~(1 << index);
    }
}

TORCH_LIBRARY(qmb_test, m) {
    m.def("test(Tensor configs, Tensor psi, Tensor site, Tensor kind, Tensor coef) -> Tensor");
}

template<std::int64_t n_qubytes, std::int64_t max_op_number>
__device__ void test_kernel(
    std::int64_t term_index,
    std::int64_t batch_index,
    std::int64_t term_number,
    std::int64_t batch_size,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind, // term_number
    const std::array<double, 2>* coef, // term_number
    const std::array<std::uint8_t, n_qubytes>* configs, // batch_size
    const std::array<double, 2>* psi, // batch_size
    std::array<double, 2>* result
) {
    bool success = true;
    bool parity = false;
    std::array<std::uint8_t, n_qubytes> current_configs = configs[batch_index];
    for (auto op_index = max_op_number; op_index-- > 0;) {
        auto site_single = site[term_index][op_index];
        auto kind_single = kind[term_index][op_index];
        if (kind_single == 2) {
            continue;
        }
        auto to_what = kind_single;
        if (get_bit(&current_configs[site_single / 8], site_single % 8) == to_what) {
            success = false;
            break;
        }
        set_bit(&current_configs[site_single / 8], site_single % 8, to_what);
        for (auto s = 0; s < site_single; ++s) {
            parity ^= get_bit(&current_configs[s / 8], s % 8);
        }
    }
    if (!success) {
        return;
    }
    success = false;
    std::int64_t low = 0;
    std::int64_t high = batch_size - 1;
    std::int64_t mid = 0;
    auto compare = array_less<std::uint8_t, n_qubytes>();
    while (low <= high) {
        mid = (low + high) / 2;
        if (compare(current_configs, configs[mid])) {
            high = mid - 1;
        } else if (compare(configs[mid], current_configs)) {
            low = mid + 1;
        } else {
            success = true;
            break;
        }
    }
    if (success) {
        std::int8_t sign = parity ? -1 : +1;
        atomicAdd(&result[mid][0], sign * (coef[term_index][0] * psi[batch_index][0] - coef[term_index][1] * psi[batch_index][1]));
        atomicAdd(&result[mid][1], sign * (coef[term_index][0] * psi[batch_index][1] + coef[term_index][1] * psi[batch_index][0]));
    }
}

template<std::int64_t n_qubytes, std::int64_t max_op_number>
__global__ void test_kernel_interface(
    std::int64_t term_number,
    std::int64_t batch_size,
    const std::array<std::int16_t, max_op_number>* site,
    const std::array<std::uint8_t, max_op_number>* kind,
    const std::array<double, 2>* coef,
    const std::array<std::uint8_t, n_qubytes>* configs,
    const std::array<double, 2>* psi,
    std::array<double, 2>* result
) {
    int term_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (term_index < term_number && batch_index < batch_size) {
        test_kernel<n_qubytes, max_op_number>(term_index, batch_index, term_number, batch_size, site, kind, coef, configs, psi, result);
    }
}

template<std::int64_t n_qubytes, std::int64_t max_op_number>
auto test_interface(
    const torch::Tensor& configs,
    const torch::Tensor& psi,
    const torch::Tensor& site,
    const torch::Tensor& kind,
    const torch::Tensor& coef
) -> torch::Tensor {
    std::int64_t device_id = configs.device().index();
    std::int64_t batch_size = configs.size(0);
    std::int64_t term_number = site.size(0);
    auto policy = thrust::device.on(at::cuda::getCurrentCUDAStream(device_id));

    auto sorted_configs = configs.clone(torch::MemoryFormat::Contiguous);
    auto sorted_psi = psi.clone(torch::MemoryFormat::Contiguous);
    auto sorted_result = torch::zeros_like(sorted_psi);

    thrust::sort_by_key(
        policy,
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_configs.data_ptr()),
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_configs.data_ptr()) + batch_size,
        reinterpret_cast<std::array<double, 2>*>(sorted_psi.data_ptr()),
        array_less<std::uint8_t, n_qubytes>()
    );

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::int64_t max_threads_per_block = prop.maxThreadsPerBlock;

    auto threads_per_block = dim3{1, max_threads_per_block / 2};  // 不知道为什么，但是需要除以2避免出错
    auto num_blocks = dim3{
        (std::int32_t(term_number) + threads_per_block.x - 1) / threads_per_block.x,
        (std::int32_t(batch_size) + threads_per_block.y - 1) / threads_per_block.y
    };

    test_kernel_interface<n_qubytes, max_op_number><<<num_blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream(device_id)>>>(
        term_number,
        batch_size,
        reinterpret_cast<const std::array<std::int16_t, max_op_number>*>(site.data_ptr()),
        reinterpret_cast<const std::array<std::uint8_t, max_op_number>*>(kind.data_ptr()),
        reinterpret_cast<const std::array<double, 2>*>(coef.data_ptr()),
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_configs.data_ptr()),
        reinterpret_cast<const std::array<double, 2>*>(sorted_psi.data_ptr()),
        reinterpret_cast<std::array<double, 2>*>(sorted_result.data_ptr())
    );
    cudaDeviceSynchronize();
    return sorted_result;
}

#ifndef N_QUBYTES
#define N_QUBYTES 0
#endif
#ifndef MAX_OP_NUMBER
#define MAX_OP_NUMBER 0
#endif

TORCH_LIBRARY_IMPL(qmb_test, CUDA, m) {
    m.impl("test", test_interface<N_QUBYTES, MAX_OP_NUMBER>);
}

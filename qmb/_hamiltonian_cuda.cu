#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <torch/extension.h>

namespace qmb_hamiltonian_cuda {

constexpr torch::DeviceType device = torch::kCUDA;

template<typename T, std::int64_t size>
struct array_less {
    __device__ bool operator()(const std::array<T, size>& lhs, const std::array<T, size>& rhs) const {
        for (std::int64_t i = 0; i < size; ++i) {
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

template<typename T, std::int64_t size>
struct array_square_greater {
    __device__ T square(const std::array<T, size>& value) const {
        T result = 0;
        for (std::int64_t i = 0; i < size; ++i) {
            result += value[i] * value[i];
        }
        return result;
    }
    __device__ bool operator()(const std::array<T, size>& lhs, const std::array<T, size>& rhs) const {
        return square(lhs) > square(rhs);
    }
};

__device__ bool get_bit(std::uint8_t* data, std::uint8_t index) {
    return ((*data) >> index) & 1;
}

__device__ bool set_bit(std::uint8_t* data, std::uint8_t index, bool value) {
    if (value) {
        *data |= (1 << index);
    } else {
        *data &= ~(1 << index);
    }
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__device__ std::pair<bool, bool> hamiltonian_apply_kernel(
    std::array<std::uint8_t, n_qubytes>& current_configs,
    std::int64_t term_index,
    std::int64_t batch_index,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind // term_number
) {
    static_assert(particle_cut == 1 || particle_cut == 2, "particle_cut != 1 or 2 not implemented");
    bool success = true;
    bool parity = false;
    for (std::int64_t op_index = max_op_number; op_index-- > 0;) {
        std::int16_t site_single = site[term_index][op_index];
        std::uint8_t kind_single = kind[term_index][op_index];
        if (kind_single == 2) {
            continue;
        }
        std::uint8_t to_what = kind_single;
        if (get_bit(&current_configs[site_single / 8], site_single % 8) == to_what) {
            success = false;
            break;
        }
        set_bit(&current_configs[site_single / 8], site_single % 8, to_what);
        if constexpr (particle_cut == 1) {
            for (std::int16_t s = 0; s < site_single; ++s) {
                parity ^= get_bit(&current_configs[s / 8], s % 8);
            }
        }
    }
    return std::make_pair(success, parity);
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__device__ void apply_within_kernel(
    std::int64_t term_index,
    std::int64_t batch_index,
    std::int64_t term_number,
    std::int64_t batch_size,
    std::int64_t result_batch_size,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind, // term_number
    const std::array<double, 2>* coef, // term_number
    const std::array<std::uint8_t, n_qubytes>* configs, // batch_size
    const std::array<double, 2>* psi, // batch_size
    const std::array<std::uint8_t, n_qubytes>* result_configs, // result_batch_size
    std::array<double, 2>* result_psi
) {
    std::array<std::uint8_t, n_qubytes> current_configs = configs[batch_index];
    auto [success, parity] = hamiltonian_apply_kernel<max_op_number, n_qubytes, particle_cut>(
        /*current_configs=*/current_configs,
        /*term_index=*/term_index,
        /*batch_index=*/batch_index,
        /*site=*/site,
        /*kind=*/kind
    );

    if (!success) {
        return;
    }
    success = false;
    std::int64_t low = 0;
    std::int64_t high = result_batch_size - 1;
    std::int64_t mid = 0;
    auto compare = array_less<std::uint8_t, n_qubytes>();
    while (low <= high) {
        mid = (low + high) / 2;
        if (compare(current_configs, result_configs[mid])) {
            high = mid - 1;
        } else if (compare(result_configs[mid], current_configs)) {
            low = mid + 1;
        } else {
            success = true;
            break;
        }
    }
    if (!success) {
        return;
    }
    std::int8_t sign = parity ? -1 : +1;
    atomicAdd(&result_psi[mid][0], sign * (coef[term_index][0] * psi[batch_index][0] - coef[term_index][1] * psi[batch_index][1]));
    atomicAdd(&result_psi[mid][1], sign * (coef[term_index][0] * psi[batch_index][1] + coef[term_index][1] * psi[batch_index][0]));
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__global__ void apply_within_kernel_interface(
    std::int64_t term_number,
    std::int64_t batch_size,
    std::int64_t result_batch_size,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind, // term_number
    const std::array<double, 2>* coef, // term_number
    const std::array<std::uint8_t, n_qubytes>* configs, // batch_size
    const std::array<double, 2>* psi, // batch_size
    const std::array<std::uint8_t, n_qubytes>* result_configs, // result_batch_size
    std::array<double, 2>* result_psi
) {
    std::int64_t term_index = blockIdx.x * blockDim.x + threadIdx.x;
    std::int64_t batch_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (term_index < term_number && batch_index < batch_size) {
        apply_within_kernel<max_op_number, n_qubytes, particle_cut>(
            /*term_index=*/term_index,
            /*batch_index=*/batch_index,
            /*term_number=*/term_number,
            /*batch_size=*/batch_size,
            /*result_batch_size=*/result_batch_size,
            /*site=*/site,
            /*kind=*/kind,
            /*coef=*/coef,
            /*configs=*/configs,
            /*psi=*/psi,
            /*result_configs=*/result_configs,
            /*result_psi=*/result_psi
        );
    }
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
auto apply_within_interface(
    const torch::Tensor& configs,
    const torch::Tensor& psi,
    const torch::Tensor& result_configs,
    const torch::Tensor& site,
    const torch::Tensor& kind,
    const torch::Tensor& coef
) -> torch::Tensor {
    std::int64_t device_id = configs.device().index();
    std::int64_t batch_size = configs.size(0);
    std::int64_t result_batch_size = result_configs.size(0);
    std::int64_t term_number = site.size(0);

    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    auto policy = thrust::device.on(stream);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::int64_t max_threads_per_block = prop.maxThreadsPerBlock;
    std::int64_t max_blocks_per_multi_processor = prop.maxGridSize[0];

    auto sorted_result_configs = result_configs.clone(torch::MemoryFormat::Contiguous);
    auto result_sort_index = torch::arange(result_batch_size, torch::TensorOptions().dtype(torch::kInt64).device(device, device_id));
    auto sorted_result_psi = torch::zeros({result_batch_size, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(device, device_id));

    thrust::sort_by_key(
        policy,
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_result_configs.data_ptr()),
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_result_configs.data_ptr()) + result_batch_size,
        reinterpret_cast<std::int64_t*>(result_sort_index.data_ptr()),
        array_less<std::uint8_t, n_qubytes>()
    );

    auto threads_per_block = dim3{1, max_threads_per_block >> 1}; // I don't know why, but need to divide by 2 to avoid errors
    auto num_blocks_y = (batch_size + threads_per_block.y - 1) / threads_per_block.y;
    auto max_local_term_number = max_blocks_per_multi_processor / num_blocks_y;
    for (std::int64_t i = 0; i < term_number; i += max_local_term_number) {
        auto local_term_number = std::min(max_local_term_number, term_number - i);
        auto num_blocks = dim3{local_term_number, num_blocks_y};
        apply_within_kernel_interface<max_op_number, n_qubytes, particle_cut><<<num_blocks, threads_per_block, 0, stream>>>(
            /*term_number=*/local_term_number,
            /*batch_size=*/batch_size,
            /*result_batch_size=*/result_batch_size,
            /*site=*/reinterpret_cast<const std::array<std::int16_t, max_op_number>*>(site.data_ptr()) + i,
            /*kind=*/reinterpret_cast<const std::array<std::uint8_t, max_op_number>*>(kind.data_ptr()) + i,
            /*coef=*/reinterpret_cast<const std::array<double, 2>*>(coef.data_ptr()) + i,
            /*configs=*/reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(configs.data_ptr()),
            /*psi=*/reinterpret_cast<const std::array<double, 2>*>(psi.data_ptr()),
            /*result_configs=*/reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(sorted_result_configs.data_ptr()),
            /*result_psi=*/reinterpret_cast<std::array<double, 2>*>(sorted_result_psi.data_ptr())
        );
    }

    cudaStreamSynchronize(stream);

    auto result_psi = torch::zeros_like(sorted_result_psi);
    result_psi.index_put_({result_sort_index}, sorted_result_psi);
    return result_psi;
}

constexpr std::int64_t max_uint8_t = 256;
using largest_atomic_int = unsigned int; // The largest int type that can be atomicAdd/atomicSub
using smallest_atomic_int = unsigned short int; // The smallest int type that can be atomicCAS

template<std::int64_t n_qubytes>
struct dictionary_tree {
    using child_t = dictionary_tree<n_qubytes - 1>;
    child_t* children[max_uint8_t];
    smallest_atomic_int exist[max_uint8_t];
    largest_atomic_int nonzero_count;

    __device__ bool add(std::uint8_t* begin, double real, double imag) {
        std::uint8_t index = *begin;
        if (children[index] == nullptr) {
            if (atomicCAS(&exist[index], smallest_atomic_int(0), smallest_atomic_int(1))) {
                while (atomicCAS((largest_atomic_int*)&children[index], largest_atomic_int(0), largest_atomic_int(0)) == 0) {
                }
            } else {
                auto new_child = (child_t*)malloc(sizeof(child_t));
                memset(new_child, 0, sizeof(child_t));
                children[index] = new_child;
                __threadfence();
            }
        }
        if (children[index]->add(begin + 1, real, imag)) {
            atomicAdd(&nonzero_count, 1);
            return true;
        } else {
            return false;
        }
    }

    template<std::int64_t n_total_qubytes>
    __device__ void collect(std::uint64_t index, std::array<std::uint8_t, n_total_qubytes>* configs, std::array<double, 2>* psi) {
        std::uint64_t size_counter = 0;
        for (std::int64_t i = 0; i < max_uint8_t; ++i) {
            if (exist[i]) {
                std::uint64_t new_size_counter = size_counter + children[i]->nonzero_count;
                if (new_size_counter > index) {
                    std::uint64_t new_index = index - size_counter;
                    configs[index][n_total_qubytes - n_qubytes] = i;
                    children[i]->collect<n_total_qubytes>(new_index, &configs[size_counter], &psi[size_counter]);
                    if (atomicSub(&children[i]->nonzero_count, 1) == 1) {
                        free(children[i]);
                    };
                    return;
                }
                size_counter = new_size_counter;
            }
        }
    }
};

template<>
struct dictionary_tree<1> {
    double values[max_uint8_t][2];
    smallest_atomic_int exist[max_uint8_t];
    largest_atomic_int nonzero_count;

    __device__ bool add(std::uint8_t* begin, double real, double imag) {
        std::uint8_t index = *begin;
        atomicAdd(&values[index][0], real);
        atomicAdd(&values[index][1], imag);
        if (atomicCAS(&exist[index], smallest_atomic_int(0), smallest_atomic_int(1))) {
            return false;
        } else {
            atomicAdd(&nonzero_count, 1);
            return true;
        }
    }

    template<std::int64_t n_total_qubytes>
    __device__ void collect(std::uint64_t index, std::array<std::uint8_t, n_total_qubytes>* configs, std::array<double, 2>* psi) {
        std::uint64_t size_counter = 0;
        for (std::int64_t i = 0; i < max_uint8_t; ++i) {
            if (exist[i]) {
                if (size_counter == index) {
                    configs[index][n_total_qubytes - 1] = i;
                    psi[index][0] = values[i][0];
                    psi[index][1] = values[i][1];
                    return;
                }
                ++size_counter;
            }
        }
    }
};

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__device__ void find_relative_kernel(
    std::int64_t term_index,
    std::int64_t batch_index,
    std::int64_t term_number,
    std::int64_t batch_size,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind, // term_number
    const std::array<double, 2>* coef, // term_number
    const std::array<std::uint8_t, n_qubytes>* configs, // batch_size
    const std::array<double, 2>* psi, // batch_size
    dictionary_tree<n_qubytes>* result_tree
) {
    std::array<std::uint8_t, n_qubytes> current_configs = configs[batch_index];
    auto [success, parity] = hamiltonian_apply_kernel<max_op_number, n_qubytes, particle_cut>(
        /*current_configs=*/current_configs,
        /*term_index=*/term_index,
        /*batch_index=*/batch_index,
        /*site=*/site,
        /*kind=*/kind
    );

    if (!success) {
        return;
    }
    success = true;
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
            success = false;
            break;
        }
    }
    if (!success) {
        return;
    }
    std::int8_t sign = parity ? -1 : +1;
    result_tree->add(
        current_configs.data(),
        sign * (coef[term_index][0] * psi[batch_index][0] - coef[term_index][1] * psi[batch_index][1]),
        sign * (coef[term_index][0] * psi[batch_index][1] + coef[term_index][1] * psi[batch_index][0])
    );
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__global__ void find_relative_kernel_interface(
    std::int64_t term_number,
    std::int64_t batch_size,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind, // term_number
    const std::array<double, 2>* coef, // term_number
    const std::array<std::uint8_t, n_qubytes>* configs, // batch_size
    const std::array<double, 2>* psi, // batch_size
    dictionary_tree<n_qubytes>* result_tree
) {
    std::int64_t term_index = blockIdx.x * blockDim.x + threadIdx.x;
    std::int64_t batch_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (term_index < term_number && batch_index < batch_size) {
        find_relative_kernel<max_op_number, n_qubytes, particle_cut>(
            /*term_index=*/term_index,
            /*batch_index=*/batch_index,
            /*term_number=*/term_number,
            /*batch_size=*/batch_size,
            /*site=*/site,
            /*kind=*/kind,
            /*coef=*/coef,
            /*configs=*/configs,
            /*psi=*/psi,
            /*result_tree=*/result_tree
        );
    }
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__global__ void collect_kernel_interface(
    std::uint64_t result_size,
    dictionary_tree<n_qubytes>* result_tree,
    std::array<std::uint8_t, n_qubytes>* configs,
    std::array<double, 2>* psi
) {
    std::int64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < result_size) {
        result_tree->collect<n_qubytes>(index, configs, psi);
    }
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
auto find_relative_interface(
    const torch::Tensor& configs,
    const torch::Tensor& psi,
    const std::int64_t count_selected,
    const torch::Tensor& site,
    const torch::Tensor& kind,
    const torch::Tensor& coef
) -> torch::Tensor {
    std::int64_t device_id = configs.device().index();
    std::int64_t batch_size = configs.size(0);
    std::int64_t term_number = site.size(0);

    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    auto policy = thrust::device.on(stream);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::int64_t max_threads_per_block = prop.maxThreadsPerBlock;

    auto sorted_configs = configs.clone(torch::MemoryFormat::Contiguous);
    auto sorted_psi = psi.clone(torch::MemoryFormat::Contiguous);

    thrust::sort_by_key(
        policy,
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_configs.data_ptr()),
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_configs.data_ptr()) + batch_size,
        reinterpret_cast<std::array<double, 2>*>(sorted_psi.data_ptr()),
        array_less<std::uint8_t, n_qubytes>()
    );

    dictionary_tree<n_qubytes>* result_tree;
    cudaMalloc(&result_tree, sizeof(dictionary_tree<n_qubytes>));
    cudaMemset(result_tree, 0, sizeof(dictionary_tree<n_qubytes>));

    auto threads_per_block = dim3{1, max_threads_per_block >> 1}; // I don't know why, but need to divide by 2 to avoid errors
    auto num_blocks =
        dim3{(term_number + threads_per_block.x - 1) / threads_per_block.x, (batch_size + threads_per_block.y - 1) / threads_per_block.y};

    find_relative_kernel_interface<max_op_number, n_qubytes, particle_cut><<<num_blocks, threads_per_block, 0, stream>>>(
        /*term_number=*/term_number,
        /*batch_size=*/batch_size,
        /*site=*/reinterpret_cast<const std::array<std::int16_t, max_op_number>*>(site.data_ptr()),
        /*kind=*/reinterpret_cast<const std::array<std::uint8_t, max_op_number>*>(kind.data_ptr()),
        /*coef=*/reinterpret_cast<const std::array<double, 2>*>(coef.data_ptr()),
        /*configs=*/reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(sorted_configs.data_ptr()),
        /*psi=*/reinterpret_cast<const std::array<double, 2>*>(sorted_psi.data_ptr()),
        /*result_tree=*/result_tree
    );
    cudaStreamSynchronize(stream);

    largest_atomic_int result_size;
    cudaMemcpy(&result_size, &result_tree->nonzero_count, sizeof(largest_atomic_int), cudaMemcpyDeviceToHost);

    auto result_configs = torch::zeros({result_size, n_qubytes}, torch::TensorOptions().dtype(torch::kUInt8).device(device, device_id));
    auto result_psi = torch::zeros({result_size, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(device, device_id));

    auto threads_per_block_collect = max_threads_per_block >> 1; // I don't know why, but need to divide by 2 to avoid errors
    auto num_blocks_collect = (result_size + threads_per_block_collect - 1) / threads_per_block_collect;
    collect_kernel_interface<max_op_number, n_qubytes, particle_cut><<<num_blocks_collect, threads_per_block_collect, 0, stream>>>(
        /*result_size=*/result_size,
        /*result_tree=*/result_tree,
        /*configs=*/reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(result_configs.data_ptr()),
        /*psi=*/reinterpret_cast<std::array<double, 2>*>(result_psi.data_ptr())
    );
    cudaStreamSynchronize(stream);

    cudaFree(result_tree);

    thrust::sort_by_key(
        policy,
        reinterpret_cast<std::array<double, 2>*>(result_psi.data_ptr()),
        reinterpret_cast<std::array<double, 2>*>(result_psi.data_ptr()) + result_size,
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(result_configs.data_ptr()),
        array_square_greater<double, 2>()
    );

    return result_configs.index({torch::indexing::Slice(torch::indexing::None, count_selected)});
}

#ifndef N_QUBYTES
#define N_QUBYTES 0
#endif
#ifndef PARTICLE_CUT
#define PARTICLE_CUT 0
#endif

#if N_QUBYTES != 0
#define QMB_LIBRARY_HELPER(x, y) qmb_hamiltonian_##x##_##y
#define QMB_LIBRARY(x, y) QMB_LIBRARY_HELPER(x, y)
TORCH_LIBRARY_IMPL(QMB_LIBRARY(N_QUBYTES, PARTICLE_CUT), CUDA, m) {
    m.impl("apply_within", apply_within_interface</*max_op_number=*/4, /*n_qubytes=*/N_QUBYTES, /*particle_cut=*/PARTICLE_CUT>);
    m.impl("find_relative", find_relative_interface</*max_op_number=*/4, /*n_qubytes=*/N_QUBYTES, /*particle_cut=*/PARTICLE_CUT>);
}
#undef QMB_LIBRARY
#undef QMB_LIBRARY_HELPER
#endif

} // namespace qmb_hamiltonian_cuda

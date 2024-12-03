#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <torch/extension.h>

namespace qmb_collection_cuda {

constexpr torch::DeviceType device = torch::kCUDA;

template<int value>
struct ConstInt {
    static constexpr int get_value() {
        return value;
    }
};

template<typename R>
R to_const_int_helper(int value) {
    TORCH_CHECK(false, "dimension not allowed");
}

template<typename R, int Head, int... Tail>
R to_const_int_helper(int value) {
    if (value == Head) {
        return ConstInt<Head>();
    } else {
        return to_const_int_helper<R, Tail...>(value);
    }
}

template<int... Values>
auto to_const_int(int value, std::integer_sequence<int, Values...>) {
    using R = std::variant<ConstInt<Values>...>;
    return to_const_int_helper<R, Values...>(value);
}

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

template<typename T, int size>
struct array_equal {
    __host__ __device__ bool operator()(const std::array<T, size>& lhs, const std::array<T, size>& rhs) const {
        for (auto i = 0; i < size; ++i) {
            if (lhs[i] != rhs[i]) {
                return false;
            }
        }
        return true;
    }
};

template<typename T, int size>
struct array_reduce {
    __host__ __device__ std::array<T, size> operator()(const std::array<T, size>& lhs, const std::array<T, size>& rhs) const {
        std::array<T, size> result;
        for (auto i = 0; i < size; ++i) {
            result[i] = lhs[i] + rhs[i];
        }
        return result;
    }
};

template<typename T, int size>
struct array_square_greater {
    __host__ __device__ T square(const std::array<T, size>& value) const {
        T result = 0;
        for (auto i = 0; i < size; ++i) {
            result += value[i] * value[i];
        }
        return result;
    }
    __host__ __device__ bool operator()(const std::array<T, size>& lhs, const std::array<T, size>& rhs) const {
        return square(lhs) > square(rhs);
    }
};

template<int n_qubits, int n_values>
void sort_impl(torch::Tensor& key, torch::Tensor& value) {
    std::int64_t length = key.size(0);
    thrust::sort_by_key(
        thrust::device.on(at::cuda::getCurrentCUDAStream(key.device().index())),
        reinterpret_cast<std::array<std::uint8_t, n_qubits>*>(key.data_ptr()),
        reinterpret_cast<std::array<std::uint8_t, n_qubits>*>(key.data_ptr()) + length,
        reinterpret_cast<std::array<double, n_values>*>(value.data_ptr()),
        array_less<std::uint8_t, n_qubits>()
    );
}

template<typename NQubits, typename NValues>
void sort(int n_qubits, int n_values, torch::Tensor& key, torch::Tensor& value) {
    std::visit(
        [&](auto n_qubits_handle, auto n_values_handle) {
            constexpr int n_qubits = n_qubits_handle.get_value();
            constexpr int n_values = n_values_handle.get_value();
            sort_impl<n_qubits, n_values>(key, value);
        },
        to_const_int(n_qubits, NQubits()),
        to_const_int(n_values, NValues())
    );
}

template<int n_qubits, int n_values>
void merge_impl(
    const torch::Tensor& key_1,
    const torch::Tensor& value_1,
    const torch::Tensor& key_2,
    const torch::Tensor& value_2,
    torch::Tensor& key_result,
    torch::Tensor& value_result
) {
    std::int64_t length_1 = key_1.size(0);
    std::int64_t length_2 = key_2.size(0);
    thrust::merge_by_key(
        thrust::device.on(at::cuda::getCurrentCUDAStream(key_1.device().index())),
        reinterpret_cast<std::array<std::uint8_t, n_qubits>*>(key_1.data_ptr()),
        reinterpret_cast<std::array<std::uint8_t, n_qubits>*>(key_1.data_ptr()) + length_1,
        reinterpret_cast<std::array<std::uint8_t, n_qubits>*>(key_2.data_ptr()),
        reinterpret_cast<std::array<std::uint8_t, n_qubits>*>(key_2.data_ptr()) + length_2,
        reinterpret_cast<std::array<double, n_values>*>(value_1.data_ptr()),
        reinterpret_cast<std::array<double, n_values>*>(value_2.data_ptr()),
        reinterpret_cast<std::array<std::uint8_t, n_qubits>*>(key_result.data_ptr()),
        reinterpret_cast<std::array<double, n_values>*>(value_result.data_ptr()),
        array_less<std::uint8_t, n_qubits>()
    );
}

template<typename NQubits, typename NValues>
void merge(
    int n_qubits,
    int n_values,
    const torch::Tensor& key_1,
    const torch::Tensor& value_1,
    const torch::Tensor& key_2,
    const torch::Tensor& value_2,
    torch::Tensor& key_result,
    torch::Tensor& value_result
) {
    std::visit(
        [&](auto n_qubits_handle, auto n_values_handle) {
            constexpr int n_qubits = n_qubits_handle.get_value();
            constexpr int n_values = n_values_handle.get_value();
            merge_impl<n_qubits, n_values>(key_1, value_1, key_2, value_2, key_result, value_result);
        },
        to_const_int(n_qubits, NQubits()),
        to_const_int(n_values, NValues())
    );
}

template<int n_qubits, int n_values>
std::int64_t reduce_impl(const torch::Tensor& key, const torch::Tensor& value, torch::Tensor& key_result, torch::Tensor& value_result) {
    std::int64_t length = key.size(0);
    auto [key_end, value_end] = thrust::reduce_by_key(
        thrust::device.on(at::cuda::getCurrentCUDAStream(key.device().index())),
        reinterpret_cast<std::array<std::uint8_t, n_qubits>*>(key.data_ptr()),
        reinterpret_cast<std::array<std::uint8_t, n_qubits>*>(key.data_ptr()) + length,
        reinterpret_cast<std::array<double, n_values>*>(value.data_ptr()),
        reinterpret_cast<std::array<std::uint8_t, n_qubits>*>(key_result.data_ptr()),
        reinterpret_cast<std::array<double, n_values>*>(value_result.data_ptr()),
        array_equal<std::uint8_t, n_qubits>(),
        array_reduce<double, n_values>()
    );
    return key_end - reinterpret_cast<std::array<std::uint8_t, n_qubits>*>(key_result.data_ptr());
}

template<typename NQubits, typename NValues>
std::int64_t
reduce(int n_qubits, int n_values, const torch::Tensor& key, const torch::Tensor& value, torch::Tensor& key_result, torch::Tensor& value_result) {
    return std::visit(
        [&](auto n_qubits_handle, auto n_values_handle) {
            constexpr int n_qubits = n_qubits_handle.get_value();
            constexpr int n_values = n_values_handle.get_value();
            return reduce_impl<n_qubits, n_values>(key, value, key_result, value_result);
        },
        to_const_int(n_qubits, NQubits()),
        to_const_int(n_values, NValues())
    );
}

template<int n_qubits, int n_values>
__global__ void
ensure_kernel(std::int64_t length, std::int64_t length_config, const std::array<std::uint8_t, n_qubits>* key, std::array<double, n_values>* value) {
    std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length_config) {
        return;
    }
    std::int64_t low = 0;
    std::int64_t high = length - 1;
    std::int64_t mid = 0;
    auto compare = array_less<std::uint8_t, n_qubits>();
    while (low <= high) {
        mid = (low + high) / 2;
        if (compare(key[length_config + mid], key[i])) {
            low = mid + 1;
        } else if (compare(key[i], key[length_config + mid])) {
            high = mid - 1;
        } else {
            for (auto j = 0; j < n_values; ++j) {
                value[i][j] = value[length_config + mid][j];
                value[length_config + mid][j] = 0;
            }
            return;
        }
    }
}

template<int n_qubits, int n_values>
std::int64_t ensure_impl(torch::Tensor& key, torch::Tensor& value, std::int64_t length_config) {
    std::int64_t length = key.size(0) - length_config;

    std::int64_t device_id = key.device().index();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::int64_t threads_per_block = prop.maxThreadsPerBlock;
    std::int64_t num_blocks = (length_config + threads_per_block - 1) / threads_per_block;

    ensure_kernel<n_qubits, n_values><<<num_blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream(device_id)>>>(
        length,
        length_config,
        reinterpret_cast<std::array<std::uint8_t, n_qubits>*>(key.data_ptr()),
        reinterpret_cast<std::array<double, n_values>*>(value.data_ptr())
    );

    thrust::sort_by_key(
        thrust::device.on(at::cuda::getCurrentCUDAStream(device_id)),
        reinterpret_cast<std::array<double, n_values>*>(value.data_ptr()) + length_config,
        reinterpret_cast<std::array<double, n_values>*>(value.data_ptr()) + length_config + length,
        reinterpret_cast<std::array<std::uint8_t, n_qubits>*>(key.data_ptr()) + length_config,
        array_square_greater<double, n_values>()
    );

    std::array<double, n_values> zero_array;
    auto end = thrust::lower_bound(
        thrust::device.on(at::cuda::getCurrentCUDAStream(device_id)),
        reinterpret_cast<std::array<double, n_values>*>(value.data_ptr()) + length_config,
        reinterpret_cast<std::array<double, n_values>*>(value.data_ptr()) + length_config + length,
        zero_array,
        array_square_greater<double, n_values>()
    );
    return end - reinterpret_cast<std::array<double, n_values>*>(value.data_ptr());
}

template<typename NQubits, typename NValues>
std::int64_t ensure(int n_qubits, int n_values, torch::Tensor& key, torch::Tensor& value, std::int64_t length_config) {
    return std::visit(
        [&](auto n_qubits_handle, auto n_values_handle) {
            constexpr int n_qubits = n_qubits_handle.get_value();
            constexpr int n_values = n_values_handle.get_value();
            return ensure_impl<n_qubits, n_values>(key, value, length_config);
        },
        to_const_int(n_qubits, NQubits()),
        to_const_int(n_values, NValues())
    );
}

// key: A uint8 tensor of shape [length_x, n_qubits]
// value: A float64 tensor of shape [length_x, n_values] where n_values = 1 or 2.

template<typename NQubits, typename NValues>
auto sort_interface(torch::Tensor& key, torch::Tensor& value) -> std::tuple<torch::Tensor, torch::Tensor> {
    std::int64_t length = key.size(0);
    std::int64_t n_qubits = key.size(1);
    std::int64_t n_values = value.size(1);

    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(key.ndimension() == 2, "key must be a 2D tensor");
    TORCH_CHECK(value.ndimension() == 2, "value must be a 2D tensor");
    TORCH_CHECK(key.size(0) == value.size(0), "key and value must have the same length");

    sort<NQubits, NValues>(n_qubits, n_values, key, value);

    return std::make_tuple(key, value);
}

template<typename NQubits, typename NValues>
auto merge_interface(const torch::Tensor& key_1, const torch::Tensor& value_1, const torch::Tensor& key_2, const torch::Tensor& value_2)
    -> std::tuple<torch::Tensor, torch::Tensor> {
    std::int64_t length_1 = key_1.size(0);
    std::int64_t length_2 = key_2.size(0);
    std::int64_t n_qubits = key_1.size(1);
    std::int64_t n_values = value_1.size(1);

    TORCH_CHECK(key_1.is_contiguous(), "key_1 must be contiguous");
    TORCH_CHECK(key_2.is_contiguous(), "key_2 must be contiguous");
    TORCH_CHECK(value_1.is_contiguous(), "value_1 must be contiguous");
    TORCH_CHECK(value_2.is_contiguous(), "value_2 must be contiguous");
    TORCH_CHECK(key_1.ndimension() == 2, "key_1 must be a 2D tensor");
    TORCH_CHECK(key_1.size(0) == length_1, "key_1 must have the correct length");
    TORCH_CHECK(key_1.size(1) == n_qubits, "key_1 must have the correct length");
    TORCH_CHECK(value_1.ndimension() == 2, "value_1 must be a 2D tensor");
    TORCH_CHECK(value_1.size(0) == length_1, "value_1 must have the correct length");
    TORCH_CHECK(value_1.size(1) == n_values, "value_1 must have the correct length");
    TORCH_CHECK(key_2.ndimension() == 2, "key_2 must be a 2D tensor");
    TORCH_CHECK(key_2.size(0) == length_2, "key_2 must have the correct length");
    TORCH_CHECK(key_2.size(1) == n_qubits, "key_2 must have the correct length");
    TORCH_CHECK(value_2.ndimension() == 2, "value_2 must be a 2D tensor");
    TORCH_CHECK(value_2.size(0) == length_2, "value_2 must have the correct length");
    TORCH_CHECK(value_2.size(1) == n_values, "value_2 must have the correct length");

    std::int64_t device_id = key_1.device().index();
    auto key_result = torch::empty({length_1 + length_2, n_qubits}, torch::TensorOptions().dtype(torch::kUInt8).device(device, device_id));
    auto value_result = torch::empty({length_1 + length_2, n_values}, torch::TensorOptions().dtype(torch::kFloat64).device(device, device_id));

    merge<NQubits, NValues>(n_qubits, n_values, key_1, value_1, key_2, value_2, key_result, value_result);

    return std::make_tuple(key_result, value_result);
}

template<typename NQubits, typename NValues>
auto reduce_interface(const torch::Tensor& key, const torch::Tensor& value) -> std::tuple<torch::Tensor, torch::Tensor> {
    std::int64_t length = key.size(0);
    std::int64_t n_qubits = key.size(1);
    std::int64_t n_values = value.size(1);

    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(key.ndimension() == 2, "key must be a 2D tensor");
    TORCH_CHECK(value.ndimension() == 2, "value must be a 2D tensor");
    TORCH_CHECK(key.size(0) == value.size(0), "key and value must have the same length");

    std::int64_t device_id = key.device().index();
    auto key_result = torch::empty({length, n_qubits}, torch::TensorOptions().dtype(torch::kUInt8).device(device, device_id));
    auto value_result = torch::empty({length, n_values}, torch::TensorOptions().dtype(torch::kFloat64).device(device, device_id));

    std::int64_t size = reduce<NQubits, NValues>(n_qubits, n_values, key, value, key_result, value_result);
    auto slice = torch::indexing::Slice(torch::indexing::None, size);

    return std::make_tuple(key_result.index({slice}), value_result.index({slice}));
}

template<typename NQubits, typename NValues>
auto ensure_interface(torch::Tensor& key, torch::Tensor& value, std::int64_t length_config) -> std::tuple<torch::Tensor, torch::Tensor> {
    std::int64_t length = key.size(0);
    std::int64_t n_qubits = key.size(1);
    std::int64_t n_values = value.size(1);

    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(key.ndimension() == 2, "key must be a 2D tensor");
    TORCH_CHECK(value.ndimension() == 2, "value must be a 2D tensor");
    TORCH_CHECK(key.size(0) == value.size(0), "key and value must have the same length");

    std::int64_t size = ensure<NQubits, NValues>(n_qubits, n_values, key, value, length_config);
    auto slice = torch::indexing::Slice(torch::indexing::None, size);

    return std::make_tuple(key.index({slice}), value.index({slice}));
}

#ifndef NQUBYTES
#define NQUBYTES 0
#endif

#ifndef QMB_LIBRARY_HELPER
#define QMB_LIBRARY_HELPER(x) qmb_collection_##x
#endif
#ifndef QMB_LIBRARY
#define QMB_LIBRARY(x) QMB_LIBRARY_HELPER(x)
#endif

#if NQUBYTES != 0
TORCH_LIBRARY_IMPL(QMB_LIBRARY(NQUBYTES), CUDA, m) {
    m.impl("sort_", sort_interface<std::integer_sequence<int, NQUBYTES>, std::integer_sequence<int, 1, 2>>);
    m.impl("merge", merge_interface<std::integer_sequence<int, NQUBYTES>, std::integer_sequence<int, 1, 2>>);
    m.impl("reduce", reduce_interface<std::integer_sequence<int, NQUBYTES>, std::integer_sequence<int, 1, 2>>);
    m.impl("ensure_", ensure_interface<std::integer_sequence<int, NQUBYTES>, std::integer_sequence<int, 1, 2>>);
}
#endif

} // namespace qmb_collection_cuda

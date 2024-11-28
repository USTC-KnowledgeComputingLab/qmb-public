#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
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
struct zero_check {
    __host__ __device__ bool non_zero(const std::array<T, size>& value) const {
        for (auto i = 0; i < size; ++i) {
            if (value[i] != 0) {
                return true;
            }
        }
        return false;
    }
    __host__ __device__ bool operator()(const std::array<T, size>& lhs, const std::array<T, size>& rhs) const {
        return non_zero(lhs) < non_zero(rhs);
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

template<int n_qubits, int n_values>
void sort_impl(torch::Tensor& key, torch::Tensor& value) {
    std::int64_t length = key.size(0);
    thrust::sort_by_key(
        thrust::device,
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
    torch::Tensor& key_1,
    torch::Tensor& value_1,
    torch::Tensor& key_2,
    torch::Tensor& value_2,
    torch::Tensor& key_result,
    torch::Tensor& value_result
) {
    std::int64_t length_1 = key_1.size(0);
    std::int64_t length_2 = key_2.size(0);
    thrust::merge_by_key(
        thrust::device,
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
    torch::Tensor& key_1,
    torch::Tensor& value_1,
    torch::Tensor& key_2,
    torch::Tensor& value_2,
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
std::int64_t reduce_impl(torch::Tensor& key, torch::Tensor& value, torch::Tensor& key_result, torch::Tensor& value_result) {
    std::int64_t length = key.size(0);
    auto [key_end, value_end] = thrust::reduce_by_key(
        thrust::device,
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
std::int64_t reduce(int n_qubits, int n_values, torch::Tensor& key, torch::Tensor& value, torch::Tensor& key_result, torch::Tensor& value_result) {
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
void ensure_impl(torch::Tensor& key, torch::Tensor& value, torch::Tensor& config, torch::Tensor& tmp) {
    std::int64_t length = key.size(0);
    std::int64_t length_config = config.size(0);
    for (auto i = 0; i < length_config; ++i) {
        auto [begin, end] = thrust::equal_range(
            thrust::device,
            reinterpret_cast<std::array<std::uint8_t, n_qubits>*>(key.data_ptr()),
            reinterpret_cast<std::array<std::uint8_t, n_qubits>*>(key.data_ptr()) + length,
            reinterpret_cast<std::array<std::uint8_t, n_qubits>*>(config.data_ptr())[i],
            array_less<std::uint8_t, n_qubits>()
        );
        TORCH_CHECK(begin + 1 == end, "Duplicate keys found in the input tensor.");
        std::int64_t index = begin - reinterpret_cast<std::array<std::uint8_t, n_qubits>*>(key.data_ptr());
        tmp.index_put_({i}, value.index({index}));
        value.index_put_({index}, 0);
    }
    thrust::stable_sort_by_key(
        thrust::device,
        reinterpret_cast<std::array<double, n_values>*>(value.data_ptr()),
        reinterpret_cast<std::array<double, n_values>*>(value.data_ptr()) + length,
        reinterpret_cast<std::array<std::uint8_t, n_qubits>*>(key.data_ptr()),
        zero_check<double, n_values>()
    );
    key.index_put_({torch::indexing::Slice(torch::indexing::None, length_config)}, config);
    value.index_put_({torch::indexing::Slice(torch::indexing::None, length_config)}, tmp);
}

template<typename NQubits, typename NValues>
void ensure(int n_qubits, int n_values, torch::Tensor& key, torch::Tensor& value, torch::Tensor& config, torch::Tensor& tmp) {
    std::visit(
        [&](auto n_qubits_handle, auto n_values_handle) {
            constexpr int n_qubits = n_qubits_handle.get_value();
            constexpr int n_values = n_values_handle.get_value();
            ensure_impl<n_qubits, n_values>(key, value, config, tmp);
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
auto merge_interface(torch::Tensor& key_1, torch::Tensor& value_1, torch::Tensor& key_2, torch::Tensor& value_2)
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

    auto key_result = torch::empty({length_1 + length_2, n_qubits}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    auto value_result = torch::empty({length_1 + length_2, n_values}, torch::TensorOptions().dtype(torch::kFloat64).device(device));

    merge<NQubits, NValues>(n_qubits, n_values, key_1, value_1, key_2, value_2, key_result, value_result);

    return std::make_tuple(key_result, value_result);
}

template<typename NQubits, typename NValues>
auto reduce_interface(torch::Tensor& key, torch::Tensor& value) -> std::tuple<torch::Tensor, torch::Tensor> {
    std::int64_t length = key.size(0);
    std::int64_t n_qubits = key.size(1);
    std::int64_t n_values = value.size(1);

    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(key.ndimension() == 2, "key must be a 2D tensor");
    TORCH_CHECK(value.ndimension() == 2, "value must be a 2D tensor");
    TORCH_CHECK(key.size(0) == value.size(0), "key and value must have the same length");

    auto key_result = torch::empty({length, n_qubits}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    auto value_result = torch::empty({length, n_values}, torch::TensorOptions().dtype(torch::kFloat64).device(device));

    std::int64_t size = reduce<NQubits, NValues>(n_qubits, n_values, key, value, key_result, value_result);
    auto slice = torch::indexing::Slice(torch::indexing::None, size);

    return std::make_tuple(key_result.index({slice}), value_result.index({slice}));
}

template<typename NQubits, typename NValues>
auto ensure_interface(torch::Tensor& key, torch::Tensor& value, torch::Tensor& config) -> std::tuple<torch::Tensor, torch::Tensor> {
    std::int64_t length = key.size(0);
    std::int64_t n_qubits = key.size(1);
    std::int64_t n_values = value.size(1);
    std::int64_t length_config = config.size(0);

    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(config.is_contiguous(), "config must be contiguous");
    TORCH_CHECK(key.ndimension() == 2, "key must be a 2D tensor");
    TORCH_CHECK(value.ndimension() == 2, "value must be a 2D tensor");
    TORCH_CHECK(config.ndimension() == 2, "config must be a 2D tensor");
    TORCH_CHECK(key.size(0) == value.size(0), "key and value must have the same length");
    TORCH_CHECK(config.size(1) == key.size(1), "config must have the same number of qubits as key");

    auto tmp = torch::empty({length_config, n_values}, torch::TensorOptions().dtype(torch::kFloat64).device(device));

    ensure<NQubits, NValues>(n_qubits, n_values, key, value, config, tmp);

    return std::make_tuple(key, value);
}

TORCH_LIBRARY_IMPL(qmb_collection, CUDA, m) {
    m.impl("sort_", sort_interface<std::make_integer_sequence<int, 30>, std::integer_sequence<int, 1, 2>>);
    m.impl("merge", merge_interface<std::make_integer_sequence<int, 30>, std::integer_sequence<int, 1, 2>>);
    m.impl("reduce", reduce_interface<std::make_integer_sequence<int, 30>, std::integer_sequence<int, 1, 2>>);
    m.impl("ensure_", ensure_interface<std::make_integer_sequence<int, 30>, std::integer_sequence<int, 1, 2>>);
}

} // namespace qmb_collection_cuda

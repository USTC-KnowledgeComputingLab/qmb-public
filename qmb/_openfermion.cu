#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr int max_op_number = 4;
constexpr torch::DeviceType device = torch::kCUDA;

__device__ void search_kernel(
    std::int64_t term_index,
    std::int64_t batch_index,
    std::int64_t batch_size,
    std::int64_t term_number,
    torch::PackedTensorAccessor64<int16_t, 2>& site_accesor,
    torch::PackedTensorAccessor64<int8_t, 2>& kind_accesor,
    torch::PackedTensorAccessor64<double, 2>& coef_accesor,
    torch::PackedTensorAccessor64<bool, 3>& configs_j_matrix_accesor,
    torch::PackedTensorAccessor64<double, 3>& coefs_matrix_accesor
) {
    bool success = true;
    bool parity = false;
    for (auto op_index = max_op_number; op_index-- > 0;) {
        auto site_single = site_accesor[term_index][op_index];
        auto kind_single = kind_accesor[term_index][op_index];
        if (kind_single == 2) {
            // 2 for empty
            continue;
        }
        auto to_what = bool(kind_single);
        // 0 for annihilation
        // 1 for creation
        if (configs_j_matrix_accesor[term_index][batch_index][site_single] == to_what) {
            success = false;
            break;
        }
        configs_j_matrix_accesor[term_index][batch_index][site_single] = to_what;
        for (auto s = 0; s < site_single; ++s) {
            parity ^= configs_j_matrix_accesor[term_index][batch_index][s];
        }
    }
    if (success) {
        std::int8_t sign = parity ? -1 : +1;
        coefs_matrix_accesor[term_index][batch_index][0] = sign * coef_accesor[term_index][0];
        coefs_matrix_accesor[term_index][batch_index][1] = sign * coef_accesor[term_index][1];
    }
}

__global__ void search_kernel_interface(
    std::int64_t term_number,
    std::int64_t batch_size,
    torch::PackedTensorAccessor64<int16_t, 2> site_accesor,
    torch::PackedTensorAccessor64<int8_t, 2> kind_accesor,
    torch::PackedTensorAccessor64<double, 2> coef_accesor,
    torch::PackedTensorAccessor64<bool, 3> configs_j_matrix_accesor,
    torch::PackedTensorAccessor64<double, 3> coefs_matrix_accesor
) {
    int term_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (term_index < term_number && batch_index < batch_size) {
        search_kernel(
            term_index,
            batch_index,
            batch_size,
            term_number,
            site_accesor,
            kind_accesor,
            coef_accesor,
            configs_j_matrix_accesor,
            coefs_matrix_accesor
        );
    }
}

void launch_search_kernel(
    std::int64_t term_number,
    std::int64_t batch_size,
    torch::PackedTensorAccessor64<int16_t, 2>& site_accesor,
    torch::PackedTensorAccessor64<int8_t, 2>& kind_accesor,
    torch::PackedTensorAccessor64<double, 2>& coef_accesor,
    torch::PackedTensorAccessor64<bool, 3>& configs_j_matrix_accesor,
    torch::PackedTensorAccessor64<double, 3>& coefs_matrix_accesor
) {
    auto threads_per_block = dim3{16, 16};
    auto num_blocks = dim3{
        (std::int32_t(term_number) + threads_per_block.x - 1) / threads_per_block.x,
        (std::int32_t(batch_size) + threads_per_block.y - 1) / threads_per_block.y
    };
    search_kernel_interface<<<num_blocks, threads_per_block>>>(
        term_number,
        batch_size,
        site_accesor,
        kind_accesor,
        coef_accesor,
        configs_j_matrix_accesor,
        coefs_matrix_accesor
    );
    cudaDeviceSynchronize();
}

class Hamiltonian {
  public:
    std::int64_t term_number;
    torch::Tensor site; // int16[max_op_number, term_number]
    torch::Tensor kind; // int8[max_op_number, term_number]
    torch::Tensor coef; // float64[term_number, 2]

    Hamiltonian(py::dict openfermion_hamiltonian) {
        auto& self = *this;

        self.term_number = openfermion_hamiltonian.size();

        std::int64_t index = 0;
        torch::Tensor cpu_site = torch::zeros({self.term_number, max_op_number}, torch::kInt16);
        torch::Tensor cpu_kind = torch::zeros({self.term_number, max_op_number}, torch::kInt8);
        torch::Tensor cpu_coef = torch::zeros({self.term_number, 2}, torch::kDouble);
        auto cpu_site_accessor = cpu_site.accessor<std::int16_t, 2>();
        auto cpu_kind_accessor = cpu_kind.accessor<std::int8_t, 2>();
        auto cpu_coef_accessor = cpu_coef.accessor<double, 2>();

        for (auto item : openfermion_hamiltonian) {
            auto key = item.first.cast<py::tuple>();
            auto value_is_float = py::isinstance<py::float_>(item.second);
            auto value = value_is_float ? std::complex<double>(item.second.cast<double>()) : item.second.cast<std::complex<double>>();

            std::int64_t op_number = key.size();
            for (auto i = 0; i < op_number; ++i) {
                cpu_site_accessor[index][i] = key[i].cast<py::tuple>()[0].cast<std::int16_t>();
                cpu_kind_accessor[index][i] = key[i].cast<py::tuple>()[1].cast<std::int8_t>();
            }
            for (auto i = op_number; i < max_op_number; ++i) {
                cpu_kind_accessor[index][i] = 2;
            }

            cpu_coef_accessor[index][0] = value.real();
            cpu_coef_accessor[index][1] = value.imag();

            ++index;
        }

        site = cpu_site.to(device);
        kind = cpu_kind.to(device);
        coef = cpu_coef.to(device);
    }

    auto relative(torch::Tensor configs_i) {
        auto& self = *this;
        // Parameters
        // configs_i : bool[batch_size, n_qubits]
        // Returns
        // valid_index_i : int64[...]
        // valid_configs_j : bool[..., n_qubits]
        // valid_coefs : float64[..., 2]

        std::int64_t batch_size = configs_i.size(0);
        std::int64_t n_qubits = configs_i.size(1);

        // configs_j_matrix: bool[term_number, batch_size, n_qubits]
        auto configs_j_matrix = configs_i.unsqueeze(0).repeat({self.term_number, 1, 1});
        // coefs_matrix: float64[term_number, batch_size, 2]
        auto coefs_matrix = torch::zeros({self.term_number, batch_size, 2}, torch::TensorOptions().dtype(torch::kDouble).device(device));
        // Apply all term to configs_j_matrix and update coefs_matrix
        auto site_accesor = self.site.packed_accessor64<std::int16_t, 2>();
        auto kind_accesor = self.kind.packed_accessor64<std::int8_t, 2>();
        auto coef_accesor = self.coef.packed_accessor64<double, 2>();
        auto configs_j_matrix_accesor = configs_j_matrix.packed_accessor64<bool, 3>();
        auto coefs_matrix_accesor = coefs_matrix.packed_accessor64<double, 3>();
        launch_search_kernel(self.term_number, batch_size, site_accesor, kind_accesor, coef_accesor, configs_j_matrix_accesor, coefs_matrix_accesor);

        auto index_i = torch::arange(batch_size, torch::TensorOptions().dtype(torch::kInt64).device(device));
        // index_i_matrix : int64[term_number, batch_size]
        auto index_i_matrix = index_i.unsqueeze(0).repeat({term_number, 1});
        // non_zero_matrix : bool[term_number, batch_size]
        auto non_zero_matrix = torch::any(coefs_matrix != 0, -1);

        // view configs_j_matrix, coefs_matrix, index_i_matrix, non_zero_matrix as vector
        auto configs_j_vector = configs_j_matrix.view({-1, n_qubits});
        auto coefs_vector = coefs_matrix.view({-1, 2});
        auto index_i_vector = index_i_matrix.view({-1});
        auto non_zero_vector = non_zero_matrix.view({-1});
        auto non_zero_indices = non_zero_vector.nonzero().squeeze(1);

        // select them only when non zero
        auto valid_configs_j = configs_j_vector.index_select(0, non_zero_indices);
        auto valid_coefs = coefs_vector.index_select(0, non_zero_indices);
        auto valid_index_i = index_i_vector.index_select(0, non_zero_indices);

        return std::make_tuple(valid_index_i, valid_configs_j, valid_coefs);
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Hamiltonian>(m, "Hamiltonian", py::module_local())
        .def(py::init<py::dict>(), py::arg("openfermion_hamiltonian"))
        .def("relative", &Hamiltonian::relative, py::arg("configs_i"));
}

#include <cuda_runtime.h>
#include <torch/extension.h>

// This file implements the iteration over Hamiltonian terms for quantum many-body systems.
// It utilizes several template arguments to tailor the computation:
//   - max_op_number: Specifies the maximum number of operations for all terms in the Hamiltonian.
//   - particle_cut: Determines the system type; particle_cut >= 2 indicates a boson system with a specific number cut,
//                   while particle_cut = 1 signifies a fermion system.

// The device is currently set to CUDA, which may be subject to change in the future.
// If we decide to transition to a different device, it will necessitate modifications to the kernel code.
constexpr torch::DeviceType device = torch::kCUDA;

// The search kernel is designed for iterating over Hamiltonian terms in quantum many-body systems.
// Each thread processes a single term of the Hamiltonian and a single configuration within the batch.
// The parameter `max_op_number` typically has a value of 4, indicating the maximum number of operations for all terms in the Hamiltonian.
// The parameter `particle_cut` differentiates between fermion and boson systems:
//   - A value of 1 for `particle_cut` corresponds to a fermion system.
//   - A value of 2 or greater for `particle_cut` designates a boson system with a specific particle cutoff, applicable to various spin systems,
//     notably including spin-1/2 systems where particle_cut = 2.
// The `site` and `kind` tensors are expected to have a shape of [term_number, max_op_number],
// where `term_number` represents the number of terms in the Hamiltonian
// and `max_op_number` is the maximum number of operations for all terms.
// The `coef` tensor has a shape of [term_number, 2], with the second dimension representing the real and imaginary parts of the coefficient.
// This structure is chosen because the C++ API of PyTorch does not support complex numbers well.
// The `configs_j_matrix` and `coefs_matrix` tensors are output matrices.
// The `configs_j_matrix` has a shape of [batch_size, term_number, n_qubits],
// The `coefs_matrix` tensor is structured as [batch_size, term_number, 2],
// where the final dimension delineates the real and imaginary components of the coefficients.
template<std::int64_t max_op_number, std::int64_t particle_cut>
__device__ void search_kernel(
    std::int64_t term_index,
    std::int64_t batch_index,
    std::int64_t batch_size,
    std::int64_t term_number,
    torch::PackedTensorAccessor64<std::int16_t, 2>& site_accesor,
    torch::PackedTensorAccessor64<std::int8_t, 2>& kind_accesor,
    torch::PackedTensorAccessor64<double, 2>& coef_accesor,
    torch::PackedTensorAccessor64<std::int8_t, 3>& configs_j_matrix_accesor,
    torch::PackedTensorAccessor64<double, 3>& coefs_matrix_accesor
) {
    if constexpr (particle_cut == 1) {
        // Occasionally, such as when attempting to annihilate on a vacuum state, the state may become zero. In such cases, we need to skip these
        // terms.
        bool success = true;
        // The parity is crucial when applying fermion operators to the state, due to the anti-commutation rules of fermions.
        bool parity = false;
        // Apply the operators in reverse order to the state.
        for (auto op_index = max_op_number; op_index-- > 0;) {
            auto site_single = site_accesor[term_index][op_index];
            auto kind_single = kind_accesor[term_index][op_index];
            // When `kind_single` equals 2, it represents an empty operator, effectively an identity operation that has no impact on the state.
            // Thus, we proceed to process the next operation in the sequence.
            if (kind_single == 2) {
                continue;
            }
            // When `kind_single` is 1, it represents a creation operator, and when it is 0, it represents an annihilation operator.
            // We must verify whether the state is already occupied or vacant, and adjust the state accordingly.
            auto to_what = kind_single;
            if (configs_j_matrix_accesor[term_index][batch_index][site_single] == to_what) {
                success = false;
                break;
            }
            configs_j_matrix_accesor[term_index][batch_index][site_single] = to_what;
            // Calculate the parity by summing the parity of the particle number up to the current site.
            for (auto s = 0; s < site_single; ++s) {
                parity ^= configs_j_matrix_accesor[term_index][batch_index][s];
            }
        }
        // Upon successful completion, store the coefficients in the `coefs_matrix` tensor.
        // This involves applying the sign to both the real and imaginary parts of the coefficients.
        // The configuration matrix `configs_j` has already been computed in the preceding steps.
        if (success) {
            std::int8_t sign = parity ? -1 : +1;
            coefs_matrix_accesor[term_index][batch_index][0] = sign * coef_accesor[term_index][0];
            coefs_matrix_accesor[term_index][batch_index][1] = sign * coef_accesor[term_index][1];
        }
    } else if constexpr (particle_cut == 2) {
        // For the boson case with a particle cutoff of 2, the operations are identical to those in the fermion case,
        // with the notable exception that parity considerations are not required.
        bool success = true;
        for (auto op_index = max_op_number; op_index-- > 0;) {
            auto site_single = site_accesor[term_index][op_index];
            auto kind_single = kind_accesor[term_index][op_index];
            if (kind_single == 2) {
                continue;
            }
            auto to_what = kind_single;
            if (configs_j_matrix_accesor[term_index][batch_index][site_single] == to_what) {
                success = false;
                break;
            }
            configs_j_matrix_accesor[term_index][batch_index][site_single] = to_what;
        }
        if (success) {
            coefs_matrix_accesor[term_index][batch_index][0] = coef_accesor[term_index][0];
            coefs_matrix_accesor[term_index][batch_index][1] = coef_accesor[term_index][1];
        }
    } else {
        static_assert(particle_cut <= 2, "particle_cut > 2 not implemented");
    }
}

// The kernel interface for invoking the `search_kernel`.
// This interface is responsible for calculating the `term_index` and `batch_index` based on the block and thread indices.
template<std::int64_t max_op_number, std::int64_t particle_cut>
__global__ void search_kernel_interface(
    std::int64_t term_number,
    std::int64_t batch_size,
    torch::PackedTensorAccessor64<std::int16_t, 2> site_accesor,
    torch::PackedTensorAccessor64<std::int8_t, 2> kind_accesor,
    torch::PackedTensorAccessor64<double, 2> coef_accesor,
    torch::PackedTensorAccessor64<std::int8_t, 3> configs_j_matrix_accesor,
    torch::PackedTensorAccessor64<double, 3> coefs_matrix_accesor
) {
    int term_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (term_index < term_number && batch_index < batch_size) {
        search_kernel<max_op_number, particle_cut>(
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

// Launch the search kernel, specifying the grid and block dimensions.
// We have two distinct data scenarios:
// 1. Processing multiple batch configurations and numerous Hamiltonian terms.
//    In this case, we utilize the term index for the x-axis and the batch index for the y-axis.
//    Each block consists of 16 threads along the x-axis and 16 threads along the y-axis.
// 2. Handling a single configuration.
//    Here, we only use the term index for the x-axis.
//    Each block is set to 256 threads to maximize throughput.
template<std::int64_t max_op_number, std::int64_t particle_cut>
void launch_search_kernel(
    std::int64_t term_number,
    std::int64_t batch_size,
    torch::PackedTensorAccessor64<std::int16_t, 2>& site_accesor,
    torch::PackedTensorAccessor64<std::int8_t, 2>& kind_accesor,
    torch::PackedTensorAccessor64<double, 2>& coef_accesor,
    torch::PackedTensorAccessor64<std::int8_t, 3>& configs_j_matrix_accesor,
    torch::PackedTensorAccessor64<double, 3>& coefs_matrix_accesor
) {
    if (batch_size != 1) {
        auto threads_per_block = dim3{16, 16};
        auto num_blocks = dim3{
            (std::int32_t(term_number) + threads_per_block.x - 1) / threads_per_block.x,
            (std::int32_t(batch_size) + threads_per_block.y - 1) / threads_per_block.y
        };
        search_kernel_interface<max_op_number, particle_cut><<<num_blocks, threads_per_block>>>(
            term_number,
            batch_size,
            site_accesor,
            kind_accesor,
            coef_accesor,
            configs_j_matrix_accesor,
            coefs_matrix_accesor
        );
    } else {
        auto threads_per_block = dim3{256, 1};
        auto num_blocks = dim3{(std::int32_t(term_number) + threads_per_block.x - 1) / threads_per_block.x, 1};
        search_kernel_interface<max_op_number, particle_cut><<<num_blocks, threads_per_block>>>(
            term_number,
            batch_size,
            site_accesor,
            kind_accesor,
            coef_accesor,
            configs_j_matrix_accesor,
            coefs_matrix_accesor
        );
    }
    cudaDeviceSynchronize();
}

template<std::int64_t max_op_number, std::int64_t particle_cut>
class Hamiltonian {
  public:
    // Number of terms in the Hamiltonian
    std::int64_t term_number;

    // Tensor storing the site indices for each operation in the Hamiltonian terms
    // Shape: [max_op_number, term_number], dtype: int16
    torch::Tensor site;

    // Tensor storing the kind of each operation in the Hamiltonian terms
    // Shape: [max_op_number, term_number], dtype: int8
    torch::Tensor kind;

    // Tensor storing the coefficients (real and imaginary parts) for each Hamiltonian term
    // Shape: [term_number, 2], dtype: float64
    torch::Tensor coef;

    // The argument `hamiltonian` should be a dictionary where:
    // - The keys are tuples of tuples, each representing a sequence of operations for a term.
    // - The values are complex numbers representing the coefficient of the corresponding term.
    // Each inner tuple represents a single operator, with the first integer indicating the site index
    // and the second integer indicating the kind of operator (0 for annihilation, 1 for creation).
    Hamiltonian(py::dict hamiltonian) {
        auto& self = *this;

        self.term_number = hamiltonian.size();

        std::int64_t index = 0;
        auto cpu_site = torch::zeros({self.term_number, max_op_number}, torch::kInt16);
        auto cpu_kind = torch::zeros({self.term_number, max_op_number}, torch::kInt8);
        auto cpu_coef = torch::zeros({self.term_number, 2}, torch::kDouble);
        auto cpu_site_accessor = cpu_site.template accessor<std::int16_t, 2>();
        auto cpu_kind_accessor = cpu_kind.template accessor<std::int8_t, 2>();
        auto cpu_coef_accessor = cpu_coef.template accessor<double, 2>();

        for (auto item : hamiltonian) {
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

    // This function computes the relative configurations and coefficients for a given set of input configurations.
    // It applies the Hamiltonian terms to the input configurations and evaluate the coefficients accordingly.
    // The function returns the indices of the valid input configurations, the corresponding output configurations, and the updated coefficients.
    auto relative(torch::Tensor configs_i) {
        auto& self = *this;
        // Parameters:
        // configs_i: A int8 tensor of shape [batch_size, n_qubits] representing the input configurations.
        //
        // Returns:
        // valid_index_i: A tensor of shape [...] containing the indices of valid input configurations.
        // valid_configs_j: A int8 tensor of shape [..., n_qubits] representing the output configurations.
        // valid_coefs: A tensor of shape [..., 2] containing the updated coefficients.

        std::int64_t batch_size = configs_i.size(0);
        std::int64_t n_qubits = configs_i.size(1);

        // configs_j_matrix: A int8 tensor of shape [term_number, batch_size, n_qubits],
        // that captures the modifications to the input configurations by applying each term in the Hamiltonian.
        auto configs_j_matrix = configs_i.unsqueeze(0).repeat(std::initializer_list<std::int64_t>{self.term_number, 1, 1});
        // coefs_matrix: A tensor of shape [term_number, batch_size, 2] initialized to zero.
        auto coefs_matrix = torch::zeros({self.term_number, batch_size, 2}, torch::TensorOptions().dtype(torch::kDouble).device(device));
        // Obtain accessors for each relevant tensor to facilitate efficient data access within the kernel.
        auto site_accesor = self.site.template packed_accessor64<std::int16_t, 2>();
        auto kind_accesor = self.kind.template packed_accessor64<std::int8_t, 2>();
        auto coef_accesor = self.coef.template packed_accessor64<double, 2>();
        auto configs_j_matrix_accesor = configs_j_matrix.template packed_accessor64<std::int8_t, 3>();
        auto coefs_matrix_accesor = coefs_matrix.template packed_accessor64<double, 3>();
        // Apply all Hamiltonian terms to the configurations in configs_j_matrix(copyied from configs_i),
        // and evaluate the corresponding coefficients in coefs_matrix.
        launch_search_kernel<max_op_number, particle_cut>(
            self.term_number,
            batch_size,
            site_accesor,
            kind_accesor,
            coef_accesor,
            configs_j_matrix_accesor,
            coefs_matrix_accesor
        );

        // index_i : int64[batch_size]
        auto index_i = torch::arange(batch_size, torch::TensorOptions().dtype(torch::kInt64).device(device));
        // index_i_matrix : int64[term_number, batch_size]
        auto index_i_matrix = index_i.unsqueeze(0).repeat({term_number, 1});
        // non_zero_matrix : bool[term_number, batch_size]
        auto non_zero_matrix = torch::any(coefs_matrix != 0, -1);

        // View configs_j_matrix, coefs_matrix, index_i_matrix, and non_zero_matrix into vector form
        auto configs_j_vector = configs_j_matrix.view({-1, n_qubits});
        auto coefs_vector = coefs_matrix.view({-1, 2});
        auto index_i_vector = index_i_matrix.view({-1});
        auto non_zero_vector = non_zero_matrix.view({-1});

        // Identify the indices of non-zero elements for further processing
        auto non_zero_indices = non_zero_vector.nonzero().squeeze(1);

        // Select the valid configurations, coefficients, and indices based on non-zero elements
        auto valid_configs_j = configs_j_vector.index_select(0, non_zero_indices);
        auto valid_coefs = coefs_vector.index_select(0, non_zero_indices);
        auto valid_index_i = index_i_vector.index_select(0, non_zero_indices);

        return std::make_tuple(valid_index_i, valid_configs_j, valid_coefs);
    }
};

using FermiHamiltonian = Hamiltonian</*max_op_number=*/4, /*particle_cut=*/1>;
using Bose2Hamiltonian = Hamiltonian</*max_op_number=*/4, /*particle_cut=*/2>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<FermiHamiltonian>(m, "fermi", py::module_local())
        .def(py::init<py::dict>(), py::arg("hamiltonian"))
        .def("relative", &FermiHamiltonian::relative, py::arg("configs_i"));
    py::class_<Bose2Hamiltonian>(m, "bose2", py::module_local())
        .def(py::init<py::dict>(), py::arg("hamiltonian"))
        .def("relative", &Bose2Hamiltonian::relative, py::arg("configs_i"));
}

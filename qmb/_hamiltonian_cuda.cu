// This file implements a PyTorch operator designed to efficiently iterate over Hamiltonian terms in quantum many-body systems on CUDA devices.
// It is tailored to support both fermion and boson systems, with specific optimizations for different particle cutoffs.
// It utilizes several template arguments to tailor the computation:
//   - max_op_number: Specifies the maximum number of operations for all terms in the Hamiltonian, typically set to 4.
//   - particle_cut: Determines the system type; particle_cut >= 2 indicates a boson system with a specific number cut,
//                   while particle_cut = 1 signifies a fermion system.
// This file encompasses multiple functions designed to achieve the following objectives:
// 1. `search_kernel`: A device function responsible for processing a single term and a single configuration within the Hamiltonian.
// 2. `search_kernel_interface`: A global function that orchestrates the invocation of `search_kernel`. It determines which term and configuration
//    each thread should process based on the thread and grid indices.
// 3. `launch_search_kernel`: A host function dedicated to launching the `search_kernel_interface`. It strategically allocates grid and thread
//    dimensions to ensure all terms and configurations are processed efficiently.
// 4. `python_interface`: The PyTorch operator interface, which integrates the CUDA kernels into the PyTorch framework.

#include <cuda_runtime.h>
#include <torch/extension.h>

namespace qmb_hamiltonian_cuda {

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
// where the last dimension delineates the real and imaginary components of the coefficients.
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
#pragma unroll
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
#pragma unroll
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

std::int64_t int_sqrt(std::int64_t x) {
    for (std::int64_t i = 0;; ++i) {
        if (1 << (2 * i) > x) {
            return 1 << (i - 1);
        }
    }
}

// Launch the search kernel, specifying the grid and block dimensions.
// We have two distinct data scenarios:
// 1. Processing multiple batch configurations and numerous Hamiltonian terms.
//    In this case, we utilize the term index for the x-axis and the batch index for the y-axis.
//    Each block consists of sqrt(max_threads_per_block) threads along the x-axis and sqrt(max_threads_per_block) threads along the y-axis.
// 2. Handling a single configuration.
//    Here, we only use the term index for the x-axis.
//    Each block is set to max_threads_per_block threads to maximize throughput.
template<std::int64_t max_op_number, std::int64_t particle_cut>
void launch_search_kernel(
    std::int64_t term_number,
    std::int64_t batch_size,
    torch::PackedTensorAccessor64<std::int16_t, 2>& site_accesor,
    torch::PackedTensorAccessor64<std::int8_t, 2>& kind_accesor,
    torch::PackedTensorAccessor64<double, 2>& coef_accesor,
    torch::PackedTensorAccessor64<std::int8_t, 3>& configs_j_matrix_accesor,
    torch::PackedTensorAccessor64<double, 3>& coefs_matrix_accesor,
    std::int64_t device_id
) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::int64_t max_threads_per_block = prop.maxThreadsPerBlock;
    std::int64_t sqrt_max_threads_per_block = int_sqrt(max_threads_per_block);

    if (batch_size != 1) {
        auto threads_per_block = dim3{sqrt_max_threads_per_block, sqrt_max_threads_per_block};
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
        auto threads_per_block = dim3{max_threads_per_block, 1};
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

// This function computes the relative configurations and coefficients based on a given set of input configurations.
// It applies the Hamiltonian terms to the input configurations and evaluate the coefficients accordingly.
// The function returns the indices of the valid input configurations, the corresponding output configurations, and the coefficients.
// The last three arguments are prepared by the `prepare` function and stored on the Python side.
// Refer to `_hamiltonian.cpp` for additional details and context.
template<std::int64_t max_op_number, std::int64_t particle_cut>
auto python_interface(torch::Tensor configs_i, torch::Tensor site, torch::Tensor kind, torch::Tensor coef, bool early_drop) {
    std::int64_t batch_size = configs_i.size(0);
    std::int64_t n_qubits = configs_i.size(1);
    std::int64_t term_number = site.size(0);

    // configs_j_matrix: A int8 tensor of shape [term_number, batch_size, n_qubits],
    // that captures the modifications to the input configurations by applying each term in the Hamiltonian.
    auto configs_j_matrix = configs_i.unsqueeze(0).repeat(std::initializer_list<std::int64_t>{term_number, 1, 1});
    // coefs_matrix: A tensor of shape [term_number, batch_size, 2] initialized to zero.
    auto coefs_matrix = torch::zeros({term_number, batch_size, 2}, torch::TensorOptions().dtype(torch::kDouble).device(device));
    // Obtain accessors for each relevant tensor to facilitate efficient data access within the kernel.
    auto site_accesor = site.template packed_accessor64<std::int16_t, 2>();
    auto kind_accesor = kind.template packed_accessor64<std::int8_t, 2>();
    auto coef_accesor = coef.template packed_accessor64<double, 2>();
    auto configs_j_matrix_accesor = configs_j_matrix.template packed_accessor64<std::int8_t, 3>();
    auto coefs_matrix_accesor = coefs_matrix.template packed_accessor64<double, 3>();
    // Apply all Hamiltonian terms to the configurations in configs_j_matrix(copyied from configs_i),
    // and evaluate the corresponding coefficients in coefs_matrix.
    launch_search_kernel<max_op_number, particle_cut>(
        term_number,
        batch_size,
        site_accesor,
        kind_accesor,
        coef_accesor,
        configs_j_matrix_accesor,
        coefs_matrix_accesor,
        configs_i.device().index()
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

auto drop_early(torch::Tensor index_i, torch::Tensor configs_j, torch::Tensor coefs, torch::Tensor configs_i) {
    torch::Tensor configs_i_and_j = torch::cat({configs_i, configs_j}, /*dim=*/0);
    auto [pool, both_to_pool, none] = torch::unique_dim(configs_i_and_j, /*dim=*/0, /*sorted=*/false, /*return_inverse=*/true);
    configs_i_and_j.reset();

    std::int64_t batch_size = configs_i.size(0);
    std::int64_t pool_size = pool.size(0);

    torch::Tensor pool_to_source = torch::full({pool_size}, -1, torch::TensorOptions().dtype(torch::kInt64).device(device));
    torch::Tensor source_to_pool = both_to_pool.index({torch::indexing::Slice(torch::indexing::None, batch_size)});
    pool_to_source.index_put_({source_to_pool}, torch::arange(batch_size, torch::TensorOptions().dtype(torch::kInt64).device(device)));
    source_to_pool.reset();

    torch::Tensor destination_to_pool = both_to_pool.index({torch::indexing::Slice(batch_size, torch::indexing::None)});
    torch::Tensor destination_to_source = pool_to_source.index({destination_to_pool});
    pool_to_source.reset();
    destination_to_pool.reset();

    torch::Tensor usable = destination_to_source != -1;
    destination_to_source.reset();

    return std::make_tuple(index_i.index({usable}), configs_j.index({usable}), coefs.index({usable}));
}

// This function encapsulates the `python_interface` to facilitate job splitting, thereby mitigating potential memory leaks.
// It partitions the input tensor into pieces, invokes the `python_interface` on each segment, and subsequently merges the results.
template<std::int64_t max_op_number, std::int64_t particle_cut>
auto python_interface_with_batch_split(torch::Tensor configs_i, torch::Tensor site, torch::Tensor kind, torch::Tensor coef, bool early_drop) {
    std::int64_t batch_size = configs_i.size(0);
    std::vector<torch::Tensor> index_i_pool;
    std::vector<torch::Tensor> configs_j_pool;
    std::vector<torch::Tensor> coefs_pool;
    for (std::int64_t i = 0; i < batch_size; ++i) {
        auto [index_i, configs_j, coefs] = python_interface<max_op_number, particle_cut>(
            configs_i.index({torch::indexing::Slice(i, i + 1, torch::indexing::None)}),
            site,
            kind,
            coef,
            early_drop
        );
        if (early_drop) {
            auto [dropped_index_i, dropped_configs_j, dropped_coefs] = drop_early(index_i, configs_j, coefs, configs_i);
            index_i_pool.push_back(dropped_index_i);
            configs_j_pool.push_back(dropped_configs_j);
            coefs_pool.push_back(dropped_coefs);
        } else {
            index_i_pool.push_back(index_i);
            configs_j_pool.push_back(configs_j);
            coefs_pool.push_back(coefs);
        }
    }
    return std::make_tuple(torch::cat(index_i_pool, /*dim=*/0), torch::cat(configs_j_pool, /*dim=*/0), torch::cat(coefs_pool, /*dim=*/0));
}

// This function encapsulates the `python_interface` to facilitate job splitting, thereby mitigating potential memory leaks.
// It partitions the input tensor into pieces, invokes the `python_interface` on each segment, and subsequently merges the results.
template<std::int64_t max_op_number, std::int64_t particle_cut, std::int64_t group_size>
auto python_interface_with_term_group(torch::Tensor configs_i, torch::Tensor site, torch::Tensor kind, torch::Tensor coef, bool early_drop) {
    std::int64_t batch_size = configs_i.size(0);
    std::int64_t term_number = site.size(0);
    std::vector<torch::Tensor> index_i_pool;
    std::vector<torch::Tensor> configs_j_pool;
    std::vector<torch::Tensor> coefs_pool;
    for (std::int64_t i = 0; i < term_number; i += group_size) {
        auto [index_i, configs_j, coefs] = python_interface<max_op_number, particle_cut>(
            configs_i,
            site.index({torch::indexing::Slice(i, i + group_size, torch::indexing::None)}),
            kind.index({torch::indexing::Slice(i, i + group_size, torch::indexing::None)}),
            coef.index({torch::indexing::Slice(i, i + group_size, torch::indexing::None)}),
            early_drop
        );
        if (early_drop) {
            auto [dropped_index_i, dropped_configs_j, dropped_coefs] = drop_early(index_i, configs_j, coefs, configs_i);
            index_i_pool.push_back(dropped_index_i);
            configs_j_pool.push_back(dropped_configs_j);
            coefs_pool.push_back(dropped_coefs);
        } else {
            index_i_pool.push_back(index_i);
            configs_j_pool.push_back(configs_j);
            coefs_pool.push_back(coefs);
        }
    }
    return std::make_tuple(torch::cat(index_i_pool, /*dim=*/0), torch::cat(configs_j_pool, /*dim=*/0), torch::cat(coefs_pool, /*dim=*/0));
}

// Definition of the CUDA kernel implementation for operators.
TORCH_LIBRARY_IMPL(_hamiltonian, CUDA, m) {
    m.impl("fermi", python_interface_with_term_group</*max_op_number=*/4, /*particle_cut=*/1, /*group_size=*/256>);
    m.impl("bose2", python_interface_with_term_group</*max_op_number=*/4, /*particle_cut=*/2, /*group_size=*/256>);
}

} // namespace qmb_hamiltonian_cuda

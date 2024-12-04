// This file implements auxiliary functions related to Hamiltonian iteration results.
// The iteration results come in three formats: Raw, Inside, and Outside.
// Raw represents the format returned by the kernel without any processing.
// Inside restricts the results to the original subspace.
// Outside does not impose any restrictions.
// This file provides merge functions for each of these formats.
// Additionally, it includes functions to convert Raw to Inside and Outside formats.

#include <pybind11/complex.h>
#include <torch/extension.h>

namespace qmb_collection {

// The raw format is the output directly from the kernel, and it follows this structure:
// - `index_i`: An int64 tensor of shape [batch_size], which indexes the original terms.
// - `configs_j`: An uint8 tensor of shape [batch_size, n_qubits], providing the configuration details of the results.
// - `coefs`: A float64 tensor of shape [batch_size, 2], encoding the coefficients (both real and imaginary parts) of the terms.
// Here, `batch_size` represents the number of non-zero terms after iterating over the Hamiltonian based on the given configurations.
using Raw = std::tuple<
    torch::Tensor, // index_i
    torch::Tensor, // configs_j
    torch::Tensor // coefs
    >;

// The inside format is a collection of terms that are inside the subspace, and it follows this structure:
// - `index_i`: An int64 tensor of shape [batch_size], which indexes the original configs.
// - `index_j`: An int64 tensor of shape [batch_size], which indexes the original configs.
// - `coefs`: A float64 tensor of shape [batch_size, 2], encoding the coefficients (both real and imaginary parts) of the terms.
using Inside = std::tuple<
    torch::Tensor, // index_i
    torch::Tensor, // index_j
    torch::Tensor // coefs
    >;

// The outside format is a collection of terms that are outside the subspace, and it follows this structure:
// - `index_i`: An int64 tensor of shape [batch_size], which indexes the original configs.
// - `index_j`: An int64 tensor of shape [batch_size], which indexes the result configs.
// - `coefs`: A float64 tensor of shape [batch_size, 2], encoding the coefficients (both real and imaginary parts) of the terms.
// - `configs_j`: A uint8 tensor of shape [batch_size, num_qubits], encoding the result configs.
using Outside = std::tuple<
    torch::Tensor, // index_i
    torch::Tensor, // index_j
    torch::Tensor, // coefs
    torch::Tensor // configs_j
    >;

// Differently from the former three formats, which are a sparse representation of the hamiltonian.
// Sometimes we need to represent the sparse wavefunction only, which is what this format is for.
// The sparse format of wavefunctions is a collection of configuration and the corresponding wavefunction values, and it follows this structure:
// - `psi_j`: A float64 tensor of shape [batch_size, 1/2], encoding the wavefunction values of the terms, which maybe real of complex.
// - `configs_j`: A uint8 tensor of shape [batch_size, num_qubits], encoding the result configs.
using Sparse = std::tuple<
    torch::Tensor, // psi_j
    torch::Tensor // configs_j
    >;

// Merge a collection of Raw objects into a single Raw object.
auto merge_raw(std::vector<Raw> raws) -> Raw {
    auto index_i_pool = std::vector<torch::Tensor>();
    auto configs_j_pool = std::vector<torch::Tensor>();
    auto coefs_pool = std::vector<torch::Tensor>();

    for (auto& raw : raws) {
        auto& [index_i, configs_j, coefs] = raw;
        index_i_pool.push_back(index_i);
        configs_j_pool.push_back(configs_j);
        coefs_pool.push_back(coefs);
    }

    auto index_i = torch::cat(index_i_pool, /*dim=*/0);
    auto configs_j = torch::cat(configs_j_pool, /*dim=*/0);
    auto coefs = torch::cat(coefs_pool, /*dim=*/0);

    return Raw(index_i, configs_j, coefs);
}

// Convert a Raw object to an Inside object.
auto raw_to_inside(Raw raw, torch::Tensor configs_i) -> Inside {
    auto device = configs_i.device();
    auto& [index_i, configs_j, coefs] = raw;

    auto [pool, both_to_pool, nothing] = torch::unique_dim(
        torch::cat({configs_i, configs_j}, /*dim=*/0),
        /*dim=*/0,
        /*sorted=*/false,
        /*return_inverse=*/true,
        /*return_counts=*/false
    );

    std::int64_t src_size = configs_i.size(0);
    std::int64_t pool_size = pool.size(0);

    auto dst_to_src = [&]() {
        auto pool_to_src = [&]() {
            auto src_to_pool = both_to_pool.index({torch::indexing::Slice(torch::indexing::None, src_size)});
            auto pool_to_src = torch::full({pool_size}, -1, torch::TensorOptions().dtype(torch::kInt64).device(device));
            pool_to_src.index_put_({src_to_pool}, torch::arange(src_size, torch::TensorOptions().dtype(torch::kInt64).device(device)));
            return pool_to_src;
        }();
        auto dst_to_pool = both_to_pool.index({torch::indexing::Slice(src_size, torch::indexing::None)});
        auto dst_to_src = pool_to_src.index({dst_to_pool});
        return dst_to_src;
    }();

    auto usable = dst_to_src != -1;

    auto usable_index_i = index_i.index({usable});
    auto usable_index_j = dst_to_src.index({usable});
    auto usable_coefs = coefs.index({usable});

    return Inside(usable_index_i, usable_index_j, usable_coefs);
}

// Merge a collection of Inside objects into a single Inside object.
auto merge_inside(std::vector<Inside> insides) -> Inside {
    auto index_i_pool = std::vector<torch::Tensor>();
    auto index_j_pool = std::vector<torch::Tensor>();
    auto coefs_pool = std::vector<torch::Tensor>();

    for (auto& inside : insides) {
        auto& [index_i, index_j, coefs] = inside;
        index_i_pool.push_back(index_i);
        index_j_pool.push_back(index_j);
        coefs_pool.push_back(coefs);
    }

    auto index_i = torch::cat(index_i_pool, /*dim=*/0);
    auto index_j = torch::cat(index_j_pool, /*dim=*/0);
    auto coefs = torch::cat(coefs_pool, /*dim=*/0);

    return Inside(index_i, index_j, coefs);
}

// Convert a Raw object into an Outside object.
auto raw_to_outside(Raw raw, torch::Tensor configs_i) -> Outside {
    auto device = configs_i.device();
    auto& [index_i, configs_j, coefs] = raw;

    auto [pool, both_to_pool, nothing] = torch::unique_dim(
        torch::cat({configs_i, configs_j}, /*dim=*/0),
        /*dim=*/0,
        /*sorted=*/false,
        /*return_inverse=*/true,
        /*return_counts=*/false
    );

    std::int64_t src_size = configs_i.size(0);
    std::int64_t pool_size = pool.size(0);

    auto pool_to_target = [&]() {
        auto src_to_pool = both_to_pool.index({torch::indexing::Slice(torch::indexing::None, src_size)});
        auto pool_to_target = torch::full({pool_size}, -1, torch::TensorOptions().dtype(torch::kInt64).device(device));
        pool_to_target.index_put_({src_to_pool}, torch::arange(src_size, torch::TensorOptions().dtype(torch::kInt64).device(device)));
        pool_to_target.index_put_(
            {pool_to_target == -1},
            torch::arange(src_size, pool_size, torch::TensorOptions().dtype(torch::kInt64).device(device))
        );
        return pool_to_target;
    }();

    auto dst_to_target = [&]() {
        auto dst_to_target = [&]() {
            auto dst_to_pool = both_to_pool.index({torch::indexing::Slice(src_size, torch::indexing::None)});
            auto dst_to_target = pool_to_target.index({dst_to_pool});
            return dst_to_target;
        }();
        return dst_to_target;
    }();

    auto target = [&]() {
        auto target = torch::empty_like(pool);
        target.index_put_({pool_to_target}, pool);
        return target;
    }();

    return Outside(index_i, dst_to_target, coefs, target);
}

// Merge a collection of Outside objects into a single Outside object.
auto merge_outside(std::vector<Outside> outsides, std::optional<torch::Tensor> configs_i) -> Outside {
    if (configs_i.has_value()) {
        auto device = configs_i.value().device();

        auto index_i_pool = std::vector<torch::Tensor>();
        auto index_j_pool = std::vector<torch::Tensor>();
        auto coefs_pool = std::vector<torch::Tensor>();
        auto configs_pool = std::vector<torch::Tensor>();

        configs_pool.push_back(configs_i.value());
        for (auto& outside : outsides) {
            auto& [index_i, index_j, coefs, configs_j] = outside;
            index_i_pool.push_back(index_i);
            index_j_pool.push_back(index_j);
            coefs_pool.push_back(coefs);
            configs_pool.push_back(configs_j);
        }

        auto [pool, all_to_pool, nothing] = torch::unique_dim(
            torch::cat(configs_pool, /*dim=*/0),
            /*dim=*/0,
            /*sorted=*/false,
            /*return_inverse=*/true,
            /*return_counts=*/false
        );

        std::int64_t src_size = configs_i.value().size(0);
        std::int64_t pool_size = pool.size(0);

        auto pool_to_target = [&]() {
            auto src_to_pool = all_to_pool.index({torch::indexing::Slice(torch::indexing::None, src_size)});
            auto pool_to_target = torch::full({pool_size}, -1, torch::TensorOptions().dtype(torch::kInt64).device(device));
            pool_to_target.index_put_({src_to_pool}, torch::arange(src_size, torch::TensorOptions().dtype(torch::kInt64).device(device)));
            pool_to_target.index_put_(
                {pool_to_target == -1},
                torch::arange(src_size, pool_size, torch::TensorOptions().dtype(torch::kInt64).device(device))
            );
            return pool_to_target;
        }();

        auto target_index_j = [&]() {
            auto target_index_j_pool = std::vector<torch::Tensor>();
            auto begin = src_size;
            for (auto& outside : outsides) {
                auto& [index_i, index_j, coefs, configs_j] = outside;

                auto this_size = configs_j.size(0);
                auto segment = torch::indexing::Slice(begin, begin + this_size);
                begin += this_size;

                auto dst_to_pool = all_to_pool.index({segment});
                auto dst_to_target = pool_to_target.index({dst_to_pool});
                target_index_j_pool.push_back(dst_to_target.index({index_j}));
            }
            return torch::cat(target_index_j_pool, /*dim=*/0);
        }();

        auto target = [&]() {
            auto target = torch::empty_like(pool);
            target.index_put_({pool_to_target}, pool);
            return target;
        }();

        auto index_i = torch::cat(index_i_pool, /*dim=*/0);
        auto coefs = torch::cat(coefs_pool, /*dim=*/0);

        return Outside(index_i, target_index_j, coefs, target);
    } else {
        auto index_i_pool = std::vector<torch::Tensor>();
        auto index_j_pool = std::vector<torch::Tensor>();
        auto coefs_pool = std::vector<torch::Tensor>();
        auto configs_pool = std::vector<torch::Tensor>();

        for (auto& outside : outsides) {
            auto& [index_i, index_j, coefs, configs_j] = outside;
            index_i_pool.push_back(index_i);
            index_j_pool.push_back(index_j);
            coefs_pool.push_back(coefs);
            configs_pool.push_back(configs_j);
        }

        auto [target, all_to_target, nothing] = torch::unique_dim(
            torch::cat(configs_pool, /*dim=*/0),
            /*dim=*/0,
            /*sorted=*/false,
            /*return_inverse=*/true,
            /*return_counts=*/false
        );

        auto target_index_j = [&]() {
            auto target_index_j_pool = std::vector<torch::Tensor>();
            auto begin = 0;
            for (auto& outside : outsides) {
                auto& [index_i, index_j, coefs, configs_j] = outside;

                auto this_size = configs_j.size(0);
                auto segment = torch::indexing::Slice(begin, begin + this_size);
                begin += this_size;

                auto dst_to_target = all_to_target.index({segment});
                target_index_j_pool.push_back(dst_to_target.index({index_j}));
            }
            return torch::cat(target_index_j_pool, /*dim=*/0);
        }();

        auto index_i = torch::cat(index_i_pool, /*dim=*/0);
        auto coefs = torch::cat(coefs_pool, /*dim=*/0);

        return Outside(index_i, target_index_j, coefs, target);
    }
}

// Convert a Raw object to a Sparse object.
// Differently from former three formats, we need psi_i and configs_i to compute the result.
// The result of this function is v.conj() @ H when squared is false, and |v.^2| @ |H.^2| when squared is true.
auto raw_apply_outside(Raw raw, torch::Tensor psi_i, torch::Tensor configs_i, bool squared) -> Sparse {
    auto& [index_i, configs_j, coefs] = raw;
    std::int64_t count_j = configs_j.size(0);

    auto item = psi_i.index({index_i});
    auto item_real = item.index({torch::indexing::Slice(), 0});
    auto item_imag = item.index({torch::indexing::Slice(), 1});
    auto coefs_real = coefs.index({torch::indexing::Slice(), 0});
    auto coefs_imag = coefs.index({torch::indexing::Slice(), 1});
    if (squared) {
        // psi_j = (item.real ** 2 + item.imag ** 2) * (coefs.real ** 2 + coefs.imag ** 2)
        auto psi_j = (item_real.square() + item_imag.square()) * (coefs_real.square() + coefs_imag.square());

        return Sparse(psi_j.unsqueeze(-1), configs_j);
    } else {
        // psi_j = item.conj() * coefs
        auto psi_j_real = item_real * coefs_real + item_imag * coefs_imag;
        auto psi_j_imag = item_real * coefs_imag - item_imag * coefs_real;
        auto psi_j = torch::stack({psi_j_real, psi_j_imag}, /*dim=*/1);

        return Sparse(psi_j, configs_j);
    }
}

// Merge a collection of Sparse objects into a single Sparse object.
auto merge_apply_outside(std::vector<Sparse> sparses, std::optional<torch::Tensor> configs_i) -> Sparse {
    if (configs_i.has_value()) {
        auto device = configs_i.value().device();

        auto psi_j_pool = std::vector<torch::Tensor>();
        auto configs_j_pool = std::vector<torch::Tensor>();

        configs_j_pool.push_back(configs_i.value());
        for (auto& sparse : sparses) {
            auto& [psi_j, configs_j] = sparse;
            psi_j_pool.push_back(psi_j);
            configs_j_pool.push_back(configs_j);
        }

        auto [pool, all_to_pool, nothing] = torch::unique_dim(
            torch::cat(configs_j_pool, /*dim=*/0),
            /*dim=*/0,
            /*sorted=*/false,
            /*return_inverse=*/true,
            /*return_counts=*/false
        );

        std::int64_t src_size = configs_i.value().size(0);
        std::int64_t pool_size = pool.size(0);

        auto pool_to_target = [&]() {
            auto src_to_pool = all_to_pool.index({torch::indexing::Slice(torch::indexing::None, src_size)});
            auto pool_to_target = torch::full({pool_size}, -1, torch::TensorOptions().dtype(torch::kInt64).device(device));
            pool_to_target.index_put_({src_to_pool}, torch::arange(src_size, torch::TensorOptions().dtype(torch::kInt64).device(device)));
            pool_to_target.index_put_(
                {pool_to_target == -1},
                torch::arange(src_size, pool_size, torch::TensorOptions().dtype(torch::kInt64).device(device))
            );
            return pool_to_target;
        }();

        auto target = [&]() {
            auto target = torch::empty_like(pool);
            target.index_put_({pool_to_target}, pool);
            return target;
        }();

        auto psi_j_target = [&]() {
            auto all_to_target = pool_to_target.index({all_to_pool.index({torch::indexing::Slice(src_size)})});
            auto psi_j_all = torch::cat(psi_j_pool, /*dim=*/0);
            auto psi_j_target = torch::zeros({target.size(0), psi_j_all.size(1)}, psi_j_all.options());
            auto index = all_to_target.unsqueeze(1).expand({-1, psi_j_all.size(1)});
            psi_j_target.scatter_add_(/*dim=*/0, /*index=*/index, /*src=*/psi_j_all);
            return psi_j_target;
        }();

        return Sparse(psi_j_target, target);
    } else {
        auto psi_j_pool = std::vector<torch::Tensor>();
        auto configs_j_pool = std::vector<torch::Tensor>();

        for (auto& sparse : sparses) {
            auto& [psi_j, configs_j] = sparse;
            psi_j_pool.push_back(psi_j);
            configs_j_pool.push_back(configs_j);
        }

        auto [target, all_to_target, nothing] = torch::unique_dim(
            torch::cat(configs_j_pool, /*dim=*/0),
            /*dim=*/0,
            /*sorted=*/false,
            /*return_inverse=*/true,
            /*return_counts=*/false
        );

        auto psi_j_target = [&]() {
            auto psi_j_all = torch::cat(psi_j_pool, /*dim=*/0);
            auto psi_j_target = torch::zeros({target.size(0), psi_j_all.size(1)}, psi_j_all.options());
            auto index = all_to_target.unsqueeze(1).expand({-1, psi_j_all.size(1)});
            psi_j_target.scatter_add_(/*dim=*/0, /*index=*/index, /*src=*/psi_j_all);
            return psi_j_target;
        }();

        return Sparse(psi_j_target, target);
    }
}

#ifndef NQUBYTES
#define NQUBYTES 0
#endif

#if NQUBYTES == 0
PYBIND11_MODULE(qmb_collection, m) {
    m.def("merge_raw", merge_raw, py::arg("raws"));
    m.def("raw_to_inside", raw_to_inside, py::arg("raw"), py::arg("configs_i"));
    m.def("merge_inside", merge_inside, py::arg("insides"));
    m.def("raw_to_outside", raw_to_outside, py::arg("raw"), py::arg("configs_i"));
    m.def("merge_outside", merge_outside, py::arg("outsides"), py::arg("configs_i"));
    m.def("raw_apply_outside", raw_apply_outside, py::arg("raw"), py::arg("psi_i"), py::arg("configs_i"), py::arg("squared"));
    m.def("merge_apply_outside", merge_apply_outside, py::arg("applies"), py::arg("configs_i"));
}
#endif

// The merging of Sparse object is slow above, so we use torch kernels to merge them.
// The following code is the declaration of the kernels, which are:
// - sort_ : sort the given key(uint8[batch_size, n_qubits]) and value(float[batch_size, 1/2]) tensors by key.
// - merge : merge two key-value pairs into one.
// - reduce : reduce the key-value pairs, sum the values for each group of consecutive same keys.
// - ensure_ : ensure the first length_config config is in the front of key-value pairs.
// It should be noticed that user need to call key = [config, key] and value = [0, value] before calling ensure_.
// The ensure_ kernel will move the value of the first length_config config to the front of value and leave the origin value as 0.
// After moving, the rest of key-value pairs will be sorted by value(descending).

#ifndef QMB_LIBRARY_HELPER
#define QMB_LIBRARY_HELPER(x) qmb_collection_##x
#endif
#ifndef QMB_LIBRARY
#define QMB_LIBRARY(x) QMB_LIBRARY_HELPER(x)
#endif

#if NQUBYTES != 0
TORCH_LIBRARY_FRAGMENT(QMB_LIBRARY(NQUBYTES), m) {
    m.def("sort_(Tensor key, Tensor value) -> (Tensor, Tensor)");
    m.def("merge(Tensor key_1, Tensor value_1, Tensor key_2, Tensor value_2) -> (Tensor, Tensor)");
    m.def("reduce(Tensor key, Tensor value) -> (Tensor, Tensor)");
    m.def("ensure_(Tensor key, Tensor value, int length_config) -> (Tensor, Tensor)");
}
#endif

} // namespace qmb_collection

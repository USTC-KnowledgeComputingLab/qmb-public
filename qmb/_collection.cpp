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

using Raw = std::tuple<
    torch::Tensor, // index_i
    torch::Tensor, // configs_j
    torch::Tensor // coefs
    >;

using Inside = std::tuple<
    torch::Tensor, // index_i
    torch::Tensor, // index_j
    torch::Tensor // coefs
    >;

using Outside = std::tuple<
    torch::Tensor, // index_i
    torch::Tensor, // index_j
    torch::Tensor, // coefs
    torch::Tensor // configs_j
    >;

using Sparse = std::tuple<
    torch::Tensor, // psi_j
    torch::Tensor // configs_j
    >;

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

PYBIND11_MODULE(_collection, m) {
    m.def("merge_raw", merge_raw, py::arg("raws"));
    m.def("raw_to_inside", raw_to_inside, py::arg("raw"), py::arg("configs_i"));
    m.def("merge_inside", merge_inside, py::arg("insides"));
    m.def("raw_to_outside", raw_to_outside, py::arg("raw"), py::arg("configs_i"));
    m.def("merge_outside", merge_outside, py::arg("outsides"), py::arg("configs_i"));
    m.def("raw_apply_outside", raw_apply_outside, py::arg("raw"), py::arg("psi_i"), py::arg("configs_i"), py::arg("squared"));
    m.def("merge_apply_outside", merge_apply_outside, py::arg("applies"), py::arg("configs_i"));
}

} // namespace qmb_collection

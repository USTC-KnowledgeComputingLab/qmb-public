#include <array>
#include <cstdint>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>
#include <utility>
#include <vector>

#include "_binary_tree.hpp"

namespace py = pybind11;

// Hamiltonian handle for openfermion data.
// Every term of hamiltonian is operators less or equal than 4,
// so we use std::pair<Site, Type> to represent the term.
class Hamiltonian {
    using Coef = std::complex<double>;
    using Site = int16_t;
    using Type = int16_t; // 0 for empty, 1 for annihilation, 2 for creation
    using Op = std::pair<Site, Type>;
    using Ops = std::array<Op, 4>;
    using Term = std::pair<Ops, Coef>;
    std::vector<Term> terms;

    // Convert c++ vector to numpy array, with given shape.
    template<typename T>
    static py::array_t<T> vector_to_array(const std::vector<T>& vec, std::vector<int64_t> shape) {
        py::array_t<T> result(shape);
        auto result_buffer = result.request();
        T* ptr = static_cast<T*>(result_buffer.ptr);
        std::copy(vec.begin(), vec.end(), ptr);
        return result;
    }

  public:
    // Analyze openfermion data, which is converted from python,
    // It may be slow, but we only need to call this constructor once every process so it is ok.
    Hamiltonian(const std::vector<std::tuple<std::vector<std::pair<int, int>>, std::complex<double>>>& openfermion_hamiltonian) {
        for (const auto& [openfermion_ops, coef] : openfermion_hamiltonian) {
            Ops ops;
            size_t i = 0;
            for (; i < openfermion_ops.size(); ++i) {
                ops[i].first = openfermion_ops[i].first;
                ops[i].second = 1 + openfermion_ops[i].second; // in openfermion, 0 for annihilation, 1 for creation
            }
            for (; i < 4; ++i) {
                ops[i].second = 0; // 0 empty
            }
            terms.emplace_back(ops, coef);
        }
    }

    template<bool outside>
    auto call(const py::array_t<int64_t, py::array::c_style>& configs) {
        // Configs is usually a numpy array, with rank of 2.
        py::buffer_info configs_buf = configs.request();
        const int64_t batch = configs_buf.shape[0];
        const int64_t sites = configs_buf.shape[1];
        int64_t* configs_ptr = static_cast<int64_t*>(configs_buf.ptr);

        // config dict map every config to a index, default is -1 for missing configs.
        Tree<int64_t, -1> config_dict;
        // the prime configs count, which is at least batch size, since we will insert given configs first of all.
        int64_t prime_count = batch;
        // The prime configs array, used in outside mode.
        std::vector<int64_t> config_j_pool;
        // The indices_i_and_j and coefs is the main body of sparse matrix.
        std::vector<int64_t> indices_i_and_j;
        std::vector<std::complex<double>> coefs;
        // config j in temperary vector, allocate it here for better performance
        std::vector<int64_t> config_j(sites);

        // Set input configs index in configs dict.
        for (int64_t i = 0; i < batch; ++i) {
            config_dict.set(&configs_ptr[i * sites], &configs_ptr[(i + 1) * sites], i);
        }
        // In outside mode, we need prime config array, so create it by copy the given configuration as the first batch size configs.
        if constexpr (outside) {
            config_j_pool.resize(batch * sites);
            for (int64_t i = 0; i < batch * sites; ++i) {
                config_j_pool[i] = configs_ptr[i];
            }
        }

        // Loop over every batch and every hamiltonian term
        for (int64_t index_i = 0; index_i < batch; ++index_i) {
            for (const auto& [ops, coef] : terms) {
                // Prepare config j to be operated by hamiltonian term
                for (int64_t i = 0; i < sites; ++i) {
                    config_j[i] = configs_ptr[index_i * sites + i];
                }
                bool success = true;
                bool parity = false;
                // Apply operator one by one
                for (auto i = 4; i-- > 0;) {
                    auto [site, operation] = ops[i];
                    if (operation == 0) {
                        // Empty operator, nothing happens
                        continue;
                    } else if (operation == 1) {
                        // Annihilation operator
                        if (config_j[site] != 1) {
                            success = false;
                            break;
                        }
                        config_j[site] = 0;
                        if (std::accumulate(config_j.begin(), config_j.begin() + site, 0) % 2 == 1) {
                            parity ^= true;
                        }
                    } else {
                        // Creation operator
                        if (config_j[site] != 0) {
                            success = false;
                            break;
                        }
                        config_j[site] = 1;
                        if (std::accumulate(config_j.begin(), config_j.begin() + site, 0) % 2 == 1) {
                            parity ^= true;
                        }
                    }
                }

                if (success) {
                    // Success, insert this term to sparse matrix
                    // Find the index j first
                    int64_t index_j = config_dict.get(config_j.begin(), config_j.end());
                    if (index_j == -1) {
                        // If index j not found
                        if constexpr (outside) {
                            // Insert it to prime config pool in outside mode
                            int64_t size = config_j_pool.size();
                            config_j_pool.resize(size + sites);
                            for (int64_t i = 0; i < sites; ++i) {
                                config_j_pool[i + size] = config_j[i];
                            }
                            index_j = prime_count;
                            config_dict.set(config_j.begin(), config_j.end(), prime_count++);
                        } else {
                            // Continue to next batch or hamiltonian term in inside mode.
                            continue;
                        }
                    }
                    indices_i_and_j.push_back(index_i);
                    indices_i_and_j.push_back(index_j);
                    coefs.push_back(parity ? -coef : +coef);
                }
            }
        }

        int64_t term_count = coefs.size();
        if constexpr (outside) {
            return std::make_tuple(
                vector_to_array(indices_i_and_j, {term_count, 2}),
                vector_to_array(coefs, {term_count}),
                vector_to_array(config_j_pool, {prime_count, sites})
            );
        } else {
            return std::make_tuple(vector_to_array(indices_i_and_j, {term_count, 2}), vector_to_array(coefs, {term_count}));
        }
    }
};

PYBIND11_MODULE(_openfermion, m) {
    py::class_<Hamiltonian>(m, "Hamiltonian", py::module_local())
        .def(py::init<std::vector<std::tuple<std::vector<std::pair<int, int>>, std::complex<double>>>>(), py::arg("openfermion_hamiltonian"))
        .def("inside", &Hamiltonian::call<false>, py::arg("configs"))
        .def("outside", &Hamiltonian::call<true>, py::arg("configs"));
}

#include <array>
#include <cstdint>
#include <memory>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

template<typename T, T miss>
class Tree {
    std::unique_ptr<Tree<T, miss>> _left;
    std::unique_ptr<Tree<T, miss>> _right;
    std::unique_ptr<T> _value;

    T& value() {
        if (!_value) {
            _value = std::make_unique<T>();
        }
        return *_value;
    }

    Tree<T, miss>& left() {
        if (!_left) {
            _left = std::make_unique<Tree<T, miss>>();
        }
        return *_left;
    }

    Tree<T, miss>& right() {
        if (!_right) {
            _right = std::make_unique<Tree<T, miss>>();
        }
        return *_right;
    }

  public:
    template<typename It>
    void set(It begin, It end, T v) {
        if (begin == end) {
            value() = v;
        } else {
            if (*(begin++)) {
                right().set(begin, end, v);
            } else {
                left().set(begin, end, v);
            }
        }
    }

    template<typename It>
    T get(It begin, It end) {
        if (begin == end) {
            return value();
        } else {
            if (*(begin++)) {
                if (_right) {
                    return right().get(begin, end);
                } else {
                    return miss;
                }
            } else {
                if (_left) {
                    return left().get(begin, end);
                } else {
                    return miss;
                }
            }
        }
    }
};

namespace py = pybind11;

class Hamiltonian {
    using Coef = std::complex<double>;
    using Site = int16_t;
    using Type = int16_t; // 0 for empty, 1 for annihilation, 2 for creation
    using Op = std::pair<Site, Type>;
    using Ops = std::array<Op, 4>;
    using Term = std::pair<Ops, Coef>;
    std::vector<Term> terms;

    template<typename T>
    static py::array_t<T> vector_to_array(const std::vector<T>& vec, std::vector<int64_t> shape) {
        py::array_t<T> result(shape);
        auto result_buffer = result.request();
        T* ptr = static_cast<T*>(result_buffer.ptr);
        std::copy(vec.begin(), vec.end(), ptr);
        return result;
    }

  public:
    Hamiltonian(const std::vector<std::tuple<std::vector<std::pair<int, int>>, std::complex<double>>>& openfermion_hamiltonian) {
        for (const auto& [openfermion_ops, coef] : openfermion_hamiltonian) {
            Ops ops;
            size_t i = 0;
            for (; i < openfermion_ops.size(); i++) {
                ops[i].first = openfermion_ops[i].first;
                ops[i].second = 1 + openfermion_ops[i].second; // 0 for annihilation, 1 for creation
            }
            for (; i < 4; i++) {
                ops[i].second = 0; // 0 empty
            }
            terms.emplace_back(ops, coef);
        }
    }

    template<bool outside>
    auto call(const py::array_t<int64_t, py::array::c_style>& configs) {
        py::buffer_info configs_buf = configs.request();
        const int64_t batch = configs_buf.shape[0];
        const int64_t sites = configs_buf.shape[1];
        int64_t* configs_ptr = static_cast<int64_t*>(configs_buf.ptr);

        Tree<int64_t, -1> config_dict;
        int64_t prime_count = batch;
        std::vector<int64_t> indices_i_and_j;
        std::vector<std::complex<double>> coefs;
        std::vector<int64_t> config_j_pool;
        std::vector<int64_t> config_j(sites);

        for (int64_t i = 0; i < batch; ++i) {
            config_dict.set(&configs_ptr[i * sites], &configs_ptr[(i + 1) * sites], i);
        }
        if constexpr (outside) {
            config_j_pool.resize(batch * sites);
            for (int64_t i = 0; i < batch * sites; ++i) {
                config_j_pool[i] = configs_ptr[i];
            }
        }

        for (int64_t index_i = 0; index_i < batch; ++index_i) {
            for (const auto& [ops, coef] : terms) {
                for (int64_t i = 0; i < sites; i++) {
                    config_j[i] = configs_ptr[index_i * sites + i];
                }
                bool success = true;
                bool parity = false;

                for (auto i = 4; i-- > 0;) {
                    auto [site, operation] = ops[i];
                    if (operation == 0) {
                        continue;
                    } else if (operation == 1) {
                        if (config_j[site] != 1) {
                            success = false;
                            break;
                        }
                        config_j[site] = 0;
                        if (std::accumulate(config_j.begin(), config_j.begin() + site, 0) % 2 == 1) {
                            parity ^= true;
                        }
                    } else {
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
                    int64_t index_j = config_dict.get(config_j.begin(), config_j.end());
                    if (index_j == -1) {
                        if constexpr (outside) {
                            int64_t size = config_j_pool.size();
                            config_j_pool.resize(size + sites);
                            for (int64_t i = 0; i < sites; i++) {
                                config_j_pool[i + size] = config_j[i];
                            }
                            index_j = prime_count;
                            config_dict.set(config_j.begin(), config_j.end(), prime_count++);
                        } else {
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
            return py::make_tuple(
                vector_to_array(indices_i_and_j, {term_count, 2}),
                vector_to_array(coefs, {term_count}),
                vector_to_array(config_j_pool, {prime_count, sites})
            );
        } else {
            return py::make_tuple(vector_to_array(indices_i_and_j, {term_count, 2}), vector_to_array(coefs, {term_count}));
        }
    }
};

PYBIND11_MODULE(openfermion_to_sparse, m) {
    py::class_<Hamiltonian>(m, "Hamiltonian")
        .def(py::init<std::vector<std::tuple<std::vector<std::pair<int, int>>, std::complex<double>>>>())
        .def("inside", &Hamiltonian::call<false>)
        .def("outside", &Hamiltonian::call<true>);
}

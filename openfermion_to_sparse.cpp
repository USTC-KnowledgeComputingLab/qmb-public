#include <memory>
#include <numeric>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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

std::tuple<std::vector<int>, std::vector<int>, std::vector<std::complex<double>>> openfermion_to_sparse(
    const std::vector<std::tuple<std::vector<std::pair<int, int>>, std::complex<double>>>& openfermion_hamiltonian,
    const std::vector<std::vector<int>>& configs
) {
    Tree<int64_t, -1> config_dict;
    for (size_t i = 0; i < configs.size(); ++i) {
        config_dict.set(configs[i].begin(), configs[i].end(), i);
    }

    std::vector<int> indices_i;
    std::vector<int> indices_j;
    std::vector<std::complex<double>> values;

    std::vector<int> config_j;
    for (size_t index_i = 0; index_i < configs.size(); ++index_i) {
        const std::vector<int>& config_i = configs[index_i];
        for (const auto& [key, value] : openfermion_hamiltonian) {
            config_j = config_i;
            std::complex<double> current_value = value;
            bool success = true;

            for (auto it = key.rbegin(); it != key.rend(); ++it) {
                int site = it->first;
                int operation = it->second;

                if (operation == 0) {
                    if (config_j[site] != 1) {
                        success = false;
                        break;
                    }
                    config_j[site] = 0;
                    if (std::accumulate(config_j.begin(), config_j.begin() + site, 0) % 2 == 1) {
                        current_value = -current_value;
                    }
                } else {
                    if (config_j[site] != 0) {
                        success = false;
                        break;
                    }
                    config_j[site] = 1;
                    if (std::accumulate(config_j.begin(), config_j.begin() + site, 0) % 2 == 1) {
                        current_value = -current_value;
                    }
                }
            }

            if (success) {
                auto index_j = config_dict.get(config_j.begin(), config_j.end());
                if (index_j != -1) {
                    indices_i.push_back(index_i);
                    indices_j.push_back(index_j);
                    values.push_back(current_value);
                }
            }
        }
    }

    return std::make_tuple(indices_i, indices_j, values);
}

PYBIND11_MODULE(openfermion_to_sparse, m) {
    m.def("openfermion_to_sparse", &openfermion_to_sparse, "Convert OpenFermion Hamiltonian to sparse matrix format");
}

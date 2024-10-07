#include <cstdint>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>
#include <vector>

#include "_binary_tree.hpp"

namespace py = pybind11;

// Hamiltonian handle.
class Hamiltonian {
    using Coef = std::complex<double>;

    Coef X, Y, Z, XX, YY, ZZ;

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
    Hamiltonian(Coef _X, Coef _Y, Coef _Z, Coef _XX, Coef _YY, Coef _ZZ) : X(_X), Y(_Y), Z(_Z), XX(_XX), YY(_YY), ZZ(_ZZ) { }

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
            for (int64_t type = 0; type < 6; ++type) {
                for (int64_t site = 0; site < sites; ++site) {
                    // type 0, 1, 2: X, Y, Z on site
                    // type 3, 4, 5: XX, YY, ZZ on site and site-1
                    if ((type >= 3) && (site == 0)) {
                        continue;
                    }
                    Coef coef;
                    switch (type) {
                    case (0):
                        coef = X;
                        break;
                    case (1):
                        coef = Y;
                        break;
                    case (2):
                        coef = Z;
                        break;
                    case (3):
                        coef = XX;
                        break;
                    case (4):
                        coef = YY;
                        break;
                    case (5):
                        coef = ZZ;
                        break;
                    }
                    if (coef == Coef()) {
                        continue;
                    }
                    // Prepare config j to be operated by hamiltonian term
                    for (int64_t i = 0; i < sites; ++i) {
                        config_j[i] = configs_ptr[index_i * sites + i];
                    }
                    Coef param;
                    switch (type) {
                    case (0):
                        param = 1;
                        config_j[site] = 1 - config_j[site];
                        break;
                    case (1):
                        param = config_j[site] == 0 ? Coef(0, +1) : Coef(0, -1);
                        config_j[site] = 1 - config_j[site];
                        break;
                    case (2):
                        param = config_j[site] == 0 ? +1 : -1;
                        break;
                    case (3):
                        param = 1;
                        config_j[site] = 1 - config_j[site];
                        config_j[site - 1] = 1 - config_j[site - 1];
                        break;
                    case (4):
                        param = config_j[site - 1] == config_j[site] ? -1 : +1;
                        config_j[site] = 1 - config_j[site];
                        config_j[site - 1] = 1 - config_j[site - 1];
                        break;
                    case (5):
                        param = config_j[site - 1] == config_j[site] ? +1 : -1;
                        break;
                    }

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
                    coefs.push_back(coef * param);
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

PYBIND11_MODULE(_ising, m) {
    py::class_<Hamiltonian>(m, "Hamiltonian", py::module_local())
        .def(
            py::init<
                std::complex<double>,
                std::complex<double>,
                std::complex<double>,
                std::complex<double>,
                std::complex<double>,
                std::complex<double>>(),
            py::arg("X"),
            py::arg("Y"),
            py::arg("Z"),
            py::arg("XX"),
            py::arg("YY"),
            py::arg("ZZ")
        )
        .def("inside", &Hamiltonian::call<false>, py::arg("configs"))
        .def("outside", &Hamiltonian::call<true>, py::arg("configs"));
}

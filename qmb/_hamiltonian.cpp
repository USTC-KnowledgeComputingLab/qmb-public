// This file is dedicated to processing iterations over Hamiltonian terms based on provided configurations.
// It consists of two primary components:
// 1. The `prepare` function is responsible for parsing a raw Python dictionary and transforming it into a structured tuple of tensors.
//    This tuple is subsequently stored on the Python side and utilized in subsequent calls to the PyTorch operators for further processing.
//    The tuple comprises three tensors, each serving a specific purpose:
//    - `site`: An int16 tensor of shape [term_number, max_op_number], representing the site indices of the operators for each term in the
//    Hamiltonian.
//    - `kind`: An int8 tensor of shape [term_number, max_op_number], representing the type of operator for each term in the Hamiltonian. The values
//    are encoded as follows:
//      - 0: Annihilation operator
//      - 1: Creation operator
//      - 2: Empty (identity operator)
//    - `coef`: A float64 tensor of shape [term_number, 2], representing the coefficients of each term. The first element in each pair denotes the
//    real part, while the second element denotes the imaginary part.
// 2. A suite of PyTorch operators, including `fermi` and `bose2`, are designed to process the tuples of tensors produced by the `prepare` function in
//    conjunction with a specified sequence of configurations.
//    The configuration sequence is represented as an int8 tensor with dimensions [batch_size, n_qubits], encapsulating the state configurations of
//    the system.
//    The function sequentially applies operators for each term based on the provided configurations, resulting in three essential tensors:
//    - `index_i`: An int64 tensor of shape [valid_size], indexing the resultant terms.
//    - `configs_j`: An int8 tensor of shape [valid_size, n_qubits], detailing the configurations of the resultant terms.
//    - `coefs`: A float64 tensor of shape [valid_size, 2], encoding the coefficients (both real and imaginary parts) of the resultant terms.
//    Here, `valid_size` signifies the count of non-zero terms post-iteration over the Hamiltonian based on the provided configurations.
// The `max_op_number` template argument specifies the maximum number of operators per term, typically set to 4 for 2-body interactions.
// This file defines the `prepare` function and declares the PyTorch operators, with their specific device implementations located in separate files.

#include <torch/extension.h>

namespace qmb_hamiltonian {

// The `prepare` function is responsible for parsing a raw Python dictionary representing Hamiltonian terms
// and transforming it into a structured tuple of tensors. This tuple is then stored on the Python side
// and utilized in subsequent calls to the PyTorch operators for further processing.
//
// The function takes a Python dictionary `hamiltonian` as input, where each key-value pair represents a term
// in the Hamiltonian. The key is a tuple of tuples, where each inner tuple contains two elements:
// - The first element is an integer representing the site index of the operator.
// - The second element is an integer representing the type of operator (0 for annihilation, 1 for creation).
// The value is either a float or a complex number representing the coefficient of the term.
//
// The function processes the dictionary and constructs three tensors:
// - `site`: An int16 tensor of shape [term_number, max_op_number], representing the site indices of the operators for each term.
// - `kind`: An int8 tensor of shape [term_number, max_op_number], representing the type of operator for each term.
// - `coef`: A float64 tensor of shape [term_number, 2], representing the coefficients of each term, with two elements for real and imaginary parts.
template<std::int64_t max_op_number>
auto prepare(py::dict hamiltonian) {
    std::int64_t term_number = hamiltonian.size();

    auto site = torch::empty({term_number, max_op_number}, torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU));
    // No need to initialize
    auto kind = torch::full({term_number, max_op_number}, 2, torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU));
    // Initialize to 2 for identity
    auto coef = torch::empty({term_number, 2}, torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU));
    // No need to initialize

    auto site_accessor = site.template accessor<std::int16_t, 2>();
    auto kind_accessor = kind.template accessor<std::int8_t, 2>();
    auto coef_accessor = coef.template accessor<double, 2>();

    std::int64_t index = 0;
    for (auto item : hamiltonian) {
        auto key = item.first.cast<py::tuple>();
        auto value_is_float = py::isinstance<py::float_>(item.second);
        auto value = value_is_float ? std::complex<double>(item.second.cast<double>()) : item.second.cast<std::complex<double>>();

        std::int64_t op_number = key.size();
        for (auto i = 0; i < op_number; ++i) {
            site_accessor[index][i] = key[i].cast<py::tuple>()[0].cast<std::int16_t>();
            kind_accessor[index][i] = key[i].cast<py::tuple>()[1].cast<std::int8_t>();
        }

        coef_accessor[index][0] = value.real();
        coef_accessor[index][1] = value.imag();

        ++index;
    }

    return std::make_tuple(site, kind, coef);
}

// Expose the `prepare` function to Python.
PYBIND11_MODULE(_hamiltonian, m) {
    m.def("prepare", prepare</*max_op_number=*/4>, py::arg("hamiltonian"));
}

// This section declares the custom PyTorch operators such as `fermi` and `bose2`.
// These operators are designed to work in conjunction with the structured tuple of tensors produced by the `prepare` function.
// The operators take the following inputs:
// - `configs_i`: An int8 tensor of shape [batch_size, n_qubits], representing the given configurations of the system.
// - `site`: An int16 tensor of shape [term_number, max_op_number], produced by the `prepare` function.
// - `kind`: An int8 tensor of shape [term_number, max_op_number], produced by the `prepare` function.
// - `coef`: A float64 tensor of shape [term_number, 2], produced by the `prepare` function.
//
// The operators sequentially apply the operators for each term based on the provided configurations, resulting in the following outputs:
// - `index_i`: An int64 tensor of shape [valid_size], indexing the original terms.
// - `configs_j`: An int8 tensor of shape [valid_size, n_qubits], detailing the configurations of the results.
// - `coefs`: A float64 tensor of shape [valid_size, 2], encoding the coefficients (both real and imaginary parts) of the terms.
// Here, `valid_size` signifies the count of non-zero terms post-iteration over the Hamiltonian based on the provided configurations.
TORCH_LIBRARY(_hamiltonian, m) {
    m.def("fermi(Tensor configs_i, Tensor site, Tensor kind, Tensor coef) -> (Tensor, Tensor, Tensor)");
    m.def("bose2(Tensor configs_i, Tensor site, Tensor kind, Tensor coef) -> (Tensor, Tensor, Tensor)");
}

} // namespace qmb_hamiltonian

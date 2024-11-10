"""
This file contains common steps used in other scripts.
"""

import logging
import torch
from .model_dict import ModelProto
from .lobpcg import lobpcg


def extend_with_select(
    model: ModelProto,
    configs_core: torch.Tensor,
    psi_core: torch.Tensor,
    count_selected: int,
) -> tuple[
        torch.Tensor,
        torch.Tensor,
]:
    """
    Extend configs_core based on the model, calculate their importance based on psi_core and select them based on count_selected.
    """

    logging.info("Starting extend with selection process")

    count_core = len(configs_core)
    logging.info("Number of core configurations: %d", count_core)

    logging.info("Calculating extended configurations")
    indices_i_and_j, values, configs_extended = model.outside(configs_core)
    logging.info("Extended configurations have been created")
    count_extended = len(configs_extended)
    logging.info("Number of extended configurations: %d", count_extended)

    logging.info("Converting sparse matrix data into a sparse tensor.")
    hamiltonian = torch.sparse_coo_tensor(indices_i_and_j.T, values, [count_core, count_extended], dtype=torch.complex128).to_sparse_csr()
    del indices_i_and_j
    del values
    logging.info("Sparse extending Hamiltonian matrix has been created")

    logging.info("Estimating the importance of extended configurations")
    importance = (psi_core.conj() * psi_core).abs() @ (hamiltonian.conj() * hamiltonian).abs()
    del hamiltonian
    importance[:count_core] += importance.max()
    logging.info("Importance of extended configurations has been calculated")

    logging.info("Selecting extended configurations based on importance")
    selected_indices = importance.sort(descending=True).indices[:count_selected].sort().values
    del importance
    logging.info("Indices for selected extended configurations have been prepared")

    logging.info("Selecting extended configurations")
    configs_extended = configs_extended[selected_indices]
    del selected_indices
    logging.info("Extended configurations have been selected")
    count_extended = len(configs_extended)
    logging.info("Number of selected extended configurations: %d", count_extended)

    logging.info("Preparing initial amplitudes for future use")
    psi_extended = torch.cat([psi_core, torch.zeros([count_extended - count_core], dtype=psi_core.dtype, device=psi_core.device)], dim=0).view([-1, 1])
    logging.info("Initial amplitudes for future use has been created")

    logging.info("Extend with selection process completed")

    return configs_extended, psi_extended


def lobpcg_process(
    model: ModelProto,
    configs: torch.Tensor,
    psi: torch.Tensor,
) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
]:
    """
    Perform LOBPCG on the configurations.
    """

    logging.info("Starting LOBPCG process on the given configurations with prior amplitudes.")

    count = len(configs)
    logging.info("Total number of configurations: %d", count)

    logging.info("Calculating sparse data for the Hamiltonian matrix on the configurations.")
    indices_i_and_j, values = model.inside(configs)
    logging.info("Converting sparse matrix data into a sparse tensor.")
    hamiltonian = torch.sparse_coo_tensor(indices_i_and_j.T, values, [count, count], dtype=torch.complex128).to_sparse_csr()
    del indices_i_and_j
    del values
    logging.info("Sparse Hamiltonian matrix on configurations has been created.")

    logging.info("Calculating the minimum energy eigenvalue on the configurations.")
    energy, psi = lobpcg(hamiltonian, psi.view([-1, 1]), maxiter=1024)
    psi = psi.flatten()
    logging.info("Energy eigenvalue on configurations: %.10f, Reference energy: %.10f, Energy error: %.10f", energy.item(), model.ref_energy, energy.item() - model.ref_energy)

    logging.info("LOBPCG process completed.")

    return energy, hamiltonian, psi


def select_by_lobpcg(
    model: ModelProto,
    configs: torch.Tensor,
    psi: torch.Tensor,
    count_selected: int,
) -> tuple[
        torch.Tensor,
        torch.Tensor,
]:
    """
    Select the most important configurations based on the solution calculated by LOBPCG.
    """

    logging.info("Starting LOBPCG-based selection process.")

    _, _, psi = lobpcg_process(model, configs, psi)

    logging.info("Identifying the indices of the most significant configurations.")
    indices = torch.argsort(psi.abs())[-count_selected:]
    logging.info("Indices of the most significant configurations have been identified.")

    logging.info("Refining configurations to include only the most significant ones.")
    configs = configs[indices]
    psi = psi[indices]
    del indices
    logging.info("Configurations have been refined to include only the most significant ones.")

    logging.info("LOBPCG-based selection process completed successfully.")

    return configs, psi

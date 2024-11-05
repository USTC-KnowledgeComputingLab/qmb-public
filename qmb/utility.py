import logging
import torch
from .lobpcg import lobpcg


def extend_and_select(model, configs_core, psi_core, count_selected):
    # Extend configs_core based on the model, calculate their importance based on psi_core and select them based on count_selected.

    count_core = len(configs_core)
    logging.info("count of core configuration is %d", count_core)

    logging.info("calculate extended configurations")
    indices_i_and_j, values, configs_extended = model.outside(configs_core)
    logging.info("extended configurations created")
    count_extended = len(configs_extended)
    logging.info("count of extended configurations count is %d", count_extended)

    logging.info("converting sparse extending matrix data to sparse matrix")
    hamiltonian = torch.sparse_coo_tensor(indices_i_and_j.T, values, [count_core, count_extended], dtype=torch.complex128).to_sparse_csr()
    logging.info("sparse extending matrix created")

    logging.info("estimating the importance of extended configurations")
    importance = (psi_core.conj() * psi_core).abs() @ (hamiltonian.conj() * hamiltonian).abs()
    importance[:count_core] += importance.max()
    logging.info("importance of extended configurations created")

    logging.info("selecting extended configurations by importance")
    selected_indices = importance.sort(descending=True).indices[:count_selected].sort().values
    logging.info("extended configurations selected indices prepared")

    logging.info("selecting extended configurations")
    configs_extended = configs_extended[selected_indices]
    logging.info("extended configurations selected")
    count_extended = len(configs_extended)
    logging.info("selected extended configurations count is %d", count_extended)

    logging.info("preparing initial psi used in future")
    psi_extended = torch.cat([psi_core, torch.zeros([count_extended - count_core], dtype=psi_core.dtype, device=psi_core.device)], dim=0).view([-1, 1])
    logging.info("initial psi used in future has been created")

    return configs_extended, psi_extended


def lobpcg_and_select(model, configs, psi, count_selected=None):
    # Perform LOBPCG on the configurations and select the most important ones.

    count = len(configs)
    logging.info("count of configuration is %d", count)

    logging.info("calculating sparse data of hamiltonian on configurations")
    indices_i_and_j, values = model.inside(configs)
    logging.info("converting sparse matrix data to sparse matrix")
    hamiltonian = torch.sparse_coo_tensor(indices_i_and_j.T, values, [count, count], dtype=torch.complex128).to_sparse_csr()
    logging.info("sparse matrix on configurations created")

    logging.info("calculating minimum energy on configurations")
    energy, psi = lobpcg(hamiltonian, psi.view([-1, 1]), maxiter=1024)
    psi = psi.flatten()
    logging.info("energy on configurations is %.10f, ref energy is %.10f, error is %.10f", energy.item(), model.ref_energy, energy.item() - model.ref_energy)

    if count_selected is not None:
        logging.info("calculating indices of new configurations")
        indices = torch.argsort(psi.abs())[-count_selected:]
        logging.info("indices of new configurations has been obtained")

        logging.info("update new configurations")
        configs = configs[indices]
        psi = psi[indices]
        logging.info("new configurations has been updated")

        return None, None, configs, psi
    else:
        return energy, hamiltonian, configs, psi

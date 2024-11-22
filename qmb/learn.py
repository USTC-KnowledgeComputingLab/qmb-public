"""
This file implements a two-step optimization process for solving quantum many-body problems.
"""

import logging
import typing
import dataclasses
import torch
import tyro
from . import losses
from .common import CommonConfig
from .model_dict import ModelProto
from .subcommand_dict import subcommand_dict
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
    psi_extended = torch.cat([psi_core, torch.zeros([count_extended - count_core], dtype=psi_core.dtype, device=psi_core.device)], dim=0)
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


def _union(
    configs_a: torch.Tensor,
    configs_b: torch.Tensor,
) -> torch.Tensor:
    # Merge two sets of configurations and their corresponding psi values, ensuring that in case of duplicates, the psi values from the second set (psi_b) are retained.
    configs_both = torch.cat([configs_a, configs_b], dim=0)
    configs_result = torch.unique(configs_both, dim=0)
    return configs_result


def _subtraction(
    configs_a: torch.Tensor,
    configs_b: torch.Tensor,
) -> torch.Tensor:
    # Subtract the configurations in configs_b from configs_a, retaining the psi values from psi_a for the remaining configurations.
    count_a = len(configs_a)
    configs_both = torch.cat([configs_a, configs_b], dim=0)
    configs_result, both_to_result = torch.unique(configs_both, dim=0, return_inverse=True)
    a_to_result = both_to_result[:count_a]
    b_to_result = both_to_result[count_a:]
    result_in_b = torch.zeros(len(configs_result), device=configs_result.device, dtype=torch.bool)
    result_in_b[b_to_result] = True
    a_not_in_b = torch.logical_not(result_in_b[a_to_result])
    return configs_a[a_not_in_b]


@dataclasses.dataclass
class LearnConfig:
    """
    The two-step optimization process for solving quantum many-body problems.
    """

    # pylint: disable=too-many-instance-attributes

    common: typing.Annotated[CommonConfig, tyro.conf.OmitArgPrefixes]

    # The sampling count
    sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-n"])] = 4000
    # The learning rate for the local optimizer
    learning_rate: typing.Annotated[float, tyro.conf.arg(aliases=["-r"], help_behavior_hint="(default: 1e-3 for Adam, 1 for LBFGS)")] = -1
    # The number of steps for the local optimizer
    local_step: typing.Annotated[int, tyro.conf.arg(aliases=["-s"], help_behavior_hint="(default: 1000 for Adam, 400 for LBFGS)")] = -1
    # The early break loss threshold for local optimization
    local_loss: typing.Annotated[float, tyro.conf.arg(aliases=["-t"])] = 1e-8
    # The number of psi values to log after local optimization
    logging_psi: typing.Annotated[int, tyro.conf.arg(aliases=["-p"])] = 30
    # The name of the loss function to use
    loss_name: typing.Annotated[str, tyro.conf.arg(aliases=["-l"])] = "log"
    # Whether to use LBFGS instead of Adam
    use_lbfgs: typing.Annotated[bool, tyro.conf.arg(aliases=["-2"])] = False
    # The number of post-sampling iterations
    post_sampling_iteration: typing.Annotated[int, tyro.conf.arg(aliases=["-i"])] = 0
    # The number of configurations to sample during post-sampling
    post_sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-c"])] = 50000

    def __post_init__(self) -> None:
        if self.learning_rate == -1:
            self.learning_rate = 1 if self.use_lbfgs else 1e-3
        if self.local_step == -1:
            self.local_step = 400 if self.use_lbfgs else 1000

    def main(self) -> None:
        """
        The main function of the two-step optimization process.
        """
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements

        model, network = self.common.main()

        logging.info(
            "Arguments Summary: "
            "Sampling Count: %d, "
            "Learning Rate: %.10f, "
            "Local Step: %d, "
            "Local Loss Threshold: %.10f, "
            "Logging Psi Count: %d, "
            "Loss Function: %s, "
            "Using LBFGS: %s, "
            "Post Sampling Iteration: %d, "
            "Post Sampling Count: %d",
            self.sampling_count,
            self.learning_rate,
            self.local_step,
            self.local_loss,
            self.logging_psi,
            self.loss_name,
            "Yes" if self.use_lbfgs else "No",
            self.post_sampling_iteration,
            self.post_sampling_count,
        )

        while True:
            logging.info("Starting a new optimization cycle")

            logging.info("Sampling configurations")
            configs, psi, _, _ = network.generate_unique(self.sampling_count)
            logging.info("Sampling completed")

            if self.post_sampling_iteration != 0:
                logging.info("Starting post-sampling process")
                logging.info("Extending and selecting %d times with a temporary sampling count of %d", self.post_sampling_iteration, self.post_sampling_count)
                configs_all = configs
                for i in range(self.post_sampling_iteration):
                    logging.info("Performing extension, iteration %d", i)
                    configs, psi = extend_with_select(model, configs, psi, self.post_sampling_count)
                    configs_all = _union(configs_all, configs)
                    if i != self.post_sampling_iteration - 1:
                        logging.info("Performing selection, iteration %d", i)
                        configs, psi = select_by_lobpcg(model, configs, psi, self.sampling_count)
                    else:
                        logging.info("Skipping selection on the last iteration")
                logging.info("Extension and selection loop concluded")
                configs_too_small = _subtraction(configs_all, configs)
                logging.info("Post-sampling process successfully completed")

            logging.info("Solving the equation within the configuration subspace")
            target_energy, hamiltonian, targets = lobpcg_process(model, configs, psi)
            targets = targets.view([-1])
            max_index = targets.abs().argmax()
            targets = targets / targets[max_index]
            logging.info("Target within the subspace has been calculated")

            loss_func: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = getattr(losses, self.loss_name)

            optimizer: torch.optim.Optimizer
            if self.use_lbfgs:
                optimizer = torch.optim.LBFGS(network.parameters(), lr=self.learning_rate)
            else:
                optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                loss = torch.tensor(0)

                amplitudes = network(configs)
                amplitudes = amplitudes / amplitudes[max_index]
                loss_main = loss_func(amplitudes, targets)
                loss_main.backward()  # type: ignore[no-untyped-call]
                loss = loss + loss_main.item()
                min_index = amplitudes.abs().argmin()

                if self.post_sampling_iteration != 0:
                    for j in range(0, len(configs_too_small), len(configs)):
                        min_amplitudes, max_amplitudes = network(configs[[min_index.item(), max_index.item()]]).abs()  # type: ignore[assignment]
                        min_amplitudes = min_amplitudes / max_amplitudes
                        amplitudes_too_small = network(configs_too_small[j:j + len(configs)]).abs() / max_amplitudes
                        targets_too_small = torch.where(amplitudes_too_small > min_amplitudes, min_amplitudes, amplitudes_too_small)
                        loss_too_small = loss_func(amplitudes_too_small, targets_too_small)
                        loss_too_small = loss_too_small * (len(amplitudes_too_small) / len(configs))
                        loss_too_small.backward()  # type: ignore[no-untyped-call]
                        loss = loss + loss_too_small.item()

                loss.amplitudes = amplitudes.detach()  # type: ignore[attr-defined]
                return loss

            logging.info("Starting local optimization process")
            loss: torch.Tensor
            for i in range(self.local_step):
                loss = optimizer.step(closure)  # type: ignore[assignment,arg-type]
                logging.info("Local optimization in progress, step %d, current loss: %.10f", i, loss.item())
                if loss < self.local_loss:
                    logging.info("Local optimization halted as the loss threshold has been met")
                    break

            logging.info("Local optimization process completed")

            logging.info("Saving model checkpoint")
            torch.save(network.state_dict(), f"{self.common.checkpoint_path}/{self.common.job_name}.pt")
            logging.info("Checkpoint successfully saved")

            logging.info("Current optimization cycle completed")

            loss = typing.cast(torch.Tensor, torch.enable_grad(closure)())  # type: ignore[no-untyped-call,call-arg]
            amplitudes: torch.Tensor = loss.amplitudes  # type: ignore[attr-defined]
            final_energy = ((amplitudes.conj() @ (hamiltonian @ amplitudes)) / (amplitudes.conj() @ amplitudes)).real
            logging.info(
                "Loss during local optimization: %.10f, Final energy: %.10f, Target energy: %.10f, Reference energy: %.10f, Final error: %.10f",
                loss.item(),
                final_energy.item(),
                target_energy.item(),
                model.ref_energy,
                final_energy.item() - model.ref_energy,
            )
            logging.info("Displaying the largest amplitudes")
            indices = targets.abs().sort(descending=True).indices
            for index in indices[:self.logging_psi]:
                this_config = "".join(f"{i:08b}" for i in configs[index].cpu().numpy())
                logging.info("Configuration: %s, Target amplitude: %s, Final amplitude: %s", this_config, f"{targets[index].item():.8f}", f"{amplitudes[index].item():.8f}")


subcommand_dict["learn"] = LearnConfig

"""
This file implements a two-step optimization process for solving quantum many-body problems.
"""

import copy
import logging
import typing
import dataclasses
import torch
import torch.utils.tensorboard
import tyro
from . import losses
from .common import CommonConfig
from .model_dict import ModelProto
from .subcommand_dict import subcommand_dict
from .lobpcg import lobpcg
from .optimizer import initialize_optimizer, scale_learning_rate


def _extend_with_select(
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

    configs_selected = model.apply_outside(psi_core, configs_core, True, count_selected)
    count_selected = len(configs_selected)
    psi_selected = torch.nn.functional.pad(psi_core, (0, count_selected - count_core))
    logging.info("Extend with selection process completed, number of selected configurations: %d", count_selected)
    return configs_selected, psi_selected


def _inside_hamiltonian(
    model: ModelProto,
    configs: torch.Tensor,
) -> torch.Tensor:
    logging.info("Calculating inside Hamiltonian ...")
    count = len(configs)
    indices_i, indices_j, values = model.inside(configs)
    hamiltonian = torch.sparse_coo_tensor(torch.stack([indices_i, indices_j], dim=0), values, [count, count], dtype=torch.complex128)
    logging.info("Inside Hamiltonian calculated")
    return hamiltonian


def _lobpcg_process(
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

    hamiltonian = _inside_hamiltonian(model, configs).to_sparse_csr()

    logging.info("Calculating the minimum energy eigenvalue on the configurations.")
    energy, psi = lobpcg(hamiltonian, psi.view([-1, 1]), maxiter=1024)
    psi = psi.flatten()
    logging.info("Energy eigenvalue on configurations: %.10f, Reference energy: %.10f, Energy error: %.10f", energy.item(), model.ref_energy, energy.item() - model.ref_energy)

    logging.info("LOBPCG process completed.")

    return energy, hamiltonian, psi


def _select_by_lobpcg(
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

    _, _, psi = _lobpcg_process(model, configs, psi)

    logging.info("Identifying the indices of the most significant configurations.")
    indices = torch.argsort(psi.abs(), descending=True)[:count_selected]

    logging.info("Refining configurations to include only the most significant ones.")
    configs = configs[indices]
    psi = psi[indices]

    logging.info("LOBPCG-based selection process completed successfully.")

    return configs, psi


@torch.jit.script
def _union(
    configs_a: torch.Tensor,
    configs_b: torch.Tensor,
) -> torch.Tensor:
    # Merge two sets of configurations and their corresponding psi values, ensuring that in case of duplicates, the psi values from the second set (psi_b) are retained.
    configs_both = torch.cat([configs_a, configs_b], dim=0)
    configs_result = torch.unique(configs_both, dim=0, sorted=False, return_inverse=False, return_counts=False)
    return configs_result


@torch.jit.script
def _subtraction(
    configs_a: torch.Tensor,
    configs_b: torch.Tensor,
) -> torch.Tensor:
    # Subtract the configurations in configs_b from configs_a, retaining the psi values from psi_a for the remaining configurations.
    configs_both = torch.cat([configs_a, configs_b], dim=0)
    configs_result, both_to_result = torch.unique(configs_both, dim=0, sorted=False, return_inverse=True, return_counts=False)
    b_to_result = both_to_result[len(configs_a):]
    result_not_in_b = torch.ones(len(configs_result), device=configs_result.device, dtype=torch.bool)
    result_not_in_b[b_to_result] = False
    return configs_result[result_not_in_b]


@dataclasses.dataclass
class LearnConfig:
    """
    The two-step optimization process for solving quantum many-body problems.
    """

    # pylint: disable=too-many-instance-attributes

    common: typing.Annotated[CommonConfig, tyro.conf.OmitArgPrefixes]

    # The sampling count
    sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-n"])] = 4000
    # The name of the loss function to use
    loss_name: typing.Annotated[str, tyro.conf.arg(aliases=["-l"])] = "sum_filtered_angle_log"
    # Whether to use the global optimizer
    global_opt: typing.Annotated[bool, tyro.conf.arg(aliases=["-g"])] = False
    # Whether to use LBFGS instead of Adam
    use_lbfgs: typing.Annotated[bool, tyro.conf.arg(aliases=["-2"])] = False
    # The learning rate for the local optimizer
    learning_rate: typing.Annotated[float, tyro.conf.arg(aliases=["-r"], help_behavior_hint="(default: 1e-3 for Adam, 1 for LBFGS)")] = -1
    # The number of steps for the local optimizer
    local_step: typing.Annotated[int, tyro.conf.arg(aliases=["-s"], help_behavior_hint="(default: 10000 for Adam, 1000 for LBFGS)")] = -1
    # The early break loss threshold for local optimization
    local_loss: typing.Annotated[float, tyro.conf.arg(aliases=["-t"])] = 1e-8
    # The number of psi values to log after local optimization
    logging_psi: typing.Annotated[int, tyro.conf.arg(aliases=["-p"])] = 30
    # The number of post-sampling iterations
    post_sampling_iteration: typing.Annotated[int, tyro.conf.arg(aliases=["-i"])] = 0
    # The number of configurations to sample during post-sampling
    post_sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-c"])] = 40000

    def __post_init__(self) -> None:
        if self.learning_rate == -1:
            self.learning_rate = 1 if self.use_lbfgs else 1e-3
        if self.local_step == -1:
            self.local_step = 1000 if self.use_lbfgs else 10000

    def main(self) -> None:
        """
        The main function of the two-step optimization process.
        """
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements
        # pylint: disable=too-many-branches

        model, network, data = self.common.main()

        logging.info(
            "Arguments Summary: "
            "Sampling Count: %d, "
            "Loss Function: %s, "
            "Global Optimizer: %s, "
            "Using LBFGS: %s, "
            "Learning Rate: %.10f, "
            "Local Step: %d, "
            "Local Loss Threshold: %.10f, "
            "Logging Psi Count: %d, "
            "Post Sampling Iteration: %d, "
            "Post Sampling Count: %d",
            self.sampling_count,
            self.loss_name,
            "Yes" if self.global_opt else "No",
            "Yes" if self.use_lbfgs else "No",
            self.learning_rate,
            self.local_step,
            self.local_loss,
            self.logging_psi,
            self.post_sampling_iteration,
            self.post_sampling_count,
        )

        optimizer = initialize_optimizer(
            network.parameters(),
            use_lbfgs=self.use_lbfgs,
            learning_rate=self.learning_rate,
            state_dict=data.get("optimizer"),
        )

        if "learn" not in data:
            data["learn"] = {"global": 0, "local": 0}

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=self.common.folder())  # type: ignore[no-untyped-call]

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
                    configs, psi = _extend_with_select(model, configs, psi, self.post_sampling_count)
                    configs_all = _union(configs_all, configs)
                    if i != self.post_sampling_iteration - 1:
                        logging.info("Performing selection, iteration %d", i)
                        configs, psi = _select_by_lobpcg(model, configs, psi, self.sampling_count)
                    else:
                        logging.info("Skipping selection on the last iteration")
                logging.info("Extension and selection loop concluded")
                configs_too_small = _subtraction(configs_all, configs)
                logging.info("Post-sampling process successfully completed")

            logging.info("Solving the equation within the configuration subspace")
            target_energy, hamiltonian, psi = _lobpcg_process(model, configs, psi)
            max_index = psi.abs().argmax()
            psi = psi / psi[max_index]
            logging.info("Target within the subspace has been calculated")

            loss_func: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = getattr(losses, self.loss_name)

            optimizer = initialize_optimizer(
                network.parameters(),
                use_lbfgs=self.use_lbfgs,
                learning_rate=self.learning_rate,
                new_opt=not self.global_opt,
                optimizer=optimizer,
            )

            loss: torch.Tensor
            try_index = 0
            while True:
                state_backup = copy.deepcopy(network.state_dict())
                optimizer_backup = copy.deepcopy(optimizer.state_dict())

                def closure() -> torch.Tensor:
                    optimizer.zero_grad()
                    loss = torch.tensor(0)

                    amplitudes = network(configs)
                    amplitudes = amplitudes / amplitudes[max_index]
                    loss_main = loss_func(amplitudes, psi)
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
                success = True
                scale_learning_rate(optimizer, 1 / (1 << try_index))
                local_step: int = data["learn"]["local"]
                for i in range(self.local_step):
                    loss = optimizer.step(closure)  # type: ignore[assignment,arg-type]
                    logging.info("Local optimization in progress, step %d, current loss: %.10f", i, loss.item())
                    writer.add_scalars("loss", {self.loss_name: loss}, local_step)  # type: ignore[no-untyped-call]
                    local_step += 1
                    if torch.isnan(loss):
                        logging.warning("Loss is NaN, restoring the previous state and exiting the optimization loop")
                        success = False
                        break
                    if loss < self.local_loss:
                        logging.info("Local optimization halted as the loss threshold has been met")
                        break
                scale_learning_rate(optimizer, 1 << try_index)
                if success:
                    if any(torch.isnan(param).any() for param in network.parameters()):
                        logging.warning("NaN detected in parameters, restoring the previous state and exiting the optimization loop")
                        data["learn"]["local"] = local_step
                        success = False
                if success:
                    logging.info("Local optimization process completed")
                    break
                network.load_state_dict(state_backup)
                optimizer.load_state_dict(optimizer_backup)
                try_index = try_index + 1

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
            step = data["learn"]["global"]
            writer.add_scalars("energy/value", {"state": final_energy, "target": target_energy, "ref": model.ref_energy}, step)  # type: ignore[no-untyped-call]
            writer.add_scalars("energy/error", {"state": final_energy - model.ref_energy, "target": target_energy - model.ref_energy, "threshold": 1.6e-3}, step)  # type: ignore[no-untyped-call]
            logging.info("Displaying the largest amplitudes")
            indices = psi.abs().argsort(descending=True)
            text = []
            for index in indices[:self.logging_psi]:
                this_config = "".join(f"{i:08b}"[::-1] for i in configs[index].cpu().numpy())
                logging.info("Configuration: %s, Target amplitude: %s, Final amplitude: %s", this_config, f"{psi[index].item():.8f}", f"{amplitudes[index].item():.8f}")
                text.append(f"Configuration: {this_config}, Target amplitude: {psi[index].item():.8f}, Final amplitude: {amplitudes[index].item():.8f}")
            writer.add_text("config", "\n".join(text), step)  # type: ignore[no-untyped-call]
            writer.flush()  # type: ignore[no-untyped-call]

            logging.info("Saving model checkpoint")
            data["learn"]["global"] += 1
            data["network"] = network.state_dict()
            data["optimizer"] = optimizer.state_dict()
            self.common.save(data, data["learn"]["global"])
            logging.info("Checkpoint successfully saved")


subcommand_dict["learn"] = LearnConfig

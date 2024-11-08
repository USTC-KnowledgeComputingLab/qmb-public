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
from .subcommand_dict import subcommand_dict
from .utility import extend_with_select, lobpcg_process, select_by_lobpcg


@dataclasses.dataclass
class LearnConfig:
    """
    The two-step optimization process for solving quantum many-body problems.
    """

    # pylint: disable=too-many-instance-attributes

    common: typing.Annotated[CommonConfig, tyro.conf.OmitArgPrefixes]

    # sampling count
    sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-n"])] = 4000
    # learning rate for the local optimizer
    learning_rate: typing.Annotated[float, tyro.conf.arg(aliases=["-r"], help_behavior_hint="(default: 1e-3 for Adam, 1 for LBFGS)")] = -1
    # step count for the local optimizer
    local_step: typing.Annotated[int, tyro.conf.arg(aliases=["-s"], help_behavior_hint="(default: 1000 for Adam, 400 for LBFGS)")] = -1
    # early break loss threshold for local optimization
    local_loss: typing.Annotated[float, tyro.conf.arg(aliases=["-t"])] = 1e-8
    # psi count to be printed after local optimizer
    logging_psi: typing.Annotated[int, tyro.conf.arg(aliases=["-p"])] = 30
    # the loss function to be used
    loss_name: typing.Annotated[str, tyro.conf.arg(aliases=["-l"])] = "log"
    # use LBFGS instead of Adam
    use_lbfgs: typing.Annotated[bool, tyro.conf.arg(aliases=["-2"])] = False
    # the post sampling iteration
    post_sampling_iteration: typing.Annotated[int, tyro.conf.arg(aliases=["-i"])] = 0
    # the post sampling count
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
                configs_backup = configs
                psi_backup = psi
                for i in range(self.post_sampling_iteration):
                    logging.info("Performing extension and selection, iteration %d", i)
                    configs, psi = extend_with_select(model, configs, psi, self.post_sampling_count)
                    configs, psi = select_by_lobpcg(model, configs, psi, self.sampling_count)
                logging.info("Extension and selection loop concluded")
                too_small_in_backup = torch.all(torch.any(configs_backup.unsqueeze(1) - configs.unsqueeze(0) != 0, dim=-1), dim=-1)
                too_small_configs = configs_backup[too_small_in_backup]
                too_small_psi = psi_backup[too_small_in_backup] / 100
                configs = torch.cat([configs, too_small_configs], dim=0)
                psi = torch.cat([psi, too_small_psi], dim=0)
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
                amplitudes = network(configs)
                amplitudes = amplitudes / amplitudes[max_index]
                loss: torch.Tensor = loss_func(amplitudes, targets)
                loss.backward()  # type: ignore[no-untyped-call]
                loss.amplitudes = amplitudes  # type: ignore[attr-defined]
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
            amplitudes = loss.amplitudes  # type: ignore[attr-defined]
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
                logging.info("Configuration: %s, Target amplitude: %s, Final amplitude: %s", "".join(map(str, configs[index].cpu().numpy())), f"{targets[index].item():.8f}",
                             f"{amplitudes[index].item():.8f}")


subcommand_dict["learn"] = LearnConfig

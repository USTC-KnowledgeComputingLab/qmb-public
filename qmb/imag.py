"""
This file implements a two-step optimization process for solving quantum many-body problems based on imaginary time.
"""

import logging
import typing
import dataclasses
import scipy
import torch
import tyro
from . import losses
from .common import CommonConfig
from .subcommand_dict import subcommand_dict
from .utility import extend_with_select


def _lanczos(hamiltonian: torch.Tensor, psi: torch.Tensor, step: int, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    v = [psi / torch.linalg.norm(psi)]  # pylint: disable=not-callable
    alpha = torch.tensor([], dtype=hamiltonian.dtype.to_real(), device=hamiltonian.device)
    beta = torch.tensor([], dtype=hamiltonian.dtype.to_real(), device=hamiltonian.device)

    w = hamiltonian @ v[-1]
    alpha = torch.cat((alpha, (v[-1].conj() @ w).real.reshape([1])), dim=0)
    w = w - alpha[-1] * v[-1]
    for _ in range(step):
        norm_w = torch.linalg.norm(w)  # pylint: disable=not-callable
        if norm_w < threshold:
            break
        beta = torch.cat((beta, norm_w.reshape([1])), dim=0)
        v.append(w / beta[-1])
        w = hamiltonian @ v[-1]
        alpha = torch.cat((alpha, (v[-1].conj() @ w).real.reshape([1])), dim=0)
        w = w - alpha[-1] * v[-1] - beta[-1] * v[-2]

    # Currently, PyTorch does not support this functionality natively, so we resort to using SciPy for this operation.
    vals, vecs = scipy.linalg.eigh_tridiagonal(alpha.cpu(), beta.cpu(), lapack_driver="stemr", select="i", select_range=[0, 0])
    energy = torch.as_tensor(vals[0])
    result = torch.sum(torch.as_tensor(vecs[:, 0]).to(device=hamiltonian.device) * torch.stack(v, dim=1), dim=1)
    return energy, result


@dataclasses.dataclass
class ImaginaryConfig:
    """
    The two-step optimization process for solving quantum many-body problems based on imaginary time.
    """

    # pylint: disable=too-many-instance-attributes

    common: typing.Annotated[CommonConfig, tyro.conf.OmitArgPrefixes]

    # The sampling count
    sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-n"])] = 4000
    # The number of configurations to sample during post-sampling
    post_sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-c"])] = 0
    # The number of Krylov iterations to perform
    krylov_iteration: typing.Annotated[int, tyro.conf.arg(aliases=["-k"])] = 16
    # The threshold for the Krylov iteration
    krylov_threshold: typing.Annotated[float, tyro.conf.arg(aliases=["-d"])] = 1e-8
    # The name of the loss function to use
    loss_name: typing.Annotated[str, tyro.conf.arg(aliases=["-l"])] = "hybrid"
    # Whether to use LBFGS instead of Adam
    use_lbfgs: typing.Annotated[bool, tyro.conf.arg(aliases=["-2"])] = False
    # The learning rate for the local optimizer
    learning_rate: typing.Annotated[float, tyro.conf.arg(aliases=["-r"], help_behavior_hint="(default: 1e-3 for Adam, 1 for LBFGS)")] = -1
    # The number of steps for the local optimizer
    local_step: typing.Annotated[int, tyro.conf.arg(aliases=["-s"], help_behavior_hint="(default: 1000 for Adam, 400 for LBFGS)")] = -1
    # The early break loss threshold for local optimization
    local_loss: typing.Annotated[float, tyro.conf.arg(aliases=["-t"])] = 1e-6
    # The number of psi values to log after local optimization
    logging_psi: typing.Annotated[int, tyro.conf.arg(aliases=["-p"])] = 30

    def __post_init__(self) -> None:
        if self.learning_rate == -1:
            self.learning_rate = 1 if self.use_lbfgs else 1e-3
        if self.local_step == -1:
            self.local_step = 400 if self.use_lbfgs else 1000

    def main(self) -> None:
        """
        The main function of two-step optimization process based on imaginary time.
        """
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements

        model, network = self.common.main()

        logging.info(
            "Arguments Summary: "
            "Sampling Count: %d, "
            "Post-Sampling Count: %d, "
            "krylov Iteration: %.10f, "
            "krylov Threshold: %.10f, "
            "Loss Function: %s, "
            "Use LBFGS: %s, "
            "Learning Rate: %.10f, "
            "Local Steps: %d, "
            "Local Loss Threshold: %.10f, "
            "Logging Psi: %d",
            self.sampling_count,
            self.post_sampling_count,
            self.krylov_iteration,
            self.krylov_threshold,
            self.loss_name,
            "Yes" if self.use_lbfgs else "No",
            self.learning_rate,
            self.local_step,
            self.local_loss,
            self.logging_psi,
        )

        while True:
            logging.info("Starting a new optimization cycle")

            logging.info("Sampling configurations")
            configs, psi, _, _ = network.generate_unique(self.sampling_count)
            configs_count = len(configs)
            logging.info("Sampling completed, pool size: %d", configs_count)

            if self.post_sampling_count != 0:
                logging.info("Extending configuration pool")
                configs, psi = extend_with_select(model, configs, psi, self.post_sampling_count)
                configs_count = len(configs)
                logging.info("Configuration pool extended, new size: %d", configs_count)

            logging.info("Calculating sparse data for the Hamiltonian matrix on the configurations.")
            indices_i_and_j, values = model.inside(configs)
            logging.info("Converting sparse matrix data into a sparse tensor.")
            hamiltonian = torch.sparse_coo_tensor(indices_i_and_j.T, values, [configs_count, configs_count], dtype=torch.complex128).to_sparse_csr()
            logging.info("Sparse Hamiltonian matrix on configurations has been created.")

            logging.info("Computing the target for local optimization")
            target_energy, targets = _lanczos(hamiltonian, psi, self.krylov_iteration, self.krylov_threshold)
            max_index = targets.abs().argmax()
            targets = targets / targets[max_index]
            logging.info("Local optimization target calculated, the target energy is %.10f", target_energy.item())

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
                loss = loss_func(amplitudes, targets)
                loss.backward()  # type: ignore[no-untyped-call]
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
                logging.info("Configuration: %s, Target amplitude: %s, Final amplitude: %s", "".join(map(str, configs[index].cpu().numpy())), f"{targets[index].item():.8f}",
                             f"{amplitudes[index].item():.8f}")


subcommand_dict["imag"] = ImaginaryConfig

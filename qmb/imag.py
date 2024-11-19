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
from .model_dict import ModelProto


class _DynamicLanczos:
    """
    This class implements the dynamic Lanczos algorithm for solving quantum many-body problems.
    """

    # pylint: disable=too-few-public-methods

    def __init__(  # pylint: disable=too-many-arguments
            self,
            *,
            model: ModelProto,
            configs: torch.Tensor,
            psi: torch.Tensor,
            step: int,
            threshold: float,
            count_extend: int,
    ):
        self.model = model
        self.configs = configs
        self.psi = psi
        self.step = step
        self.threshold = threshold
        self.count_extend = count_extend

    def _hamiltonian(self) -> torch.Tensor:
        logging.info("Constructing Hamiltonian...")
        indices_i_and_j, values = self.model.inside(self.configs)
        count_config = len(self.configs)
        hamiltonian = torch.sparse_coo_tensor(indices_i_and_j.T, values, [count_config, count_config], dtype=torch.complex128).to_sparse_csr()
        logging.info("Hamiltonian constructed")
        return hamiltonian

    def _extend(self, psi: torch.Tensor) -> None:
        logging.info("Extending basis...")
        indices_i_and_j, values, configs_extended = self.model.outside(self.configs)
        count_core = len(self.configs)
        count_extended = len(configs_extended)
        hamiltonian = torch.sparse_coo_tensor(indices_i_and_j.T, values, [count_core, count_extended], dtype=torch.complex128).to_sparse_csc()

        importance = (psi.conj() @ hamiltonian).abs()
        importance[:count_core] += importance.max()

        selected_indices = importance.sort(descending=True).indices[:count_core + self.count_extend].sort().values
        count_selected = len(selected_indices)

        self.configs = configs_extended[selected_indices]
        self.psi = torch.nn.functional.pad(self.psi, (0, count_selected - count_core))
        logging.info("Basis extended from %d to %d", count_core, count_selected)

    def run(self, keep_until: int = -1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the dynamic Lanczos algorithm.
        """
        hamiltonian = self._hamiltonian()

        v = [self.psi / torch.linalg.norm(self.psi)]  # pylint: disable=not-callable
        alpha = torch.tensor([], dtype=hamiltonian.dtype.to_real(), device=hamiltonian.device)
        beta = torch.tensor([], dtype=hamiltonian.dtype.to_real(), device=hamiltonian.device)

        if keep_until == -1:
            self._extend(v[-1])
            return self.run(keep_until=keep_until + 1)
        w = hamiltonian @ v[-1]
        alpha = torch.cat((alpha, (v[-1].conj() @ w).real.reshape([1])), dim=0)
        w = w - alpha[-1] * v[-1]
        for i in range(self.step):
            norm_w = torch.linalg.norm(w)  # pylint: disable=not-callable
            if norm_w < self.threshold:
                break
            beta = torch.cat((beta, norm_w.reshape([1])), dim=0)
            v.append(w / beta[-1])
            if keep_until == i:
                self._extend(v[-1])
                return self.run(keep_until=keep_until + 1)
            w = hamiltonian @ v[-1]
            alpha = torch.cat((alpha, (v[-1].conj() @ w).real.reshape([1])), dim=0)
            w = w - alpha[-1] * v[-1] - beta[-1] * v[-2]

        # Currently, PyTorch does not support this functionality natively, so we resort to using SciPy for this operation.
        vals, vecs = scipy.linalg.eigh_tridiagonal(alpha.cpu(), beta.cpu(), lapack_driver="stemr", select="i", select_range=[0, 0])
        energy = torch.as_tensor(vals[0])
        result = torch.sum(torch.as_tensor(vecs[:, 0]).to(device=hamiltonian.device) * torch.stack(v, dim=1), dim=1)
        return hamiltonian, energy, self.configs, result


@dataclasses.dataclass
class ImaginaryConfig:
    """
    The two-step optimization process for solving quantum many-body problems based on imaginary time.
    """

    # pylint: disable=too-many-instance-attributes

    common: typing.Annotated[CommonConfig, tyro.conf.OmitArgPrefixes]

    # The sampling count
    sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-n"])] = 4000
    # The extend count for the Krylov subspace
    krylov_extend_count: typing.Annotated[int, tyro.conf.arg(aliases=["-c"])] = 400
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
            "Krylov Extend Count: %d, "
            "krylov Iteration: %d, "
            "krylov Threshold: %.10f, "
            "Loss Function: %s, "
            "Use LBFGS: %s, "
            "Learning Rate: %.10f, "
            "Local Steps: %d, "
            "Local Loss Threshold: %.10f, "
            "Logging Psi: %d",
            self.sampling_count,
            self.krylov_extend_count,
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
            logging.info("Sampling completed")

            logging.info("Computing the target for local optimization")
            hamiltonian, target_energy, configs, psi = _DynamicLanczos(
                model=model,
                configs=configs,
                psi=psi,
                step=self.krylov_iteration,
                threshold=self.krylov_threshold,
                count_extend=self.krylov_extend_count,
            ).run()
            max_index = psi.abs().argmax()
            psi = psi / psi[max_index]
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
                loss = loss_func(amplitudes, psi)
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
            indices = psi.abs().sort(descending=True).indices
            for index in indices[:self.logging_psi]:
                logging.info("Configuration: %s, Target amplitude: %s, Final amplitude: %s", "".join(map(str, configs[index].cpu().numpy())), f"{psi[index].item():.8f}",
                             f"{amplitudes[index].item():.8f}")


subcommand_dict["imag"] = ImaginaryConfig

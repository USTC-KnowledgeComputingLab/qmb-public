"""
This file implements a two-step optimization process for solving quantum many-body problems based on imaginary time.
"""

import copy
import logging
import typing
import dataclasses
import functools
import scipy
import torch
import torch.utils.tensorboard
import tyro
from . import losses
from .common import CommonConfig
from .subcommand_dict import subcommand_dict
from .model_dict import ModelProto
from .optimizer import initialize_optimizer, scale_learning_rate


@dataclasses.dataclass
class _DynamicLanczos:
    """
    This class implements the dynamic Lanczos algorithm for solving quantum many-body problems.
    """

    # pylint: disable=too-few-public-methods

    model: ModelProto
    configs: torch.Tensor
    psi: torch.Tensor
    step: int
    threshold: float
    count_extend: int

    def _extend(self, psi: torch.Tensor) -> None:
        logging.info("Extending basis...")

        count_core = len(self.configs)
        logging.info("Number of core configurations: %d", count_core)

        self.configs = torch.cat([self.configs, self.model.find_relative(self.configs, psi, self.count_extend)])
        count_selected = len(self.configs)
        self.psi = torch.nn.functional.pad(self.psi, (0, count_selected - count_core))
        logging.info("Basis extended from %d to %d", count_core, count_selected)

    def run(self) -> typing.Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Run the Lanczos algorithm.

        Yields
        ------
        energy : torch.Tensor
            The ground energy.
        configs : torch.Tensor
            The configurations.
        psi : torch.Tensor
            The wavefunction amplitude on the configurations.
        """
        alpha: list[torch.Tensor]
        beta: list[torch.Tensor]
        v: list[torch.Tensor]
        energy: torch.Tensor
        psi: torch.Tensor
        if self.count_extend == 0:
            # Do not extend the configuration, process the standard lanczos.
            for _, [alpha, beta, v] in zip(range(1 + self.step), self._run()):
                if len(beta) != 0:
                    energy, psi = self._eigh_tridiagonal(alpha, beta, v)
                    yield energy, self.configs, psi
        else:
            # Extend the configuration, during processing the dynamic lanczos.
            for step in range(1 + self.step):
                for _, [alpha, beta, v] in zip(range(1 + step), self._run()):
                    pass
                if len(beta) != 0:
                    energy, psi = self._eigh_tridiagonal(alpha, beta, v)
                    yield energy, self.configs, psi
                self._extend(v[-1])

    def _run(self) -> typing.Iterable[tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]]:
        """
        Process the standard lanczos.

        Yields
        ------
        alpha : list[torch.Tensor]
            The alpha values.
        beta : list[torch.Tensor]
            The beta values.
        v : list[torch.Tensor]
            The v values.
        """
        # In this function, we distribute data to the GPU and CPU.
        # The details are as follows:
        # All data other than v is always on the GPU.
        # The last v is always on the GPU and the rest are moved to the CPU immediately after necessary calculations.
        v: list[torch.Tensor] = [self.psi / torch.linalg.norm(self.psi)]  # pylint: disable=not-callable
        alpha: list[torch.Tensor] = []
        beta: list[torch.Tensor] = []
        w: torch.Tensor
        w = self.model.apply_within(self.configs, v[-1], self.configs)  # pylint: disable=assignment-from-no-return
        alpha.append((w.conj() @ v[-1]).real)
        yield (alpha, beta, v)
        w = w - alpha[-1] * v[-1]
        while True:
            norm_w = torch.linalg.norm(w)  # pylint: disable=not-callable
            if norm_w < self.threshold:
                break
            beta.append(norm_w)
            v.append(w / beta[-1])
            w = self.model.apply_within(self.configs, v[-1], self.configs)  # pylint: disable=assignment-from-no-return
            alpha.append((w.conj() @ v[-1]).real)
            yield (alpha, beta, v)
            w = w - alpha[-1] * v[-1] - beta[-1] * v[-2]
            v[-2] = v[-2].cpu()  # v maybe very large, so we need to move it to CPU

    def _eigh_tridiagonal(
        self,
        alpha: list[torch.Tensor],
        beta: list[torch.Tensor],
        v: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Currently, PyTorch does not support eigh_tridiagonal natively, so we resort to using SciPy for this operation.
        # We can only use 'stebz' or 'stemr' drivers in the current version of SciPy.
        # However, 'stemr' consumes a lot of memory, so we opt for 'stebz' here.
        # 'stebz' is efficient and only takes a few seconds even for large matrices with dimensions up to 10,000,000.
        vals, vecs = scipy.linalg.eigh_tridiagonal(torch.stack(alpha, dim=0).cpu(), torch.stack(beta, dim=0).cpu(), lapack_driver="stebz", select="i", select_range=(0, 0))
        energy = torch.as_tensor(vals[0])
        result = functools.reduce(torch.add, (weight[0] * vector.to(device=self.configs.device) for weight, vector in zip(vecs, v)))
        return energy, result


@dataclasses.dataclass
class ImaginaryConfig:
    """
    The two-step optimization process for solving quantum many-body problems based on imaginary time.
    """

    # pylint: disable=too-many-instance-attributes

    common: typing.Annotated[CommonConfig, tyro.conf.OmitArgPrefixes]

    # The sampling count
    sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-n"])] = 2048
    # The extend count for the Krylov subspace
    krylov_extend_count: typing.Annotated[int, tyro.conf.arg(aliases=["-c"])] = 64
    # The number of Krylov iterations to perform
    krylov_iteration: typing.Annotated[int, tyro.conf.arg(aliases=["-k"])] = 31
    # The threshold for the Krylov iteration
    krylov_threshold: typing.Annotated[float, tyro.conf.arg(aliases=["-d"])] = 1e-8
    # The name of the loss function to use
    loss_name: typing.Annotated[str, tyro.conf.arg(aliases=["-l"])] = "sum_filtered_angle_log"
    # Whether to use the global optimizer
    global_opt: typing.Annotated[bool, tyro.conf.arg(aliases=["-g"])] = False
    # Whether to use LBFGS instead of Adam
    use_lbfgs: typing.Annotated[bool, tyro.conf.arg(aliases=["-2"])] = False
    # The learning rate for the local optimizer
    learning_rate: typing.Annotated[float, tyro.conf.arg(aliases=["-r"], help_behavior_hint="(default: 1e-3 for Adam, 1 for LBFGS)")] = -1
    # The local batch count used to avoid memory overflow
    local_batch_count: typing.Annotated[int, tyro.conf.arg(aliases=["-b"])] = 1
    # The number of steps for the local optimizer
    local_step: typing.Annotated[int, tyro.conf.arg(aliases=["-s"], help_behavior_hint="(default: 10000 for Adam, 1000 for LBFGS)")] = -1
    # The early break loss threshold for local optimization
    local_loss: typing.Annotated[float, tyro.conf.arg(aliases=["-t"])] = 1e-8
    # The number of psi values to log after local optimization
    logging_psi: typing.Annotated[int, tyro.conf.arg(aliases=["-p"])] = 30

    def __post_init__(self) -> None:
        if self.learning_rate == -1:
            self.learning_rate = 1 if self.use_lbfgs else 1e-3
        if self.local_step == -1:
            self.local_step = 1000 if self.use_lbfgs else 10000

    def main(self) -> None:
        """
        The main function of two-step optimization process based on imaginary time.
        """
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements

        model, network, data = self.common.main()

        logging.info(
            "Arguments Summary: "
            "Sampling Count: %d, "
            "Krylov Extend Count: %d, "
            "krylov Iteration: %d, "
            "krylov Threshold: %.10f, "
            "Loss Function: %s, "
            "Global Optimizer: %s, "
            "Use LBFGS: %s, "
            "Learning Rate: %.10f, "
            "Local Batch Count: %d, "
            "Local Steps: %d, "
            "Local Loss Threshold: %.10f, "
            "Logging Psi: %d",
            self.sampling_count,
            self.krylov_extend_count,
            self.krylov_iteration,
            self.krylov_threshold,
            self.loss_name,
            "Yes" if self.global_opt else "No",
            "Yes" if self.use_lbfgs else "No",
            self.learning_rate,
            self.local_batch_count,
            self.local_step,
            self.local_loss,
            self.logging_psi,
        )

        optimizer = initialize_optimizer(
            network.parameters(),
            use_lbfgs=self.use_lbfgs,
            learning_rate=self.learning_rate,
            state_dict=data.get("optimizer"),
        )

        if "imag" not in data:
            data["imag"] = {"global": 0, "local": 0, "lanczos": 0}

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=self.common.folder())  # type: ignore[no-untyped-call]

        while True:
            logging.info("Starting a new optimization cycle")

            logging.info("Sampling configurations")
            configs, target_psi, _, _ = network.generate_unique(self.sampling_count)
            logging.info("Sampling completed")

            logging.info("Computing the target for local optimization")
            target_energy: torch.Tensor
            for target_energy, configs, target_psi in _DynamicLanczos(
                    model=model,
                    configs=configs,
                    psi=target_psi,
                    step=self.krylov_iteration,
                    threshold=self.krylov_threshold,
                    count_extend=self.krylov_extend_count,
            ).run():
                logging.info("The current energy is %.10f", target_energy.item())
                writer.add_scalar("imag/lanczos/energy", target_energy, data["imag"]["lanczos"])  # type: ignore[no-untyped-call]
                writer.add_scalar("imag/lanczos/error", target_energy - model.ref_energy, data["imag"]["lanczos"])  # type: ignore[no-untyped-call]
                data["imag"]["lanczos"] += 1
            max_index = target_psi.abs().argmax()
            target_psi = target_psi / target_psi[max_index]
            logging.info("Local optimization target calculated, the target energy is %.10f", target_energy.item())

            loss_func: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = getattr(losses, self.loss_name)

            optimizer = initialize_optimizer(
                network.parameters(),
                use_lbfgs=self.use_lbfgs,
                learning_rate=self.learning_rate,
                new_opt=not self.global_opt,
                optimizer=optimizer,
            )

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                total_size = len(configs)
                batch_size = total_size // self.local_batch_count
                remainder = total_size % self.local_batch_count
                total_loss = 0.0
                total_psi = []
                for i in range(self.local_batch_count):
                    if i < remainder:
                        current_batch_size = batch_size + 1
                    else:
                        current_batch_size = batch_size
                    start_index = i * batch_size + min(i, remainder)
                    end_index = start_index + current_batch_size
                    batch_indices = torch.arange(start_index, end_index, device=configs.device, dtype=torch.int64)
                    psi_batch = target_psi[batch_indices]
                    batch_indices = torch.cat((batch_indices, torch.tensor([max_index], device=configs.device, dtype=torch.int64)))
                    batch_configs = configs[batch_indices]
                    psi = network(batch_configs)
                    psi_max = psi[-1]
                    psi = psi[:-1]
                    psi = psi / psi_max
                    loss = loss_func(psi, psi_batch)
                    loss = loss * (current_batch_size / total_size)
                    loss.backward()  # type: ignore[no-untyped-call]
                    total_loss += loss.item()
                    total_psi.append(psi.detach())
                total_loss_tensor = torch.tensor(total_loss)
                total_loss_tensor.psi = torch.cat(total_psi)  # type: ignore[attr-defined]
                return total_loss_tensor

            loss: torch.Tensor
            try_index = 0
            while True:
                state_backup = copy.deepcopy(network.state_dict())
                optimizer_backup = copy.deepcopy(optimizer.state_dict())

                logging.info("Starting local optimization process")
                success = True
                last_loss: float = 0.0
                local_step: int = data["imag"]["local"]
                scale_learning_rate(optimizer, 1 / (1 << try_index))
                for i in range(self.local_step):
                    loss = optimizer.step(closure)  # type: ignore[assignment,arg-type]
                    logging.info("Local optimization in progress, step %d, current loss: %.10f", i, loss.item())
                    writer.add_scalar(f"imag/loss/{self.loss_name}", loss, local_step)  # type: ignore[no-untyped-call]
                    local_step += 1
                    if torch.isnan(loss) or torch.isinf(loss):
                        logging.warning("Loss is NaN, restoring the previous state and exiting the optimization loop")
                        success = False
                        break
                    if loss < self.local_loss:
                        logging.info("Local optimization halted as the loss threshold has been met")
                        break
                    if abs(loss - last_loss) < self.local_loss:
                        logging.info("Local optimization halted as the loss difference is too small")
                        break
                    last_loss = loss.item()
                scale_learning_rate(optimizer, 1 << try_index)
                if success:
                    if any(torch.isnan(param).any() or torch.isinf(param).any() for param in network.parameters()):
                        logging.warning("NaN detected in parameters, restoring the previous state and exiting the optimization loop")
                        success = False
                if success:
                    logging.info("Local optimization process completed")
                    data["imag"]["local"] = local_step
                    break
                network.load_state_dict(state_backup)
                optimizer.load_state_dict(optimizer_backup)
                try_index = try_index + 1

            logging.info("Current optimization cycle completed")

            loss = typing.cast(torch.Tensor, torch.enable_grad(closure)())  # type: ignore[no-untyped-call,call-arg]
            psi: torch.Tensor = loss.psi  # type: ignore[attr-defined]
            final_energy = ((psi.conj() @ model.apply_within(configs, psi, configs)) / (psi.conj() @ psi)).real
            logging.info(
                "Loss during local optimization: %.10f, Final energy: %.10f, Target energy: %.10f, Reference energy: %.10f, Final error: %.10f",
                loss.item(),
                final_energy.item(),
                target_energy.item(),
                model.ref_energy,
                final_energy.item() - model.ref_energy,
            )
            writer.add_scalar("imag/energy/state", final_energy, data["imag"]["global"])  # type: ignore[no-untyped-call]
            writer.add_scalar("imag/energy/target", target_energy, data["imag"]["global"])  # type: ignore[no-untyped-call]
            writer.add_scalar("imag/error/state", final_energy - model.ref_energy, data["imag"]["global"])  # type: ignore[no-untyped-call]
            writer.add_scalar("imag/error/target", target_energy - model.ref_energy, data["imag"]["global"])  # type: ignore[no-untyped-call]
            logging.info("Displaying the largest amplitudes")
            indices = target_psi.abs().argsort(descending=True)
            text = []
            for index in indices[:self.logging_psi]:
                this_config = model.show_config(configs[index])
                logging.info("Configuration: %s, Target amplitude: %s, Final amplitude: %s", this_config, f"{target_psi[index].item():.8f}", f"{psi[index].item():.8f}")
                text.append(f"Configuration: {this_config}, Target amplitude: {target_psi[index].item():.8f}, Final amplitude: {psi[index].item():.8f}")
            writer.add_text("config", "\n".join(text), data["imag"]["global"])  # type: ignore[no-untyped-call]
            writer.flush()  # type: ignore[no-untyped-call]

            logging.info("Saving model checkpoint")
            data["imag"]["global"] += 1
            data["network"] = network.state_dict()
            data["optimizer"] = optimizer.state_dict()
            self.common.save(data, data["imag"]["global"])
            logging.info("Checkpoint successfully saved")

            logging.info("Current optimization cycle completed")


subcommand_dict["imag"] = ImaginaryConfig

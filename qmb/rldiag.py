"""
This file implements the reinforcement learning based subspace diagonalization algorithm.
"""

import sys
import logging
import typing
import dataclasses
import tyro
import scipy
import torch
from .common import CommonConfig
from .subcommand_dict import subcommand_dict
from .model_dict import ModelProto
from .optimizer import initialize_optimizer
from .bitspack import pack_int
from .random_engine import dump_random_engine_state


def lanczos_energy(model: ModelProto, configs: torch.Tensor, step: int, threshold: float) -> float:
    """
    Calculate the energy using the Lanczos method.
    """
    vector = torch.randn([configs.size(0)], dtype=torch.complex128, device=configs.device)

    v: list[torch.Tensor] = [vector / torch.linalg.norm(vector)]  # pylint: disable=not-callable
    alpha: list[torch.Tensor] = []
    beta: list[torch.Tensor] = []
    w: torch.Tensor
    w = model.apply_within(configs, v[-1], configs)
    alpha.append((w.conj() @ v[-1]).real)
    w = w - alpha[-1] * v[-1]
    i = 0
    while True:
        norm_w = torch.linalg.norm(w)  # pylint: disable=not-callable
        if norm_w < threshold:
            break
        beta.append(norm_w)
        v.append(w / beta[-1])
        w = model.apply_within(configs, v[-1], configs)
        alpha.append((w.conj() @ v[-1]).real)
        if i == step:
            break
        w = w - alpha[-1] * v[-1] - beta[-1] * v[-2]
        v[-2] = v[-2].cpu()  # v maybe very large, so we need to move it to CPU
        i += 1

    if len(beta) == 0:
        return alpha[0].item()
    vals, _ = scipy.linalg.eigh_tridiagonal(torch.stack(alpha, dim=0).cpu(), torch.stack(beta, dim=0).cpu(), lapack_driver="stebz", select="i", select_range=(0, 0))
    energy = torch.as_tensor(vals[0])
    return energy.item()


@dataclasses.dataclass
class RldiagConfig:
    """
    The reinforcement learning based subspace diagonalization algorithm.
    """

    # pylint: disable=too-many-instance-attributes

    common: typing.Annotated[CommonConfig, tyro.conf.OmitArgPrefixes]

    # The initial configuration for the first step, which is usually the Hatree-Fock state for quantum chemistry system
    initial_config: typing.Annotated[typing.Optional[str], tyro.conf.arg(aliases=["-i"])] = None
    # The learning rate for the local optimizer
    learning_rate: typing.Annotated[float, tyro.conf.arg(aliases=["-r"], help_behavior_hint="(default: 1e-3 for Adam, 1 for LBFGS)")] = -1
    # Whether to use LBFGS instead of Adam
    use_lbfgs: typing.Annotated[bool, tyro.conf.arg(aliases=["-2"])] = False
    # The step of lanczos iteration for calculating the energy
    lanczos_step: typing.Annotated[int, tyro.conf.arg(aliases=["-l"])] = 32
    # The thereshold for the lanczos iteration
    lanczos_threshold: typing.Annotated[float, tyro.conf.arg(aliases=["-t"])] = 1e-8
    # The exponent for the sigma calculation
    alpha: typing.Annotated[float, tyro.conf.arg(aliases=["-a"])] = 0.5
    # The perturbation for the action
    perturb: typing.Annotated[float, tyro.conf.arg(aliases=["-p"])] = 0.0

    def __post_init__(self) -> None:
        if self.learning_rate == -1:
            self.learning_rate = 1 if self.use_lbfgs else 1e-3

    def main(self) -> None:
        """
        The main function for the reinforcement learning based subspace diagonalization algorithm.
        """

        # pylint: disable=too-many-statements
        # pylint: disable=too-many-locals

        model, network, data = self.common.main()

        logging.info(
            "Arguments Summary: "
            "Initial Configuration: %s, "
            "Learning Rate: %.10f, "
            "Use LBFGS: %s, "
            "Lanczos step: %d, "
            "Lanczos threshold: %.10f, "
            "Alpha: %.10f, "
            "Perturb: %.10f",
            self.initial_config if self.initial_config is not None else "None",
            self.learning_rate,
            "Yes" if self.use_lbfgs else "No",
            self.lanczos_step,
            self.lanczos_threshold,
            self.alpha,
            self.perturb,
        )

        optimizer = initialize_optimizer(
            network.parameters(),
            use_lbfgs=self.use_lbfgs,
            learning_rate=self.learning_rate,
            state_dict=data.get("optimizer"),
        )

        if self.initial_config is None:
            if "rldiag" not in data:  # pylint: disable=no-else-raise
                raise ValueError("The initial configuration is not set, please set it.")
            else:
                configs = data["rldiag"]["configs"].to(device=self.common.device)
        else:
            # The format of initial_config has two options:
            # 1. The 0/1 string, such as "11111100000000000000000000000011110000000000110000110000"
            # 2. The packed string, such as "63,0,0,192,3,48,12"
            if "," not in self.initial_config and all(i in "01" for i in self.initial_config):
                # The 0/1 string
                configs = pack_int(
                    torch.tensor([[int(i) for i in self.initial_config]], dtype=torch.bool, device=self.common.device),
                    size=1,
                )
            else:
                # The packed string
                configs = torch.tensor([[int(i) for i in self.initial_config.split(",")]], dtype=torch.uint8, device=self.common.device)
            if "rldiag" not in data:
                data["rldiag"] = {"global": 0, "local": 0, "configs": configs, "sigma": [[0]], "chain": [[]]}
            else:
                data["rldiag"]["configs"] = configs
                data["rldiag"]["local"] = 0
                data["rldiag"]["sigma"].append([0])
                data["rldiag"]["chain"].append([])

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=self.common.folder())  # type: ignore[no-untyped-call]

        while True:
            logging.info("Starting a new cycle")

            logging.info("Evaluating each configuration")
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                score = network(configs)
            logging.info("All configurations are evaluated")

            logging.info("Applying the action")
            action = score.real >= torch.randn_like(score.real) * self.perturb
            pruned_configs = configs[action]
            extended_configs = torch.cat([
                pruned_configs,
                model.find_relative(
                    pruned_configs,
                    torch.ones([pruned_configs.size(0)], dtype=torch.complex128, device=self.common.device),
                    pruned_configs.size(0),
                    configs,
                ),
            ])
            logging.info("Action has been applied")
            configs_size = extended_configs.size(0)
            logging.info("Configuration pool size: %d", configs_size)
            writer.add_scalar("rldiag/configs/global", configs_size, data["rldiag"]["global"])  # type: ignore[no-untyped-call]
            writer.add_scalar("rldiag/configs/local", configs_size, data["rldiag"]["local"])  # type: ignore[no-untyped-call]
            if configs_size == 0:
                logging.info("All configurations has been pruned, please start a new configuration pool state")
                sys.exit(0)

            old_configs = configs
            configs = extended_configs
            energy = lanczos_energy(model, configs, self.lanczos_step, self.lanczos_threshold)
            logging.info("Current energy is %.10f, Reference energy is %.10f, Energy error is %.10f", energy, model.ref_energy, energy - model.ref_energy)
            writer.add_scalar("rldiag/energy/state/global", energy, data["rldiag"]["global"])  # type: ignore[no-untyped-call]
            writer.add_scalar("rldiag/energy/state/local", energy, data["rldiag"]["local"])  # type: ignore[no-untyped-call]
            writer.add_scalar("rldiag/energy/error/global", energy - model.ref_energy, data["rldiag"]["global"])  # type: ignore[no-untyped-call]
            writer.add_scalar("rldiag/energy/error/local", energy - model.ref_energy, data["rldiag"]["local"])  # type: ignore[no-untyped-call]
            if "base" not in data["rldiag"]:
                # This is the first time to calculate the energy, which is usually the energy of the Hatree-Fock state for quantum chemistry system
                # This will not be flushed acrossing different cycle chains.
                data["rldiag"]["base"] = energy
            sigma = (energy - data["rldiag"]["base"]) / (configs_size**self.alpha)
            logging.info("Current sigma is %.10f", sigma)
            writer.add_scalar("rldiag/sigma/global", sigma, data["rldiag"]["global"])  # type: ignore[no-untyped-call]
            writer.add_scalar("rldiag/sigma/local", sigma, data["rldiag"]["local"])  # type: ignore[no-untyped-call]
            reward = -(sigma - data["rldiag"]["sigma"][-1][-1])
            data["rldiag"]["sigma"][-1].append(sigma)
            data["rldiag"]["chain"][-1].append((old_configs, action, reward))
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                loss_term = score * torch.where(action, +1, -1)
                loss = -reward * loss_term.sum()
                optimizer.zero_grad()
                loss.backward()
            optimizer.step()  # pylint: disable=no-value-for-parameter

            logging.info("Saving model checkpoint")
            data["rldiag"]["configs"] = configs
            data["rldiag"]["energy"] = energy
            data["rldiag"]["global"] += 1
            data["rldiag"]["local"] += 1
            data["network"] = network.state_dict()
            data["optimizer"] = optimizer.state_dict()
            data["random"] = {"host": torch.get_rng_state(), "device": dump_random_engine_state(self.common.device)}
            self.common.save(data, data["rldiag"]["global"])
            logging.info("Checkpoint successfully saved")

            logging.info("Current cycle completed")


subcommand_dict["rldiag"] = RldiagConfig

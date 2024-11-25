"""
This file contains the common step to create a model and network for various scripts.
"""

import sys
import logging
import typing
import pathlib
import dataclasses
import torch
import tyro
from .model_dict import model_dict, ModelProto, NetworkProto


@dataclasses.dataclass
class CommonConfig:
    """
    This class defines the common settings needed to create a model and network.
    """

    # pylint: disable=too-many-instance-attributes

    # The model name
    model_name: typing.Annotated[str, tyro.conf.Positional, tyro.conf.arg(metavar="MODEL")]
    # The network name
    network_name: typing.Annotated[str, tyro.conf.Positional, tyro.conf.arg(metavar="NETWORK")]
    # Arguments for physical model
    physics_args: typing.Annotated[tuple[str, ...], tyro.conf.arg(aliases=["-P"]), tyro.conf.UseAppendAction] = ()
    # Arguments for network
    network_args: typing.Annotated[tuple[str, ...], tyro.conf.arg(aliases=["-N"]), tyro.conf.UseAppendAction] = ()

    # The job name used in checkpoint and log, leave empty to use the preset job name given by the model and network
    job_name: typing.Annotated[str | None, tyro.conf.arg(aliases=["-J"])] = None
    # The checkpoint path
    checkpoint_path: typing.Annotated[pathlib.Path, tyro.conf.arg(aliases=["-C"])] = pathlib.Path("checkpoints")
    # The log path
    log_path: typing.Annotated[pathlib.Path, tyro.conf.arg(aliases=["-L"])] = pathlib.Path("logs")
    # The manual random seed, leave empty for set seed automatically
    random_seed: typing.Annotated[int | None, tyro.conf.arg(aliases=["-S"])] = None

    def main(self) -> tuple[ModelProto, NetworkProto]:
        """
        The main function to create the model and network.
        """

        if "-h" in self.network_args or "--help" in self.network_args:
            model_dict[self.model_name].network_dict[self.network_name](object(), self.network_args)
        default_job_name: str = model_dict[self.model_name].preparse(self.physics_args)
        if self.job_name is None:
            self.job_name = default_job_name

        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            handlers=[logging.StreamHandler(), logging.FileHandler(self.log_path / f"{self.job_name}.log")],
            level=logging.INFO,
            format=f"[%(process)d] %(asctime)s {self.job_name}({self.network_name}) %(levelname)s: %(message)s",
        )

        logging.info("Starting script with arguments: %a", sys.argv)
        logging.info("Model: %s, Network: %s, Job: %s", self.model_name, self.network_name, self.job_name)
        logging.info("Log directory: %s, Checkpoint directory: %s", self.log_path, self.checkpoint_path)
        logging.info("Physics arguments: %a", self.physics_args)
        logging.info("Network arguments: %a", self.network_args)

        if self.random_seed is not None:
            logging.info("Setting random seed to: %d", self.random_seed)
            torch.manual_seed(self.random_seed)
        else:
            logging.info("Random seed not specified, using current seed: %d", torch.seed())

        logging.info("Disabling PyTorch's default gradient computation")
        torch.set_grad_enabled(False)

        logging.info("Loading model: %s with arguments: %a", self.model_name, self.physics_args)
        model: ModelProto = model_dict[self.model_name].parse(self.physics_args)
        logging.info("Physical model loaded successfully")

        logging.info("Initializing network: %s and initializing with model and arguments: %a", self.network_name, self.network_args)
        network: NetworkProto = model_dict[self.model_name].network_dict[self.network_name](model, self.network_args)
        logging.info("Network initialized successfully")

        logging.info("Attempting to load checkpoint")
        checkpoint_path = self.checkpoint_path / f"{self.job_name}.pt"
        if self.checkpoint_path.exists():
            logging.info("Checkpoint found at: %s, loading...", checkpoint_path)
            network.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
            logging.info("Checkpoint loaded successfully")
        else:
            logging.info("Checkpoint not found at: %s", checkpoint_path)
        logging.info("Moving model to CUDA")
        network.cuda()
        logging.info("Model moved to CUDA successfully")

        return model, network

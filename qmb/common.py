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

    # The log path
    log_path: typing.Annotated[pathlib.Path, tyro.conf.arg(aliases=["-L"])] = pathlib.Path("logs")
    # The group name, leave empty to use the preset one given by the model
    group_name: typing.Annotated[str | None, tyro.conf.arg(aliases=["-G"])] = None
    # The job name, where it is recommended to use distinct job names for runs with varying parameters
    job_name: typing.Annotated[str, tyro.conf.arg(aliases=["-J"])] = "main"
    # The manual random seed, leave empty for set seed automatically
    random_seed: typing.Annotated[int | None, tyro.conf.arg(aliases=["-S"])] = None
    # The interval to save the checkpoint
    checkpoint_interval: typing.Annotated[int, tyro.conf.arg(aliases=["-I"])] = 5
    # The device to run on
    device: typing.Annotated[torch.device, tyro.conf.arg(aliases=["-D"])] = torch.device(type="cuda", index=0)

    def folder(self) -> pathlib.Path:
        """
        Get the folder name for the current job.
        """
        assert self.group_name is not None
        return self.log_path / self.group_name / self.job_name

    def save(self, data: typing.Any, step: int) -> None:
        """
        Save data to checkpoint.
        """
        if step % self.checkpoint_interval == 0:
            torch.save(data, self.folder() / f"data.{step}.pth")
            (self.folder() / "data.pth").unlink(missing_ok=True)
            (self.folder() / "data.pth").symlink_to(f"data.{step}.pth")
        else:
            torch.save(data, self.folder() / "data.pth")

    def main(self) -> tuple[ModelProto, NetworkProto, typing.Any]:
        """
        The main function to create the model and network.
        """

        if "-h" in self.network_args or "--help" in self.network_args:
            model_dict[self.model_name].network_dict[self.network_name](object(), self.network_args)  # type: ignore[arg-type]
        default_group_name: str = model_dict[self.model_name].preparse(self.physics_args)
        if self.group_name is None:
            self.group_name = default_group_name

        self.folder().mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            handlers=[logging.StreamHandler(), logging.FileHandler(self.folder() / "run.log")],
            level=logging.INFO,
            format=f"[%(process)d] %(asctime)s {self.group_name}({self.network_name}) %(levelname)s: %(message)s",
        )

        logging.info("Starting script with arguments: %a", sys.argv)
        logging.info("Model: %s, Network: %s", self.model_name, self.network_name)
        logging.info("Log directory: %s, Group name: %s, Job name: %s", self.log_path, self.group_name, self.job_name)
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
        data: typing.Any = {}
        checkpoint_path = self.folder() / "data.pth"
        if checkpoint_path.exists():
            logging.info("Checkpoint found at: %s, loading...", checkpoint_path)
            data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            logging.info("Checkpoint loaded successfully")
        else:
            logging.info("Checkpoint not found at: %s", checkpoint_path)
        if "network" in data:
            logging.info("Loading state dict of the network")
            network.load_state_dict(data["network"])
        else:
            logging.info("Skipping loading state dict of the network")
        logging.info("Moving model to the device: %a", self.device)
        network.to(device=self.device)

        logging.info("The checkpoints will be saved every %d steps", self.checkpoint_interval)

        return model, network, data

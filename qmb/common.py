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
from .random_engine import dump_random_engine_state, load_random_engine_state


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
    current_job_name: typing.Annotated[str, tyro.conf.arg(aliases=["-J"])] = "main"
    # The parent job name, it is only used for loading the checkpoint from the parent job, leave empty to use the current job name
    parent_job_name: typing.Annotated[str | None, tyro.conf.arg(aliases=["-F"])] = None
    # The manual random seed, leave empty for set seed automatically
    random_seed: typing.Annotated[int | None, tyro.conf.arg(aliases=["-S"])] = None
    # The interval to save the checkpoint
    checkpoint_interval: typing.Annotated[int, tyro.conf.arg(aliases=["-I"])] = 5
    # The device to run on
    device: typing.Annotated[torch.device, tyro.conf.arg(aliases=["-D"])] = torch.device(type="cuda", index=0)
    # The dtype of the network, leave empty to skip modifying the dtype
    dtype: typing.Annotated[str | None, tyro.conf.arg(aliases=["-T"])] = None
    # The maximum absolute step for the process, leave empty to loop forever
    max_absolute_step: typing.Annotated[int | None, tyro.conf.arg(aliases=["-A"])] = None
    # The maximum relative step for the process, leave empty to loop forever
    max_relative_step: typing.Annotated[int | None, tyro.conf.arg(aliases=["-R"])] = None

    def __post_init__(self) -> None:
        if self.log_path is not None:
            self.log_path = pathlib.Path(self.log_path)
        if self.device is not None:
            self.device = torch.device(self.device)
        if self.max_absolute_step is not None and self.max_relative_step is not None:
            raise ValueError("Both max_absolute_step and max_relative_step are set, please set only one of them.")

    def folder(self) -> pathlib.Path:
        """
        Get the folder name for the current job.
        """
        assert self.group_name is not None
        return self.log_path / self.group_name / self.current_job_name

    def parent_folder(self) -> pathlib.Path:
        """
        Get the folder name for the current job.
        """
        assert self.group_name is not None
        return self.log_path / self.group_name / (self.parent_job_name if self.parent_job_name is not None else self.current_job_name)

    def save(self, data: typing.Any, step: int) -> None:
        """
        Save data to checkpoint.
        """
        data["random"] = {"host": torch.get_rng_state(), "device": dump_random_engine_state(self.device), "device_type": self.device.type}
        data_pth = self.folder() / "data.pth"
        local_data_pth = self.folder() / f"data.{step}.pth"
        torch.save(data, local_data_pth)
        data_pth.unlink(missing_ok=True)
        if step % self.checkpoint_interval == 0:
            data_pth.symlink_to(f"data.{step}.pth")
        else:
            local_data_pth.rename(data_pth)
        if self.max_relative_step is not None:
            self.max_absolute_step = step + self.max_relative_step - 1
            self.max_relative_step = None
        if step == self.max_absolute_step:
            logging.info("Reached the maximum step, exiting.")
            sys.exit(0)

    def main(self, *, model_param: typing.Any = None, network_param: typing.Any = None) -> tuple[ModelProto, NetworkProto, typing.Any]:
        """
        The main function to create the model and network.
        """

        # pylint: disable=too-many-statements
        # pylint: disable=too-many-branches

        model_t = model_dict[self.model_name]
        model_config_t = model_t.config_t
        network_config_t = model_t.network_dict[self.network_name]

        if "-h" in self.network_args or "--help" in self.network_args:
            tyro.cli(network_config_t, args=self.network_args)
        if self.group_name is None:
            model_param_for_group_name = tyro.cli(model_config_t, args=self.physics_args)
            self.group_name = model_t.default_group_name(model_param_for_group_name)

        self.folder().mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            handlers=[logging.StreamHandler(), logging.FileHandler(self.folder() / "run.log")],
            level=logging.INFO,
            format=f"[%(process)d] %(asctime)s {self.group_name}({self.current_job_name}) %(levelname)s: %(message)s",
        )

        logging.info("Starting script with arguments: %a", sys.argv)
        logging.info("Model: %s, Network: %s", self.model_name, self.network_name)
        logging.info("Log directory: %s, Group name: %s, Job name: %s", self.log_path, self.group_name, self.current_job_name)
        if model_param is not None:
            logging.info("Model parameters: %a", model_param)
        else:
            logging.info("Physics arguments: %a", self.physics_args)
        if network_param is not None:
            logging.info("Network parameters: %a", network_param)
        else:
            logging.info("Network arguments: %a", self.network_args)

        logging.info("Disabling PyTorch's default gradient computation")
        torch.set_grad_enabled(False)

        logging.info("Attempting to load checkpoint")
        data: typing.Any = {}
        checkpoint_path = self.parent_folder() / "data.pth"
        if checkpoint_path.exists():
            logging.info("Checkpoint found at: %s, loading...", checkpoint_path)
            data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            logging.info("Checkpoint loaded successfully")
        else:
            if self.parent_job_name is not None:
                raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
            logging.info("Checkpoint not found at: %s", checkpoint_path)

        if self.random_seed is not None:
            logging.info("Setting random seed to: %d", self.random_seed)
            torch.manual_seed(self.random_seed)
        elif "random" in data:
            logging.info("Loading random seed from the checkpoint")
            torch.set_rng_state(data["random"]["host"])
            if data["random"]["device_type"] == self.device.type:
                load_random_engine_state(data["random"]["device"], self.device)
            else:
                logging.info("Skipping loading random engine state for device since the device type does not match")
        else:
            logging.info("Random seed not specified, using current seed: %d", torch.seed())

        if model_param is None:
            logging.info("Parsing configurations for model: %s with arguments: %a", self.model_name, self.physics_args)
            model_param = tyro.cli(model_config_t, args=self.physics_args)
        else:
            logging.info("The model parameters are given as %a, skipping parsing model arguments", model_param)
        logging.info("Loading the model")
        model: ModelProto = model_t(model_param)
        logging.info("Physical model loaded successfully")

        if network_param is None:
            logging.info("Parsing configurations for network: %s with arguments: %a", self.network_name, self.network_args)
            network_param = tyro.cli(network_config_t, args=self.network_args)
        else:
            logging.info("The network parameters are given as %a, skipping parsing network arguments", network_param)
        logging.info("Initializing the network")
        network: NetworkProto = network_param.create(model)
        logging.info("Network initialized successfully")

        if "network" in data:
            logging.info("Loading state dict of the network")
            network.load_state_dict(data["network"])
        else:
            logging.info("Skipping loading state dict of the network")

        logging.info("Moving model to the device: %a", self.device)
        network.to(device=self.device)
        if self.dtype is not None:
            logging.info("Changing network dtype to: %s", self.dtype)
            match self.dtype:
                case "bfloat16":
                    network.bfloat16()
                case "float16":
                    network.half()
                case "float32":
                    network.float()
                case "float64":
                    network.double()
                case _:
                    raise ValueError(f"Unknown dtype: {self.dtype}")

        logging.info("Compiling the network")
        network = torch.jit.script(network)  # type: ignore[assignment]

        logging.info("The checkpoints will be saved every %d steps", self.checkpoint_interval)

        return model, network, data

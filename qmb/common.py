import os
import sys
import logging
import typing
import pathlib
import dataclasses
import torch
import tyro
from . import openfermion
from . import openfermion_operator
from . import ising

model_dict = {
    "openfermion": openfermion.Model,
    "openfermion_operator": openfermion_operator.Model,
    "ising": ising.Model,
}


@dataclasses.dataclass
class CommonConfig:
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

    def main(self):
        if "-h" in self.network_args or "--help" in self.network_args:
            getattr(model_dict[self.model_name], self.network_name)(object(), self.network_args)
        default_job_name = model_dict[self.model_name].preparse(self.physics_args)
        if self.job_name is None:
            self.job_name = default_job_name

        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

        logging.basicConfig(
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{self.log_path}/{self.job_name}.log")],
            level=logging.INFO,
            format=f"[%(process)d] %(asctime)s {self.job_name}({self.network_name}) %(levelname)s: %(message)s",
        )

        logging.info("%s script start, with %a", os.path.splitext(os.path.basename(sys.argv[0]))[0], sys.argv)
        logging.info("model name: %s, network name: %s, job name: %s", self.model_name, self.network_name, self.job_name)
        logging.info("log path: %s, checkpoint path: %s", self.log_path, self.checkpoint_path)
        logging.info("arguments will be passed to network parser: %a", self.network_args)
        logging.info("arguments will be passed to physics parser: %a", self.physics_args)

        if self.random_seed is not None:
            logging.info("setting random seed to %d", self.random_seed)
            torch.manual_seed(self.random_seed)
        else:
            logging.info("random seed not set, using %d", torch.seed())

        logging.info("disabling torch default gradient behavior")
        torch.set_grad_enabled(False)

        logging.info("loading %s model as physical model", self.model_name)
        model = model_dict[self.model_name].parse(self.physics_args)
        logging.info("the physical model has been loaded")

        logging.info("loading network %s and create network with physical model and args %s", self.network_name, self.network_args)
        network = getattr(model, self.network_name)(self.network_args)
        logging.info("network created")

        logging.info("trying to load checkpoint")
        if os.path.exists(f"{self.checkpoint_path}/{self.job_name}.pt"):
            logging.info("checkpoint found, loading")
            network.load_state_dict(torch.load(f"{self.checkpoint_path}/{self.job_name}.pt", map_location="cpu", weights_only=True))
            logging.info("checkpoint loaded")
        else:
            logging.info("checkpoint not found")
        logging.info("moving model to cuda")
        network.cuda()
        logging.info("model has been moved to cuda")

        return model, network

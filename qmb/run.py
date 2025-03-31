"""
This file implements the run command for the qmb package, which is different from the other commands.
It is used to run a configuration file that contains the settings for a model and network, instead of using command line arguments.
And it is not a specific algorithm, but a general command to run any other command with a configuration file.
"""

import typing
import dataclasses
import pathlib
import yaml
import tyro
from .subcommand_dict import subcommand_dict
from .model_dict import model_dict
from .common import CommonConfig


@dataclasses.dataclass
class RunConfig:
    """
    The execution of the configuration file with other specific commands.
    """

    # The configuration file name
    file_name: typing.Annotated[pathlib.Path, tyro.conf.Positional, tyro.conf.arg(metavar="CONFIG")]

    def main(self) -> None:
        """
        Run the configuration file.
        """
        with open(self.file_name, "rt", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        common_data = data.pop("common")
        physics_data = data.pop("physics")
        network_data = data.pop("network")
        script, param = next(iter(data.items()))

        common = CommonConfig(**common_data)
        run = subcommand_dict[script](**param, common=common)

        model_t = model_dict[common.model_name]
        model_config_t = model_t.config_t
        network_config_t = model_t.network_dict[common.network_name]

        network_param = network_config_t(**network_data)
        model_param = model_config_t(**physics_data)

        run.main(model_param=model_param, network_param=network_param)  # type: ignore[call-arg]


subcommand_dict["run"] = RunConfig

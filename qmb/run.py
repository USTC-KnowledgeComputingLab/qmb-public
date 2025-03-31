"""
This is an alternative entry point for the command line application..

It read a configurtion in YAML format and run the corresponding program.
"""

import sys
import typing
import yaml
from . import cuda_limit as _  # type: ignore[no-redef]
from . import openfermion as _  # type: ignore[no-redef]
from . import fcidump as _  # type: ignore[no-redef]
from . import ising as _  # type: ignore[no-redef]
from . import vmc as _  # type: ignore[no-redef]
from . import imag as _  # type: ignore[no-redef]
from . import precompile as _  # type: ignore[no-redef]
from . import list_loss as _  # type: ignore[no-redef]
from . import chop_imag as _  # type: ignore[no-redef]
from .subcommand_dict import subcommand_dict
from .model_dict import model_dict
from .common import CommonConfig


def main() -> None:
    """
    The main function for the command line application.
    """
    file_name = sys.argv[1]
    with open(file_name, "rt", encoding="utf-8") as file:
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

    network_param: typing.Any = network_config_t(**network_data)
    model_param: typing.Any = model_config_t(**physics_data)

    run.main(model_param=model_param, network_param=network_param)  # type: ignore[call-arg]


if __name__ == "__main__":
    main()

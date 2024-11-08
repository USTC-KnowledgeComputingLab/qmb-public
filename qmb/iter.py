"""
This file implements the direct iterative method for solving quantum many-body problems.
"""

import logging
import typing
import dataclasses
import tyro
from .common import CommonConfig
from .subcommand_dict import subcommand_dict
from .utility import extend_and_select, lobpcg_and_select


@dataclasses.dataclass
class IterConfig:
    """
    The direct iterative method for solving quantum many-body problems.
    """

    common: typing.Annotated[CommonConfig, tyro.conf.OmitArgPrefixes]

    # sampling count
    sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1024

    # selected extended sampling count
    selected_sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-s"])] = 65536

    def main(self) -> None:
        """
        The main function of the direct iterative method.
        """

        model, network = self.common.main()

        logging.info(
            "sampling count: %d, selected sampling count: %d",
            self.sampling_count,
            self.selected_sampling_count,
        )

        logging.info("first sampling core configurations")
        configs_core, psi_core, _, _ = network.generate_unique(self.sampling_count)
        logging.info("core configurations sampled")

        while True:
            logging.info("extend and select start")
            configs_extended, psi_extended = extend_and_select(model, configs_core, psi_core, self.selected_sampling_count)
            logging.info("extend and select finished")

            logging.info("lobpcg and select start")
            _, _, configs_core, psi_core = lobpcg_and_select(model, configs_extended, psi_extended, self.sampling_count)
            logging.info("lobpcg and select finished")


subcommand_dict["iter"] = IterConfig

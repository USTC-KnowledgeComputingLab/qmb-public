"""
This file implements the direct iterative method for solving quantum many-body problems.
"""

import logging
import typing
import dataclasses
import tyro
from .common import CommonConfig
from .subcommand_dict import subcommand_dict
from .utility import extend_with_select, select_by_lobpcg


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
            "Arguments Summary: "
            "Sampling count: %d, "
            "Selected sampling count: %d",
            self.sampling_count,
            self.selected_sampling_count,
        )

        logging.info("Initiating initial core configuration sampling")
        configs_core, psi_core, _, _ = network.generate_unique(self.sampling_count)
        logging.info("Core configurations successfully sampled")

        while True:
            logging.info("Starting a new optimization cycle")
            configs_extended, psi_extended = extend_with_select(model, configs_core, psi_core, self.selected_sampling_count)
            configs_core, psi_core = select_by_lobpcg(model, configs_extended, psi_extended, self.sampling_count)
            logging.info("Current optimization cycle completed")


subcommand_dict["iter"] = IterConfig

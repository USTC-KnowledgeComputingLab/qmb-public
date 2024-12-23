"""
This file lists loss functions used in the direct optimization of wavefunction.
"""

import dataclasses
import inspect
import torch
from . import losses
from .subcommand_dict import subcommand_dict


@dataclasses.dataclass
class ListLossConfig:
    """
    The list of loss functions used in the direct optimization of wavefunction.
    """

    def main(self) -> None:
        """
        The main function for listing loss functions.
        """
        for name, member in inspect.getmembers(losses):
            if isinstance(member, torch.jit.ScriptFunction) and not name.startswith("_"):
                print(name, member.__doc__)


subcommand_dict["list_loss"] = ListLossConfig

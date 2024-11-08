"""
This module is used to store a dictionary that maps subcommand names to their corresponding dataclass types.
Other packages can register their subcommands by adding entries to this dictionary.
"""

import typing


class SubcommandProto(typing.Protocol):
    """
    This protocol defines a dataclass with a `main` method, which will be called when the subcommand is executed.
    """

    # pylint: disable=too-few-public-methods

    def main(self) -> None:
        """
        The main method to be called when the subcommand is executed.
        """


subcommand_dict: dict[str, typing.Callable[..., SubcommandProto]] = {}

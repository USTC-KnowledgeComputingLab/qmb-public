"""
This dictionary is used to store subcommands. Other parts of the program can register their subcommands here.
"""

import typing


class DataclassWithMain(typing.Protocol):
    """
    This protocol defines a dataclass with a `main` method, which will be called when the subcommand is executed.
    """

    # pylint: disable=too-few-public-methods

    def main(self) -> None:
        """
        The main method to be called when the subcommand is executed.
        """


subcommand_dict: dict[str, typing.Callable[..., DataclassWithMain]] = {}

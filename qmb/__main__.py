"""
This is the main entry point for the command line application.
"""

import tyro
from . import openfermion as _  # type: ignore[no-redef]
from . import fcidump as _  # type: ignore[no-redef]
from . import ising as _  # type: ignore[no-redef]
from . import learn as _  # type: ignore[no-redef]
from . import vmc as _  # type: ignore[no-redef]
from . import iter as _  # type: ignore[no-redef]
from . import imag as _  # type: ignore[no-redef]
from .subcommand_dict import subcommand_dict


def main() -> None:
    """
    Main function for the command line application.
    """
    tyro.extras.subcommand_cli_from_dict(subcommand_dict).main()


if __name__ == "__main__":
    main()

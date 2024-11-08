import tyro
from . import openfermion as _
from . import fcidump as _
from . import ising as _
from . import learn as _
from . import vmc as _
from . import iter as _
from .subcommand_dict import subcommand_dict


def main() -> None:
    tyro.extras.subcommand_cli_from_dict(subcommand_dict).main()


if __name__ == "__main__":
    main()

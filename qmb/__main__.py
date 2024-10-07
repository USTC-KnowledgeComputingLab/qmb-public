import tyro
from . import learn as _
from . import vmc as _
from . import iter as _
from .subcommand_dict import subcommand_dict


def main():
    tyro.extras.subcommand_cli_from_dict(subcommand_dict).main()


if __name__ == "__main__":
    main()

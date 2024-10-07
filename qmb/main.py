import tyro
from .learn import LearnConfig
from .vmc import VmcConfig


def main():
    tyro.extras.subcommand_cli_from_dict({
        "learn": LearnConfig,
        "vmc": VmcConfig,
    }).main()


if __name__ == "__main__":
    main()

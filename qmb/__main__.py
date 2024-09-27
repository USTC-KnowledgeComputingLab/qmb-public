import tyro
from .learn import LearnConfig
from .vmc import VmcConfig
from .iter import IterConfig


def main():
    tyro.extras.subcommand_cli_from_dict({
        "learn": LearnConfig,
        "vmc": VmcConfig,
        "iter": IterConfig,
    }).main()


if __name__ == "__main__":
    main()

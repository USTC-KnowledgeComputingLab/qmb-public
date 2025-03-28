import sys
import json
import tyro
from . import __main__
from .subcommand_dict import subcommand_dict
from .common import CommonConfig


def main() -> None:
    file_name = sys.argv[1]
    with open(file_name, "rt", encoding="utf8") as file:
        config = json.load(file)
    config["common"] = CommonConfig.load(config["common"])
    script = config.pop("script")
    instance = subcommand_dict[script](**config)
    instance.main()


if __name__ == "__main__":
    main()

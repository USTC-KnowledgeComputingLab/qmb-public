import sys
import json
import tyro
from . import __main__
from .subcommand_dict import subcommand_dict


def main() -> None:
    file_name = sys.argv[1]
    script = sys.argv[2]
    args = sys.argv[3:]
    params = tyro.cli(subcommand_dict[script], args=args)
    config = vars(params).copy()
    config["script"] = script
    config["common"] = config["common"].dump()
    with open(file_name, "wt", encoding="utf8") as file:
        json.dump(config, file, indent=4)


if __name__ == "__main__":
    main()

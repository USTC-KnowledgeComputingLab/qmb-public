import sys
import tyro
from . import cuda_limit as _  # type: ignore[no-redef]
from . import openfermion as _  # type: ignore[no-redef]
from . import fcidump as _  # type: ignore[no-redef]
from . import ising as _  # type: ignore[no-redef]
from . import vmc as _  # type: ignore[no-redef]
from . import imag as _  # type: ignore[no-redef]
from . import precompile as _  # type: ignore[no-redef]
from . import list_loss as _  # type: ignore[no-redef]
from . import chop_imag as _  # type: ignore[no-redef]
from .subcommand_dict import subcommand_dict
from .common import CommonConfig
from .model_dict import model_dict


def main() -> None:
    file_name = sys.argv[1]
    import yaml
    data = yaml.safe_load(open(file_name, "rt"))
    common = data.pop("common")
    physics = data.pop("physics")
    network = data.pop("network")
    script, param = next(iter(data.items()))
    common_obj = CommonConfig(**common)
    run_obj = subcommand_dict[script](**param, common=common_obj)

    model_t = model_dict[common_obj.model_name]
    network_t = model_t.network_dict[common_obj.network_name]
    network_param = network_t(**network)
    model_param = model_t.config_t(**physics)
    run_obj.main(model_param, network_param)


if __name__ == "__main__":
    main()

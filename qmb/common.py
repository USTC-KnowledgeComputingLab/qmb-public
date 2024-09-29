import os
import sys
import logging
import torch
from .version import version
from . import openfermion
from . import openfermion_operator
from . import ising

model_dict = {
    "openfermion": openfermion.Model,
    "openfermion_operator": openfermion_operator.Model,
    "ising": ising.Model,
}


def initialize_process(parser):
    group = parser.add_argument_group("model and network")
    group.add_argument("model_name", help="model name")
    group.add_argument("network_name", help="network name")
    group.add_argument("-P", dest="physics_args", type=str, default=[], action="append", help="arguments for physical model")
    group.add_argument("-N", dest="network_args", type=str, default=[], action="append", help="arguments for network")

    group = parser.add_argument_group("miscellaneous")
    group.add_argument("-J", "--job-name", dest="job_name", type=str, default=None, help="the run name")
    group.add_argument("-C", "--checkpoint-path", dest="checkpoint_path", type=str, default="checkpoints", help="path of checkpoints folder")
    group.add_argument("-L", "--log-path", dest="log_path", type=str, default="logs", help="path of logs folder")
    group.add_argument("-S", "--random-seed", dest="random_seed", type=int, default=None, help="the manual random seed")
    group.add_argument("-v", "--version", action="version", version=f'%(prog)s {version}')

    args = parser.parse_args()

    if "-h" in args.network_args or "--help" in args.network_args:
        model_dict[args.model_name].naqs(object(), args.network_args)
    default_job_name = model_dict[args.model_name].preparse(args.physics_args)
    if args.job_name is None:
        args.job_name = default_job_name

    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    logging.basicConfig(
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{args.log_path}/{args.job_name}.log")],
        level=logging.INFO,
        format=f"[%(process)d] %(asctime)s {args.job_name}({args.network_name}) %(levelname)s: %(message)s",
    )

    logging.info("%s script start, with %a", os.path.splitext(os.path.basename(sys.argv[0]))[0], sys.argv)
    logging.info("model name: %s, network name: %s, job name: %s", args.model_name, args.network_name, args.job_name)
    logging.info("log path: %s, checkpoint path: %s", args.log_path, args.checkpoint_path)
    logging.info("arguments will be passed to network parser: %a", args.network_args)
    logging.info("arguments will be passed to physics parser: %a", args.physics_args)

    if args.random_seed is not None:
        logging.info("setting random seed to %d", args.random_seed)
        torch.manual_seed(args.random_seed)
    else:
        logging.info("random seed not set, using %d", torch.seed())

    logging.info("disabling torch default gradient behavior")
    torch.set_grad_enabled(False)

    logging.info("loading %s model as physical model", args.model_name)
    model = model_dict[args.model_name].parse(args.physics_args)
    logging.info("the physical model has been loaded")

    logging.info("loading network %s and create network with physical model and args %s", args.network_name, args.network_args)
    network = getattr(model, args.network_name)(args.network_args)
    logging.info("network created")

    logging.info("trying to load checkpoint")
    if os.path.exists(f"{args.checkpoint_path}/{args.job_name}.pt"):
        logging.info("checkpoint found, loading")
        network.load_state_dict(torch.load(f"{args.checkpoint_path}/{args.job_name}.pt", map_location="cpu", weights_only=True))
        logging.info("checkpoint loaded")
    else:
        logging.info("checkpoint not found")
    logging.info("moving model to cuda")
    network.cuda()
    logging.info("model has been moved to cuda")

    return args, model, network

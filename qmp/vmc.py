import argparse
import logging
import numpy
import scipy
import torch
from .common import initialize_process


def main():
    parser = argparse.ArgumentParser(description="approach to the ground state for the quantum manybody problem", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--sampling-count", dest="sampling_count", type=int, default=4000, help="sampling count")
    parser.add_argument("-r", "--learning-rate", dest="learning_rate", type=float, default=None, help="learning rate for the local optimizer")
    parser.add_argument("-s", "--local-step", dest="local_step", type=int, default=1000, help="step count for the local optimizer")
    parser.add_argument("-o", "--include-outside", dest="include_outside", action="store_true", help="calculate all psi(s')")
    parser.add_argument("-2", "--lbfgs", dest="use_lbfgs", action="store_true", help="Use LBFGS instead of Adam")

    args, model, network = initialize_process(parser)
    if args.learning_rate is None:
        args.learning_rate = 1 if args.use_lbfgs else 1e-3

    logging.info(
        "sampling count: %d, learning rate: %f, local step: %d, include outside: %a, use lbfgs: %a",
        args.sampling_count,
        args.learning_rate,
        args.local_step,
        args.include_outside,
        args.use_lbfgs,
    )

    logging.info("main looping")
    while True:
        logging.info("sampling configurations")
        configs_i, _, _, _ = network.generate_unique(args.sampling_count)
        logging.info("sampling done")
        unique_sampling_count = len(configs_i)
        logging.info("unique sampling count is %d", unique_sampling_count)

        if args.include_outside:
            logging.info("generating hamiltonian data to create sparse matrix outsidely")
            indices_i_and_j, values, configs_j = model.outside(configs_i.cpu())
            logging.info("sparse matrix data created")
            outside_count = len(configs_j)
            logging.info("outside configs count is %d", outside_count)
            logging.info("converting sparse matrix data to sparse matrix")
            hamiltonian = torch.sparse_coo_tensor(indices_i_and_j.T, values, [unique_sampling_count, outside_count], dtype=torch.complex128).to_sparse_csr().cuda()
            logging.info("sparse matrix created")
            logging.info("moving configs j to cuda")
            configs_j = torch.tensor(configs_j).cuda()
            logging.info("configs j has been moved to cuda")
        else:
            logging.info("generating hamiltonian data to create sparse matrix insidely")
            indices_i_and_j, values = model.inside(configs_i.cpu())
            logging.info("sparse matrix data created")
            logging.info("converting sparse matrix data to sparse matrix")
            hamiltonian = torch.sparse_coo_tensor(indices_i_and_j.T, values, [unique_sampling_count, unique_sampling_count], dtype=torch.complex128).to_sparse_csr().cuda()
            logging.info("sparse matrix created")

        if args.use_lbfgs:
            optimizer = torch.optim.LBFGS(network.parameters(), lr=args.learning_rate)
        else:
            optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)

        def closure():
            optimizer.zero_grad()
            amplitudes_i = network(configs_i)
            with torch.no_grad():
                if args.include_outside:
                    amplitudes_j = network(configs_j)
                else:
                    amplitudes_j = amplitudes_i
            energy = ((amplitudes_i.conj() @ (hamiltonian @ amplitudes_j.detach())) / (amplitudes_i.conj() @ amplitudes_i.detach())).real
            energy.backward()
            return energy

        logging.info("local optimization starting")
        for i in range(args.local_step):
            energy = optimizer.step(closure)
            logging.info("local optimizing, step %d, energy: %.10f", i, energy.item())
        logging.info("local optimization finished")
        logging.info("saving checkpoint")
        torch.save(network.state_dict(), f"{args.checkpoint_path}/{args.job_name}.pt")
        logging.info("checkpoint saved")


if __name__ == "__main__":
    main()

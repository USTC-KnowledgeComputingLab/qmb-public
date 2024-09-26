import argparse
import logging
import numpy
import scipy
import torch
from . import losses
from .common import initialize_process


def main():
    parser = argparse.ArgumentParser(description="approach to the ground state for the quantum manybody problem", formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars="@")
    parser.add_argument("-n", "--sampling-count", dest="sampling_count", type=int, default=4000, help="sampling count")
    parser.add_argument("-r", "--learning-rate", dest="learning_rate", type=float, default=None, help="learning rate for the local optimizer")
    parser.add_argument("-s", "--local-step", dest="local_step", type=int, default=None, help="step count for the local optimizer")
    parser.add_argument("-t", "--local-loss", dest="local_loss", type=float, default=1e-8, help="early break loss threshold for local optimization")
    parser.add_argument("-p", "--logging-psi", dest="logging_psi", type=int, default=30, help="psi count to be printed after local optimizer")
    parser.add_argument("-l", "--loss-name", dest="loss_name", type=str, default="log", help="the loss function to be used")
    parser.add_argument("-2", "--lbfgs", dest="use_lbfgs", action="store_true", help="Use LBFGS instead of Adam")

    args, model, network = initialize_process(parser)
    if args.learning_rate is None:
        args.learning_rate = 1 if args.use_lbfgs else 1e-3
    if args.local_step is None:
        args.local_step = 400 if args.use_lbfgs else 1000

    logging.info(
        "sampling count: %d, learning rate: %f, local step: %d, local loss: %f, logging psi: %d, loss name: %s, use_lbfgs: %a",
        args.sampling_count,
        args.learning_rate,
        args.local_step,
        args.local_loss,
        args.logging_psi,
        args.loss_name,
        args.use_lbfgs,
    )

    logging.info("main looping")
    while True:
        logging.info("sampling configurations")
        configs, pre_amplitudes, _, _ = network.generate_unique(args.sampling_count)
        logging.info("sampling done")
        unique_sampling_count = len(configs)
        logging.info("unique sampling count is %d", unique_sampling_count)

        logging.info("generating hamiltonian data to create sparse matrix")
        indices_i_and_j, values = model.inside(configs.cpu())
        logging.info("sparse matrix data created")
        logging.info("converting sparse matrix data to sparse matrix")
        hamiltonian = scipy.sparse.coo_matrix((values, indices_i_and_j.T), [unique_sampling_count, unique_sampling_count], dtype=numpy.complex128).tocsr()
        logging.info("sparse matrix created")
        logging.info("estimating ground state")
        target_energy, targets = scipy.sparse.linalg.lobpcg(hamiltonian, pre_amplitudes.cpu().reshape([-1, 1]).detach().numpy(), largest=False, maxiter=1024)
        logging.info("estimiated, target energy is %.10f, ref energy is %.10f", target_energy.item(), model.ref_energy)
        logging.info("preparing learning targets")
        targets = torch.tensor(targets).view([-1]).cuda()
        max_index = targets.abs().argmax()
        targets = targets / targets[max_index]

        logging.info("choosing loss function as %s", args.loss_name)
        loss_func = getattr(losses, args.loss_name)

        if args.use_lbfgs:
            optimizer = torch.optim.LBFGS(network.parameters(), lr=args.learning_rate)
        else:
            optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
        loss = None
        amplitudes = None

        def closure():
            nonlocal loss, amplitudes
            optimizer.zero_grad()
            amplitudes = network(configs)
            amplitudes = amplitudes / amplitudes[max_index]
            loss = loss_func(amplitudes, targets)
            loss.backward()
            return loss

        logging.info("local optimization starting")
        for i in range(args.local_step):
            optimizer.step(closure)
            logging.info("local optimizing, step %d, loss %.10f", i, loss.item())
            if loss < args.local_loss:
                logging.info("local optimization stop since local loss reached")
                break

        logging.info("local optimization finished")
        logging.info("saving checkpoint")
        torch.save(network.state_dict(), f"{args.checkpoint_path}/{args.job_name}.pt")
        logging.info("checkpoint saved")
        logging.info("calculating current energy")
        torch.enable_grad(closure)()
        amplitudes = amplitudes.cpu().detach().numpy()
        final_energy = ((amplitudes.conj() @ (hamiltonian @ amplitudes)) / (amplitudes.conj() @ amplitudes)).real
        logging.info(
            "loss = %.10f during local optimization, final energy %.10f, target energy %.10f, ref energy %.10f",
            loss.item(),
            final_energy.item(),
            target_energy.item(),
            model.ref_energy,
        )
        logging.info("printing several largest amplitudes")
        indices = targets.abs().sort(descending=True).indices
        for index in indices[:args.logging_psi]:
            logging.info("config %s, target %s, final %s", "".join(map(str, configs[index].cpu().numpy())), f"{targets[index].item():.8f}", f"{amplitudes[index].item():.8f}")


if __name__ == "__main__":
    main()

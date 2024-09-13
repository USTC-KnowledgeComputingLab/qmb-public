import os
import sys
import logging
import argparse
import numpy
import scipy
import torch
import openfermion
import naqs_network
import openfermion_to_sparse
import loss_function


def main(*, model, hidden, sampling_count, lr, local_step, logging_psi_count, loss_name, log_path, checkpoint_path, model_path):
    logging.basicConfig(
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{log_path}/{model}.log")],
        level=logging.INFO,
        format='%(asctime)s - ' + model + ' - %(levelname)s - %(message)s',
    )
    logging.info("learn script start, hidden: %a, sampling_count: %d, lr: %f, local_step: %d", hidden, sampling_count, lr, local_step)

    logging.info("loading physical model %s", model)
    physical_model = openfermion.MolecularData(filename=f"{model_path}/{model}.hdf5")
    logging.info("converting physical model to python list")
    openfermion_hamiltonian = list(openfermion.transforms.get_fermion_operator(physical_model.get_molecular_hamiltonian()).terms.items())
    logging.info("creating neural network")
    network = naqs_network.WaveFunction(
        double_sites=physical_model.n_qubits,
        physical_dim=2,
        is_complex=True,
        spin_up=physical_model.n_electrons // 2,
        spin_down=physical_model.n_electrons // 2,
        hidden_size=hidden,
        ordering=+1,
    ).double()
    if os.path.exists(f"{checkpoint_path}/{model}.pt"):
        logging.info("checkpoint found, loading")
        network.load_state_dict(torch.load(f"{checkpoint_path}/{model}.pt", map_location="cpu"))
    else:
        logging.info("checkpoint not found")
    logging.info("moving model to cuda")
    network.cuda()

    logging.info("looping")
    while True:
        logging.info("sampling")
        configs, pre_amplitudes, _, _ = network.generate_unique(sampling_count)
        logging.info("converting sampling configs to tuple")
        tuple_configs = [tuple(config.view([-1]).tolist()) for config in configs]
        logging.info("counting unique sampling count")
        unique_sampling_count = len(tuple_configs)
        logging.info("unique sampling count is %d", unique_sampling_count)

        logging.info("generating hamiltonian as sparse matrix data")
        indices_i, indices_j, values = openfermion_to_sparse.openfermion_to_sparse(openfermion_hamiltonian, tuple_configs)
        logging.info("converting sparse matrix data to coo matrix")
        hamiltonian = scipy.sparse.coo_matrix((values, (indices_i, indices_j)), [unique_sampling_count, unique_sampling_count], dtype=numpy.complex128)
        logging.info("estimating ground state")
        expected_energy, targets = scipy.sparse.linalg.lobpcg(hamiltonian, pre_amplitudes.cpu().reshape([-1, 1]).detach().numpy(), largest=False, maxiter=1024)
        logging.info("estimiated, target energy is %f, fci energy is %f", expected_energy.item(), physical_model.fci_energy.item())
        logging.info("preparing learning targets")
        targets = torch.tensor(targets).view([-1])
        targets = targets.cuda()
        max_index = targets.abs().argmax()
        targets = targets / targets[max_index]

        logging.info("local optimization starting")
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        loss_func = getattr(loss_function, loss_name)
        for i in range(local_step):
            amplitudes = network(configs)
            amplitudes = amplitudes / amplitudes[max_index]
            loss = loss_func(amplitudes, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            logging.info("local optimizing, step %d, loss %f", i, loss.item())
        logging.info("local optimization finished")
        logging.info("saving checkpoint")
        torch.save(network.state_dict(), f"{checkpoint_path}/{model}.pt")
        logging.info("calculating current energy")
        amplitudes = amplitudes.cpu().detach().numpy()
        final_energy = ((amplitudes.conj() @ hamiltonian @ amplitudes) / (amplitudes.conj() @ amplitudes)).real
        logging.info(
            "loss = %f during local optimization, final energy %f, target energy %f, fci energy %f",
            loss.item(),
            final_energy.item(),
            expected_energy.item(),
            physical_model.fci_energy.item(),
        )
        logging.info("printing several largest amplitudes")
        indices = targets.abs().sort(descending=True).indices
        for index in indices[:logging_psi_count]:
            logging.info("config %s, target %s, final %s", "".join(map(str, tuple_configs[index])), f"{targets[index].item():.4f}", f"{amplitudes[index].item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="approach to the ground state for the quantum chemistry many body system", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("-n", "--sampling-count", dest="sampling_count", type=int, default=4000, help="sampling count")
    parser.add_argument("-w", "--hidden-width", dest="hidden", type=int, default=[512], nargs="+", help="hidden width of the network")
    parser.add_argument("-r", "--learning-rate", dest="lr", type=float, default=1e-3, help="learning rate for the local optimizer")
    parser.add_argument("-s", "--local-step", dest="local_step", type=int, default=1000, help="step count for the local optimizer")
    parser.add_argument("-p", "--logging-psi-count", dest="logging_psi_count", type=int, default=30, help="psi count to be printed after local optimizer")
    parser.add_argument("-l", "--loss-name", dest="loss_name", type=str, default="log", help="the loss function to be used")
    parser.add_argument("-L", "--log-path", dest="log_path", type=str, default="logs", help="path of logs folder")
    parser.add_argument("-C", "--checkpoint-path", dest="checkpoint_path", type=str, default="checkpoints", help="path of checkpoints folder")
    parser.add_argument("-M", "--model-path", dest="model_path", type=str, default="models", help="path of models folder")
    args = parser.parse_args()
    main(
        model=args.model,
        hidden=args.hidden,
        sampling_count=args.sampling_count,
        lr=args.lr,
        local_step=args.local_step,
        logging_psi_count=args.logging_psi_count,
        loss_name=args.loss_name,
        log_path=args.log_path,
        checkpoint_path=args.checkpoint_path,
        model_path=args.model_path,
    )

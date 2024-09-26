import os
import sys
import logging
import argparse
import numpy
import scipy
import torch
import openfermion
import networks
import openfermion_to_sparse
import loss_function


def main():
    parser = argparse.ArgumentParser(description="approach to the ground state for the quantum manybody problem", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model", help="model name")
    parser.add_argument("network", help="network name")
    parser.add_argument("-n", "--sampling-count", dest="sampling_count", type=int, default=64, help="sampling count")
    parser.add_argument("-L", "--log-path", dest="log_path", type=str, default="logs", help="path of logs folder")
    parser.add_argument("-C", "--checkpoint-path", dest="checkpoint_path", type=str, default="checkpoints", help="path of checkpoints folder")
    parser.add_argument("-M", "--model-path", dest="model_path", type=str, default="models", help="path of models folder")
    parser.add_argument("-N", "--run-name", dest="run_name", type=str, default=None, help="the run name")
    parser.add_argument("-S", "--random-seed", dest="random_seed", type=int, default=None, help="the manual random seed")
    parser.add_argument("-W", dest="network_args", type=str, default=[], nargs="*", help="arguments for network")
    args = parser.parse_args()
    model_name = args.model
    network_name = args.network
    sampling_count = args.sampling_count
    log_path = args.log_path
    checkpoint_path = args.checkpoint_path
    model_path = args.model_path
    run_name = args.run_name
    if run_name is None:
        run_name = model_name
    random_seed = args.random_seed
    network_args = args.network_args

    logging.basicConfig(
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{log_path}/{run_name}.log")],
        level=logging.INFO,
        format=f"[%(process)d] %(asctime)s {model_name}({network_name}) %(levelname)s: %(message)s",
    )

    logging.info("iter script start, with %a", sys.argv)
    logging.info("model: %s, network: %s, run name: %s", model_name, network_name, run_name)
    logging.info("sampling count: %d", sampling_count)
    logging.info("log path: %s, checkpoint path: %s, model path: %s", log_path, checkpoint_path, model_path)
    logging.info("arguments will be passed to network parser: %a", network_args)

    if random_seed is not None:
        logging.info("setting random seed to %d", random_seed)
        torch.manual_seed(random_seed)
    else:
        logging.info("random seed not set, using %d", torch.seed())

    logging.info("disabling torch default gradient behavior")
    torch.set_grad_enabled(False)

    model_file_name = f"{model_path}/{model_name}.hdf5"
    logging.info("loading physical model %s from %s", model_name, model_file_name)
    physical_model = openfermion.MolecularData(filename=model_file_name)
    logging.info("physical model %s loaded", model_name)

    if hasattr(physical_model, "fci_energy"):
        fci_energy = physical_model.fci_energy.item()
        logging.info("fci energy in model data is %.10f", fci_energy)
    else:
        fci_energy = numpy.nan
        logging.info("fci energy in model data does not exist")

    logging.info("converting physical model to hamiltonian handle")
    openfermion_hamiltonian = openfermion_to_sparse.Hamiltonian(list(openfermion.transforms.get_fermion_operator(physical_model.get_molecular_hamiltonian()).terms.items()))
    logging.info("hamiltonian handle created")

    logging.info("loading network %s and create network with physical model and args %s", network_name, network_args)
    network = getattr(networks, network_name)(physical_model, network_args)
    logging.info("network created")

    logging.info("trying to load checkpoint")
    if os.path.exists(f"{checkpoint_path}/{run_name}.pt"):
        logging.info("checkpoint found, loading")
        network.load_state_dict(torch.load(f"{checkpoint_path}/{run_name}.pt", map_location="cpu"))
        logging.info("checkpoint loaded")
    else:
        logging.info("checkpoint not found")

    logging.info("first sampling core configurations")
    # TODO initialize directly rather than network
    configs_core, psi, _, _ = network.generate_unique(sampling_count)
    logging.info("core configurations sampled")

    while True:
        sampling_count_core = len(configs_core)
        logging.info("core configurations count is %d", sampling_count_core)

        logging.info("calculating extended configurations")
        _, _, configs_extended = openfermion_hamiltonian.outside(configs_core)
        logging.info("extended configurations created")
        sampling_count_extended = len(configs_extended)
        logging.info("extended configurations count is %d", sampling_count_extended)

        logging.info("calculating sparse data of hamiltonian on extended configurations")
        indices_i_and_j, values = openfermion_hamiltonian.inside(configs_extended)
        logging.info("converting sparse matrix data to coo matrix")
        hamiltonian = scipy.sparse.coo_matrix((values, indices_i_and_j.T), [sampling_count_extended, sampling_count_extended], dtype=numpy.complex128)
        logging.info("preparing initial psi used in lobpcg")
        psi = numpy.pad(psi, (0, sampling_count_extended - sampling_count_core)).reshape([-1, 1])
        # TODO estimate <n|H|psi> as sampling possibility
        logging.info("calculating minimum energy on extended configurations")
        energy, psi = scipy.sparse.linalg.lobpcg(hamiltonian, psi, largest=False, maxiter=1024)
        logging.info("energy on extended configurations is %.10f, fci energy is %.10f, error is %.10f", energy.item(), fci_energy, energy.item() - fci_energy)
        logging.info("calculating new core configurations")
        indices = numpy.argsort(numpy.abs(psi).flatten())[-sampling_count:]
        # TODO sample rather than sort
        logging.info("update new core configurations")
        configs_core = configs_extended[indices]
        psi = psi[indices].flatten()

        # TODO save


if __name__ == "__main__":
    main()

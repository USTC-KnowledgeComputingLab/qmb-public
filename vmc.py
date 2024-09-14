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


def main():
    parser = argparse.ArgumentParser(description="approach to the ground state for the quantum chemistry many body system", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model", help="model name")
    parser.add_argument("network", help="network name")
    parser.add_argument("-n", "--sampling-count", dest="sampling_count", type=int, default=4000, help="sampling count")
    parser.add_argument("-r", "--learning-rate", dest="lr", type=float, default=1e-3, help="learning rate for the local optimizer")
    parser.add_argument("-s", "--local-step", dest="local_step", type=int, default=1000, help="step count for the local optimizer")
    parser.add_argument("-o", "--include-outside", dest="include_outside", default=False, help="calculate all psi(s')", action="store_true")
    parser.add_argument("-L", "--log-path", dest="log_path", type=str, default="logs", help="path of logs folder")
    parser.add_argument("-C", "--checkpoint-path", dest="checkpoint_path", type=str, default="checkpoints", help="path of checkpoints folder")
    parser.add_argument("-M", "--model-path", dest="model_path", type=str, default="models", help="path of models folder")
    parser.add_argument("-N", "--run-name", dest="run_name", type=str, default=None, help="the run name")
    args, other_args = parser.parse_known_args()
    model_name = args.model
    network_name = args.network
    sampling_count = args.sampling_count
    lr = args.lr
    local_step = args.local_step
    include_outside = args.include_outside
    log_path = args.log_path
    checkpoint_path = args.checkpoint_path
    model_path = args.model_path
    run_name = args.run_name
    if run_name is None:
        run_name = model_name

    logging.basicConfig(
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{log_path}/{run_name}.log")],
        level=logging.INFO,
        format='%(asctime)s - ' + model_name + '(' + network_name + ') - %(levelname)s - %(message)s',
    )

    logging.info("vmc script start, with %a", sys.argv)
    logging.info("model: %s, network: %s, run name: %s", model_name, network_name, run_name)
    logging.info("sampling count: %d, learning rate: %f, local step: %d, include outside: %a", sampling_count, lr, local_step, include_outside)
    logging.info("log path: %s, checkpoint path: %s, model path: %s", log_path, checkpoint_path, model_path)
    logging.info("other arguments will be passed to network parser: %a", other_args)

    logging.info("disabling torch default gradient behavior")
    torch.set_grad_enabled(False)

    model_file_name = f"{model_path}/{model_name}.hdf5"
    logging.info("loading physical model %s from %s", model_name, model_file_name)
    physical_model = openfermion.MolecularData(filename=model_file_name)
    logging.info("physical model %s loaded", model_name)

    if hasattr(physical_model, "fci_energy"):
        fci_energy = physical_model.fci_energy.item()
        logging.info("fci energy in model data is %f", fci_energy)
    else:
        fci_energy = numpy.nan
        logging.info("fci energy in model data does not exist")

    logging.info("converting physical model to hamiltonian handle")
    openfermion_hamiltonian = openfermion_to_sparse.Hamiltonian(list(openfermion.transforms.get_fermion_operator(physical_model.get_molecular_hamiltonian()).terms.items()))
    logging.info("hamiltonian handle created")

    logging.info("loading network %s and create network with physical model and args %s", network_name, other_args)
    network = getattr(networks, network_name)(physical_model, other_args)
    logging.info("network created")

    logging.info("trying to load checkpoint")
    if os.path.exists(f"{checkpoint_path}/{run_name}.pt"):
        logging.info("checkpoint found, loading")
        network.load_state_dict(torch.load(f"{checkpoint_path}/{run_name}.pt", map_location="cpu"))
        logging.info("checkpoint loaded")
    else:
        logging.info("checkpoint not found")
    logging.info("moving model to cuda")
    network.cuda()
    logging.info("model has been moved to cuda")

    logging.info("main looping")
    while True:
        logging.info("sampling configurations")
        configs_i, _, _, _ = network.generate_unique(sampling_count)
        logging.info("sampling done")
        unique_sampling_count = len(configs_i)
        logging.info("unique sampling count is %d", unique_sampling_count)

        if include_outside:
            logging.info("generating hamiltonian data to create sparse matrix outsidely")
            indices_i_and_j, values, configs_j = openfermion_hamiltonian.outside(configs_i.cpu())
            logging.info("sparse matrix data created")
            outside_count = len(configs_j)
            logging.info("outside configs count is %d", outside_count)
            logging.info("converting sparse matrix data to coo matrix")
            hamiltonian = torch.sparse_coo_tensor(indices_i_and_j.T, values, [unique_sampling_count, outside_count], dtype=torch.complex128).cuda()
            logging.info("coo matrix created")
            logging.info("moving configs j to cuda")
            configs_j = torch.tensor(configs_j).cuda()
            logging.info("configs j has been moved to cuda")
        else:
            logging.info("generating hamiltonian data to create sparse matrix insidely")
            indices_i_and_j, values = openfermion_hamiltonian.inside(configs_i.cpu())
            logging.info("sparse matrix data created")
            logging.info("converting sparse matrix data to coo matrix")
            hamiltonian = torch.sparse_coo_tensor(indices_i_and_j.T, values, [unique_sampling_count, unique_sampling_count], dtype=torch.complex128).cuda()
            logging.info("coo matrix created")

        logging.info("local optimization starting")
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        for i in range(local_step):
            optimizer.zero_grad()
            with torch.enable_grad():
                amplitudes_i = network(configs_i)
                if include_outside:
                    amplitudes_j = network(configs_j)
                else:
                    amplitudes_j = amplitudes_i
                energy = ((amplitudes_i.conj() @ hamiltonian @ amplitudes_j) / (amplitudes_i.conj() @ amplitudes_i)).real
            energy.backward()
            optimizer.step()
            logging.info("local optimizing, step %d, energy: %f", i, energy.item())
        logging.info("local optimization finished")
        logging.info("saving checkpoint")
        torch.save(network.state_dict(), f"{checkpoint_path}/{run_name}.pt")
        logging.info("checkpoint saved")


if __name__ == "__main__":
    main()

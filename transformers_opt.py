import os
import sys
import logging
import argparse
import numpy
import scipy
import torch
import openfermion
import transformers_network
import openfermion_to_sparse

torch.set_grad_enabled(False)


def main(*, model, embedding_dim, heads_num, feed_forward_dim, depth, sampling_count, lr, local_step, log_path, checkpoint_path, model_path):
    logging.basicConfig(
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{log_path}/{model}.log")],
        level=logging.INFO,
        format='%(asctime)s - ' + model + ' - %(levelname)s - %(message)s',
    )
    logging.info(
        "opt script start, embedding_dim: %d, heads_num: %d, feed_forward_dim: %d, depth: %d, sampling_count: %d, lr: %f, local_step: %d",
        embedding_dim,
        heads_num,
        feed_forward_dim,
        depth,
        sampling_count,
        lr,
        local_step,
    )

    logging.info("loading physical model %s", model)
    physical_model = openfermion.MolecularData(filename=f"{model_path}/{model}.hdf5")
    logging.info("converting physical model to python list")
    openfermion_hamiltonian = list(openfermion.transforms.get_fermion_operator(physical_model.get_molecular_hamiltonian()).terms.items())
    logging.info("creating neural network")
    network = transformers_network.WaveFunction(
        double_sites=physical_model.n_qubits,
        physical_dim=2,
        is_complex=True,
        spin_up=physical_model.n_electrons // 2,
        spin_down=physical_model.n_electrons // 2,
        embedding_dim=embedding_dim,
        heads_num=heads_num,
        feed_forward_dim=feed_forward_dim,
        depth=depth,
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
        configs, _, _, _ = network.generate_unique(sampling_count)
        logging.info("converting sampling configs to tuple")
        tuple_configs = [tuple(config.view([-1]).tolist()) for config in configs]
        logging.info("counting unique sampling count")
        unique_sampling_count = len(tuple_configs)
        logging.info("unique sampling count is %d", unique_sampling_count)

        logging.info("generating hamiltonian as sparse matrix data")
        indices_i, indices_j, values = openfermion_to_sparse.openfermion_to_sparse(openfermion_hamiltonian, tuple_configs)
        logging.info("converting sparse matrix data to coo matrix")
        hamiltonian = torch.sparse_coo_tensor((indices_i, indices_j), values, [unique_sampling_count, unique_sampling_count], dtype=torch.complex128).cuda()

        logging.info("local optimization starting")
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        for i in range(local_step):
            logging.info("calculating energy")
            with torch.enable_grad():
                amplitudes = network(configs)
                energy = ((amplitudes.conj() @ hamiltonian @ amplitudes) / (amplitudes.conj() @ amplitudes)).real
            logging.info("energy: %f", energy.item())

            logging.info("opimization stepping")
            energy.backward()
            optimizer.step()
            optimizer.zero_grad()
        logging.info("local optimization finished")
        logging.info("saving checkpoint")
        torch.save(network.state_dict(), f"{checkpoint_path}/{model}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="approach to the ground state for the quantum chemistry many body system")
    parser.add_argument("model")
    parser.add_argument("-n", "--sampling-count", dest="sampling_count", type=int, default=4000, help="sampling count")
    parser.add_argument("-e", "--embedding-dim", dest="embedding_dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("-m", "--heads-num", dest="heads_num", type=int, default=8, help="heads number")
    parser.add_argument("-f", "--feed-forward-dim", dest="feed_forward_dim", type=int, default=2048, help="feedforward dimension")
    parser.add_argument("-d", "--depth", dest="depth", type=int, default=6, help="network depth")
    parser.add_argument("-r", "--learning-rate", dest="lr", type=float, default=1e-3, help="learning rate for the local optimizer")
    parser.add_argument("-s", "--local-step", dest="local_step", type=int, default=1000, help="step count for the local optimizer")
    parser.add_argument("-L", "--log-path", dest="log_path", type=str, default="logs", help="path of logs folder")
    parser.add_argument("-C", "--checkpoint-path", dest="checkpoint_path", type=str, default="checkpoints", help="path of checkpoints folder")
    parser.add_argument("-M", "--model-path", dest="model_path", type=str, default="models", help="path of models folder")
    args = parser.parse_args()
    main(
        model=args.model,
        embedding_dim=args.embedding_dim,
        heads_num=args.heads_num,
        feed_forward_dim=args.feed_forward_dim,
        depth=args.depth,
        sampling_count=args.sampling_count,
        lr=args.lr,
        local_step=args.local_step,
        log_path=args.log_path,
        checkpoint_path=args.checkpoint_path,
        model_path=args.model_path,
    )

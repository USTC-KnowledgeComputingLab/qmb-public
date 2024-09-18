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
import cg


def main():
    parser = argparse.ArgumentParser(description="approach to the ground state for the quantum chemistry many body system", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model", help="model name")
    parser.add_argument("network", help="network name")
    parser.add_argument("-n", "--sampling-count", dest="sampling_count", type=int, default=4000, help="sampling count")
    parser.add_argument("-r", "--learning-rate", dest="lr", type=float, default=1e-3, help="learning rate for the local optimizer")
    parser.add_argument("-s", "--local-step", dest="local_step", type=int, default=1000, help="step count for the local optimizer")
    parser.add_argument("-l", "--local-loss", dest="local_loss", type=float, default=1e-6, help="early break loss threshold for local optimization")
    parser.add_argument("-p", "--logging-psi-count", dest="logging_psi_count", type=int, default=30, help="psi count to be printed after local optimizer")
    parser.add_argument("-k", "--metric-rank", dest="metric_rank", type=int, default=128, help="the rank of metric")
    parser.add_argument("-c", "--cg-max-step", dest="cg_max_step", type=int, default=None, help="max step for cg")
    parser.add_argument("-g", "--cg-threshold", dest="cg_threshold", type=float, default=None, help="threshold for cg")
    parser.add_argument("-y", "--cg-epsilon", dest="cg_epsilon", type=float, default=1e-2, help="epsilon for cg")
    parser.add_argument("-L", "--log-path", dest="log_path", type=str, default="logs", help="path of logs folder")
    parser.add_argument("-C", "--checkpoint-path", dest="checkpoint_path", type=str, default="checkpoints", help="path of checkpoints folder")
    parser.add_argument("-M", "--model-path", dest="model_path", type=str, default="models", help="path of models folder")
    parser.add_argument("-N", "--run-name", dest="run_name", type=str, default=None, help="the run name")
    parser.add_argument("-S", "--random-seed", dest="random_seed", type=int, default=None, help="the manual random seed")
    args, other_args = parser.parse_known_args()
    model_name = args.model
    network_name = args.network
    sampling_count = args.sampling_count
    lr = args.lr
    local_step = args.local_step
    local_loss = args.local_loss
    logging_psi_count = args.logging_psi_count
    metric_rank = args.metric_rank
    cg_max_step = args.cg_max_step
    cg_threshold = args.cg_threshold
    cg_epsilon = args.cg_epsilon
    if cg_max_step is None and cg_threshold is None:
        cg_max_step = metric_rank
    log_path = args.log_path
    checkpoint_path = args.checkpoint_path
    model_path = args.model_path
    run_name = args.run_name
    if run_name is None:
        run_name = model_name
    random_seed = args.random_seed

    logging.basicConfig(
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{log_path}/{run_name}.log")],
        level=logging.INFO,
        format=f"[%(process)d] %(asctime)s {model_name}({network_name}) %(levelname)s: %(message)s",
    )

    logging.info("sr script start, with %a", sys.argv)
    logging.info("model: %s, network: %s, run name: %s", model_name, network_name, run_name)
    logging.info("sampling count: %d, learning rate: %f, local step: %d, local loss: %f, logging psi count: %d", sampling_count, lr, local_step, local_loss, logging_psi_count)
    logging.info("cg max step: %a, cg threshold: %a", cg_max_step, cg_threshold)
    logging.info("log path: %s, checkpoint path: %s, model path: %s", log_path, checkpoint_path, model_path)
    logging.info("other arguments will be passed to network parser: %a", other_args)

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
        configs, pre_amplitudes, _, _ = network.generate_unique(sampling_count)
        logging.info("sampling done")
        unique_sampling_count = len(configs)
        logging.info("unique sampling count is %d", unique_sampling_count)

        logging.info("generating hamiltonian data to create sparse matrix")
        indices_i_and_j, values = openfermion_hamiltonian.inside(configs.cpu())
        logging.info("sparse matrix data created")
        logging.info("converting sparse matrix data to coo matrix")
        hamiltonian = scipy.sparse.coo_matrix((values, indices_i_and_j.T), [unique_sampling_count, unique_sampling_count], dtype=numpy.complex128)
        logging.info("coo matrix created")
        logging.info("estimating ground state")
        expected_energy, targets = scipy.sparse.linalg.lobpcg(hamiltonian, pre_amplitudes.cpu().reshape([-1, 1]).detach().numpy(), largest=False, maxiter=1024)
        logging.info("estimiated, target energy is %f, fci energy is %f", expected_energy.item(), fci_energy)
        logging.info("preparing learning targets")
        targets = torch.tensor(targets).view([-1]).cuda()
        max_index = targets.abs().argmax()
        targets = targets / targets[max_index]
        log_targets = targets.log()

        logging.info("local optimization starting")
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        for i in range(local_step):
            with torch.enable_grad():
                amplitudes = network(configs)
                amplitudes = amplitudes / amplitudes[max_index]
                log_amplitudes = amplitudes.log()
                error = (log_amplitudes - log_targets) / (2 * torch.pi)
                rounded_error = error.real**2 + (error.imag - error.imag.round())**2
                loss = rounded_error.mean()

                logging.info("collecting jacobian")
                indices = rounded_error.sort(descending=True).indices[:metric_rank]
                plains = torch.view_as_real(log_amplitudes[indices]).view([-1])
                jacobian = []
                for j, plain in enumerate(plains):
                    logging.info("collecting jacobian %d", j)
                    jacobian.append(torch.autograd.grad(plain, network.parameters(), retain_graph=True))
                logging.info("jacobian has been collected")
                gradient = torch.autograd.grad(loss, network.parameters())
                updates = cg.cg(jacobian, gradient, max_step=cg_max_step, threshold=cg_threshold, epsilon=cg_epsilon)
                for parameter, update in zip(network.parameters(), updates):
                    parameter.grad = update

            optimizer.step()
            optimizer.zero_grad()
            logging.info("local optimizing, step %d, loss %f", i, loss.item())
            if loss < local_loss:
                logging.info("local optimization stop since local loss reached")
                break
        logging.info("local optimization finished")
        logging.info("saving checkpoint")
        torch.save(network.state_dict(), f"{checkpoint_path}/{run_name}.pt")
        logging.info("checkpoint saved")
        logging.info("calculating current energy")
        amplitudes = amplitudes.cpu().detach().numpy()
        final_energy = ((amplitudes.conj() @ hamiltonian @ amplitudes) / (amplitudes.conj() @ amplitudes)).real
        logging.info(
            "loss = %f during local optimization, final energy %f, target energy %f, fci energy %f",
            loss.item(),
            final_energy.item(),
            expected_energy.item(),
            fci_energy,
        )
        logging.info("printing several largest amplitudes")
        indices = targets.abs().sort(descending=True).indices
        for index in indices[:logging_psi_count]:
            logging.info("config %s, target %s, final %s", "".join(map(str, configs[index].cpu().numpy())), f"{targets[index].item():.4f}", f"{amplitudes[index].item():.4f}")


if __name__ == "__main__":
    main()

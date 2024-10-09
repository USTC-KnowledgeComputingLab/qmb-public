import logging
import typing
import dataclasses
import numpy
import scipy
import torch
import tyro
from . import losses
from .common import CommonConfig
from .subcommand_dict import subcommand_dict


@dataclasses.dataclass
class LearnConfig:
    common: typing.Annotated[CommonConfig, tyro.conf.OmitArgPrefixes]

    # sampling count
    sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-n"])] = 4000
    # learning rate for the local optimizer
    learning_rate: typing.Annotated[float, tyro.conf.arg(aliases=["-r"], help_behavior_hint="(default: 1e-3 for Adam, 1 for LBFGS)")] = -1
    # step count for the local optimizer
    local_step: typing.Annotated[int, tyro.conf.arg(aliases=["-s"], help_behavior_hint="(default: 1000 for Adam, 400 for LBFGS)")] = -1
    # early break loss threshold for local optimization
    local_loss: typing.Annotated[float, tyro.conf.arg(aliases=["-t"])] = 1e-8
    # psi count to be printed after local optimizer
    logging_psi: typing.Annotated[int, tyro.conf.arg(aliases=["-p"])] = 30
    # the loss function to be used
    loss_name: typing.Annotated[str, tyro.conf.arg(aliases=["-l"])] = "log"
    # Use LBFGS instead of Adam
    use_lbfgs: typing.Annotated[bool, tyro.conf.arg(aliases=["-2"])] = False

    def __post_init__(self):
        if self.learning_rate == -1:
            self.learning_rate = 1 if self.use_lbfgs else 1e-3
        if self.local_step == -1:
            self.local_step = 400 if self.use_lbfgs else 1000

    def main(self):
        model, network = self.common.main()

        logging.info(
            "sampling count: %d, learning rate: %f, local step: %d, local loss: %f, logging psi: %d, loss name: %s, use_lbfgs: %a",
            self.sampling_count,
            self.learning_rate,
            self.local_step,
            self.local_loss,
            self.logging_psi,
            self.loss_name,
            self.use_lbfgs,
        )

        logging.info("main looping")
        while True:
            logging.info("sampling configurations")
            configs, pre_amplitudes, _, _ = network.generate_unique(self.sampling_count)
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

            logging.info("choosing loss function as %s", self.loss_name)
            loss_func = getattr(losses, self.loss_name)

            if self.use_lbfgs:
                optimizer = torch.optim.LBFGS(network.parameters(), lr=self.learning_rate)
            else:
                optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)

            def closure():
                optimizer.zero_grad()
                amplitudes = network(configs)
                amplitudes = amplitudes / amplitudes[max_index]
                loss = loss_func(amplitudes, targets)
                loss.backward()
                loss.amplitudes = amplitudes
                return loss

            logging.info("local optimization starting")
            for i in range(self.local_step):
                loss = optimizer.step(closure)
                logging.info("local optimizing, step %d, loss %.10f", i, loss.item())
                if loss < self.local_loss:
                    logging.info("local optimization stop since local loss reached")
                    break

            logging.info("local optimization finished")
            logging.info("saving checkpoint")
            torch.save(network.state_dict(), f"{self.common.checkpoint_path}/{self.common.job_name}.pt")
            logging.info("checkpoint saved")
            logging.info("calculating current energy")
            torch.enable_grad(closure)()
            amplitudes = loss.amplitudes.cpu().detach().numpy()
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
            for index in indices[:self.logging_psi]:
                logging.info("config %s, target %s, final %s", "".join(map(str, configs[index].cpu().numpy())), f"{targets[index].item():.8f}", f"{amplitudes[index].item():.8f}")


subcommand_dict["learn"] = LearnConfig

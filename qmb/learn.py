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
    sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1024
    # selected sampling count
    selected_sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-b"])] = 65536
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
    # use LBFGS instead of Adam
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
            logging.info("core sampling configurations")
            configs_core, psi_core, _, _ = network.generate_unique(self.sampling_count)
            configs_core = configs_core.cpu()
            psi_core = psi_core.cpu()
            sampling_count_core = len(configs_core)
            logging.info("sampling count core is %d", sampling_count_core)

            logging.info("calculating extended configurations")
            indices_i_and_j, values, configs_extended = model.outside(configs_core)
            logging.info("extended configurations created")
            sampling_count_extended = len(configs_extended)
            logging.info("extended configurations count is %d", sampling_count_extended)

            logging.info("converting sparse extending matrix data to sparse matrix")
            hamiltonian = torch.sparse_coo_tensor(indices_i_and_j.T, values, [sampling_count_core, sampling_count_extended], dtype=torch.complex128).to_sparse_csr()
            logging.info("sparse extending matrix created")

            logging.info("estimating the importance of extended configurations")
            importance = (psi_core.conj() * psi_core).abs() @ (hamiltonian.conj() * hamiltonian).abs()
            importance[:sampling_count_core] += importance.max()
            logging.info("importance of extended configurations created")

            logging.info("selecting extended configurations by importance")
            selected_indices = importance.sort(descending=True).indices[:self.selected_sampling_count].sort().values
            logging.info("extended configurations selected indices prepared")

            logging.info("selecting extended configurations")
            configs_extended = configs_extended[selected_indices]
            logging.info("extended configurations selected")
            sampling_count_extended = len(configs_extended)
            logging.info("selected extended configurations count is %d", sampling_count_extended)

            logging.info("calculating sparse data of hamiltonian on extended configurations")
            indices_i_and_j, values = model.inside(configs_extended)
            logging.info("converting sparse matrix data to sparse matrix")
            hamiltonian = scipy.sparse.coo_matrix((values, indices_i_and_j.T), [sampling_count_extended, sampling_count_extended], dtype=numpy.complex128).tocsr()
            logging.info("sparse matrix on extended configurations created")

            logging.info("preparing initial psi used in lobpcg")
            psi_extended = numpy.pad(psi_core, (0, sampling_count_extended - sampling_count_core)).reshape([-1, 1])
            logging.info("initial psi used in lobpcg has been created")

            logging.info("calculating minimum energy on extended configurations")
            target_energy, psi_extended = scipy.sparse.linalg.lobpcg(hamiltonian, psi_extended, largest=False, maxiter=1024)
            logging.info("energy on extended configurations is %.10f, ref energy is %.10f, error is %.10f", target_energy.item(), model.ref_energy, target_energy.item() - model.ref_energy)

            logging.info("preparing learning targets")
            configs = torch.tensor(configs_extended).cuda()
            targets = torch.tensor(psi_extended).view([-1]).cuda()
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

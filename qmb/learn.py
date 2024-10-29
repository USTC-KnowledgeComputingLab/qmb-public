import logging
import typing
import dataclasses
import torch
import tyro
from . import losses
from .common import CommonConfig
from .subcommand_dict import subcommand_dict
from .utility import extend_and_select, lobpcg_and_select


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
    # use LBFGS instead of Adam
    use_lbfgs: typing.Annotated[bool, tyro.conf.arg(aliases=["-2"])] = False
    # the post sampling iteration
    post_sampling_iteration: typing.Annotated[int, tyro.conf.arg(aliases=["-i"])] = 0
    # the post sampling count
    post_sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-c"])] = 50000

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
            configs, psi, _, _ = network.generate_unique(self.sampling_count)
            logging.info("sampling done")

            for _ in range(self.post_sampling_iteration):
                logging.info("extend and select start")
                configs, psi = extend_and_select(model, configs, psi, self.post_sampling_count)
                logging.info("extend and select finished")

                logging.info("lobpcg and select start")
                _, _, configs, psi = lobpcg_and_select(model, configs, psi, self.sampling_count)
                logging.info("lobpcg and select finished")


            logging.info("lobpcg start")
            target_energy, hamiltonian, _, targets = lobpcg_and_select(model, configs, psi)
            logging.info("lobpcg finished")

            logging.info("preparing learning targets")
            targets = targets.view([-1])
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
            loss = torch.enable_grad(closure)()
            amplitudes = loss.amplitudes
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

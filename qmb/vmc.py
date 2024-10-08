import logging
import typing
import dataclasses
import torch
import tyro
from .common import CommonConfig
from .subcommand_dict import subcommand_dict


@dataclasses.dataclass
class VmcConfig:
    common: typing.Annotated[CommonConfig, tyro.conf.OmitArgPrefixes]

    # sampling count
    sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-n"])] = 4000
    # learning rate for the local optimizer
    learning_rate: typing.Annotated[float, tyro.conf.arg(aliases=["-r"], help_behavior_hint="(default: 1e-3 for Adam, 1 for LBFGS)")] = -1
    # step count for the local optimizer
    local_step: typing.Annotated[int, tyro.conf.arg(aliases=["-s"])] = 1000
    # calculate all psi(s)')
    include_outside: typing.Annotated[bool, tyro.conf.arg(aliases=["-o"])] = False
    # Use deviation instead of energy
    deviation: typing.Annotated[bool, tyro.conf.arg(aliases=["-d"])] = False
    # Fix outside phase when optimizing outside deviation
    fix_outside: typing.Annotated[bool, tyro.conf.arg(aliases=["-f"])] = False
    # Use LBFGS instead of Adam
    use_lbfgs: typing.Annotated[bool, tyro.conf.arg(aliases=["-2"])] = False
    # Do not calculate deviation or energy when optimizing energy or deviation
    omit_another: typing.Annotated[bool, tyro.conf.arg(aliases=["-i"])] = False

    def __post_init__(self):
        if self.learning_rate == -1:
            self.learning_rate = 1 if self.use_lbfgs else 1e-3

    def main(self):
        model, network = self.common.main()

        logging.info(
            "sampling count: %d, learning rate: %f, local step: %d, include outside: %a, use deviation: %a, fix outside: %a, use lbfgs: %a, omit another: %a",
            self.sampling_count,
            self.learning_rate,
            self.local_step,
            self.include_outside,
            self.deviation,
            self.fix_outside,
            self.use_lbfgs,
            self.omit_another,
        )

        logging.info("main looping")
        while True:
            logging.info("sampling configurations")
            configs_i, _, _, _ = network.generate_unique(self.sampling_count)
            logging.info("sampling done")
            unique_sampling_count = len(configs_i)
            logging.info("unique sampling count is %d", unique_sampling_count)

            if self.include_outside:
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

            if self.use_lbfgs:
                optimizer = torch.optim.LBFGS(network.parameters(), lr=self.learning_rate)
            else:
                optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)

            if self.deviation:

                def closure():
                    optimizer.zero_grad()
                    amplitudes_i = network(configs_i)
                    if self.include_outside:
                        if self.fix_outside:
                            with torch.no_grad():
                                amplitudes_j = network(configs_j)
                            amplitudes_j = torch.cat([amplitudes_i[:unique_sampling_count], amplitudes_j[unique_sampling_count:]])
                        else:
                            amplitudes_j = network(configs_j)
                    else:
                        amplitudes_j = amplitudes_i
                    hamiltonian_amplitudes_j = hamiltonian @ amplitudes_j
                    deviation = (hamiltonian_amplitudes_j / amplitudes_i).std()
                    deviation.backward()
                    if self.omit_another:
                        deviation.energy = torch.tensor(torch.nan)
                    else:
                        with torch.no_grad():
                            deviation.energy = ((amplitudes_i.conj() @ hamiltonian_amplitudes_j) / (amplitudes_i.conj() @ amplitudes_i)).real
                    return deviation

                logging.info("local optimization for deviation starting")
                for i in range(self.local_step):
                    deviation = optimizer.step(closure)
                    logging.info("local optimizing, step: %d, energy: %.10f, deviation: %.10f", i, deviation.energy.item(), deviation.item())
            else:

                def closure():
                    optimizer.zero_grad()
                    amplitudes_i = network(configs_i)
                    if self.include_outside:
                        with torch.no_grad():
                            amplitudes_j = network(configs_j)
                    else:
                        amplitudes_j = amplitudes_i
                    hamiltonian_amplitudes_j = hamiltonian @ amplitudes_j.detach()
                    energy = ((amplitudes_i.conj() @ hamiltonian_amplitudes_j) / (amplitudes_i.conj() @ amplitudes_i.detach())).real
                    energy.backward()
                    if self.omit_another:
                        energy.deviation = torch.tensor(torch.nan)
                    else:
                        with torch.no_grad():
                            energy.deviation = (hamiltonian_amplitudes_j / amplitudes_i).std()
                    return energy

                logging.info("local optimization for energy starting")
                for i in range(self.local_step):
                    energy = optimizer.step(closure)
                    logging.info("local optimizing, step: %d, energy: %.10f, deviation: %.10f", i, energy.item(), energy.deviation.item())

            logging.info("local optimization finished")
            logging.info("saving checkpoint")
            torch.save(network.state_dict(), f"{self.common.checkpoint_path}/{self.common.job_name}.pt")
            logging.info("checkpoint saved")


subcommand_dict["vmc"] = VmcConfig

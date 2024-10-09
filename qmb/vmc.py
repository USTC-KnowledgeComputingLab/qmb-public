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
    # Do not calculate deviation when optimizing energy
    omit_deviation: typing.Annotated[bool, tyro.conf.arg(aliases=["-i"])] = False

    def __post_init__(self):
        if self.learning_rate == -1:
            self.learning_rate = 1 if self.use_lbfgs else 1e-3

    def main(self):
        model, network = self.common.main()

        logging.info(
            "sampling count: %d, learning rate: %f, local step: %d, include outside: %a, use deviation: %a, fix outside: %a, use lbfgs: %a, omit deviation: %a",
            self.sampling_count,
            self.learning_rate,
            self.local_step,
            self.include_outside,
            self.deviation,
            self.fix_outside,
            self.use_lbfgs,
            self.omit_deviation,
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
                    # Optimizing deviation
                    optimizer.zero_grad()
                    # Calculate amplitudes i and amplitudes j
                    # When including outside, amplitudes j should be calculated individually, otherwise, it equals to amplitudes i
                    # It should be notices that sometimes we do not want to optimize small configurations
                    # So we calculate amplitudes j in no grad mode
                    # but the first several configurations in amplitudes j are duplicated with those in amplitudes i
                    # So cat them manually
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
                    # <s|H|psi> will be used multiple times, calculate it first
                    # as we want to optimize deviation, every value should be calculated in grad mode, so we do not detach anything
                    hamiltonian_amplitudes_j = hamiltonian @ amplitudes_j
                    # energy is just <psi|s> <s|H|psi> / <psi|s> <s|psi>
                    energy = (amplitudes_i.conj() @ hamiltonian_amplitudes_j) / (amplitudes_i.conj() @ amplitudes_i)
                    # we want to estimate variance of E_s - E with weight <psi|s><s|psi>
                    # where E_s = <s|H|psi>/<s|psi>
                    # the variance is (E_s - E).conj() @ (E_s - E) * <psi|s> <s|psi> / ... = (E_s <s|psi> - E <s|psi>).conj() @ (E_s <s|psi> - E <s|psi>) / ...
                    # so we calculate E_s <s|psi> - E <s|psi> first, which is just <s|H|psi> - <s|psi> E, we name it as `difference'
                    difference = hamiltonian_amplitudes_j - amplitudes_i * energy
                    # the numerator calculated, the following is the variance
                    variance = (difference.conj() @ difference) / (amplitudes_i.conj() @ amplitudes_i)
                    # calculate the deviation
                    deviation = variance.real.sqrt()
                    deviation.backward()
                    # As we have already calculated energy, embed it in deviation for logging
                    deviation.energy = energy.real
                    return deviation

                logging.info("local optimization for deviation starting")
                for i in range(self.local_step):
                    deviation = optimizer.step(closure)
                    logging.info("local optimizing, step: %d, energy: %.10f, deviation: %.10f", i, deviation.energy.item(), deviation.item())
            else:

                def closure():
                    # Optimizing energy
                    optimizer.zero_grad()
                    # Calculate amplitudes i and amplitudes j
                    # When including outside, amplitudes j should be calculated individually, otherwise, it equals to amplitudes i
                    # Because of gradient formula, we always calculate amplitudes j in no grad mode
                    amplitudes_i = network(configs_i)
                    if self.include_outside:
                        with torch.no_grad():
                            amplitudes_j = network(configs_j)
                    else:
                        amplitudes_j = amplitudes_i.detach()
                    # <s|H|psi> will be used multiple times, calculate it first
                    # it should be notices that this <s|H|psi> is totally detached, since both hamiltonian and amplitudes j is detached
                    hamiltonian_amplitudes_j = hamiltonian @ amplitudes_j
                    # energy is just <psi|s> <s|H|psi> / <psi|s> <s|psi>
                    # we only calculate gradient on <psi|s>, both <s|H|psi> and <s|psi> should be detached
                    # since <s|H|psi> has been detached already, we detach <s|psi> here manually
                    energy = (amplitudes_i.conj() @ hamiltonian_amplitudes_j) / (amplitudes_i.conj() @ amplitudes_i.detach())
                    # Calculate deviation
                    # The variance is (E_s <s|psi> - E <s|psi>).conj() @ (E_s <s|psi> - E <s|psi>) / <psi|s> <s|psi>
                    # Calculate E_s <s|psi> - E <s|psi> first and name it as difference
                    if self.omit_deviation:
                        deviation = torch.tensor(torch.nan)
                    else:
                        with torch.no_grad():
                            difference = hamiltonian_amplitudes_j - amplitudes_i * energy
                            variance = (difference.conj() @ difference) / (amplitudes_i.conj() @ amplitudes_i)
                            deviation = variance.real.sqrt()
                    energy = energy.real
                    energy.backward()
                    # Embed the deviation which has been calculated in energy for logging
                    energy.deviation = deviation
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

"""
This file implements a variational Monte Carlo method for solving quantum many-body problems.
"""

import logging
import typing
import dataclasses
import torch
import tyro
from .common import CommonConfig
from .subcommand_dict import subcommand_dict
from .optimizer import initialize_optimizer


@dataclasses.dataclass
class VmcConfig:
    """
    The VMC optimization for solving quantum many-body problems.
    """

    # pylint: disable=too-many-instance-attributes

    common: typing.Annotated[CommonConfig, tyro.conf.OmitArgPrefixes]

    # The sampling count
    sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-n"])] = 4000
    # Whether to exclude external configurations
    exclude_outside: typing.Annotated[bool, tyro.conf.arg(aliases=["-e"])] = False
    # Whether to use the global optimizer
    global_opt: typing.Annotated[bool, tyro.conf.arg(aliases=["-g"])] = False
    # Whether to use LBFGS instead of Adam
    use_lbfgs: typing.Annotated[bool, tyro.conf.arg(aliases=["-2"])] = False
    # The learning rate for the local optimizer
    learning_rate: typing.Annotated[float, tyro.conf.arg(aliases=["-r"], help_behavior_hint="(default: 1e-3 for Adam, 1 for LBFGS)")] = -1
    # The number of steps for the local optimizer
    local_step: typing.Annotated[int, tyro.conf.arg(aliases=["-s"])] = 1000
    # Whether to use deviation instead of energy for optimization
    deviation: typing.Annotated[bool, tyro.conf.arg(aliases=["-d"])] = False
    # Whether to fix external configurations during deviation optimization
    fix_outside: typing.Annotated[bool, tyro.conf.arg(aliases=["-f"])] = False
    # Whether to omit deviation calculation during energy optimization
    omit_deviation: typing.Annotated[bool, tyro.conf.arg(aliases=["-i"])] = False

    def __post_init__(self) -> None:
        if self.learning_rate == -1:
            self.learning_rate = 1 if self.use_lbfgs else 1e-3

    def main(self) -> None:
        """
        The main function for the VMC optimization.
        """
        # pylint: disable=too-many-statements

        model, network = self.common.main()

        logging.info(
            "Arguments Summary: "
            "Sampling Count: %d, "
            "Exclude Outside: %s, "
            "Global Optimizer: %s, "
            "Use LBFGS: %s, "
            "Learning Rate: %.10f, "
            "Local Steps: %d, "
            "Use Deviation: %s, "
            "Fix Outside: %s, "
            "Omit Deviation: %s",
            self.sampling_count,
            "Yes" if self.exclude_outside else "No",
            "Yes" if self.global_opt else "No",
            "Yes" if self.use_lbfgs else "No",
            self.learning_rate,
            self.local_step,
            "Yes" if self.deviation else "No",
            "Yes" if self.fix_outside else "No",
            "Yes" if self.omit_deviation else "No",
        )

        optimizer = initialize_optimizer(
            network.parameters(),
            use_lbfgs=self.use_lbfgs,
            learning_rate=self.learning_rate,
            optimizer_path=self.common.checkpoint_path / f"{self.common.job_name}.opt.pt",
        )

        while True:
            logging.info("Starting a new optimization cycle")

            logging.info("Sampling configurations")
            configs_i, _, _, _ = network.generate_unique(self.sampling_count)
            logging.info("Sampling completed")
            unique_sampling_count = len(configs_i)
            logging.info("Unique sampling count: %d", unique_sampling_count)

            if self.exclude_outside:
                logging.info("Generating hamiltonian data generation for internal sparse matrix")
                indices_i_and_j, values = model.inside(configs_i)
                logging.info("Internal sparse matrix data successfully generated")
                logging.info("Converting generated sparse matrix data into a sparse tensor")
                hamiltonian = torch.sparse_coo_tensor(indices_i_and_j.T, values, [unique_sampling_count, unique_sampling_count], dtype=torch.complex128).to_sparse_csr()
                logging.info("Sparse tensor successfully created")
            else:
                logging.info("Generating hamiltonian data generation for external sparse matrix")
                indices_i_and_j, values, configs_j = model.outside(configs_i)
                logging.info("External sparse matrix data successfully generated")
                outside_count = len(configs_j)
                logging.info("External configurations count: %d", outside_count)
                logging.info("Converting generated sparse matrix data into a sparse tensor")
                hamiltonian = torch.sparse_coo_tensor(indices_i_and_j.T, values, [unique_sampling_count, outside_count], dtype=torch.complex128).to_sparse_csr()
                logging.info("Sparse tensor successfully created")

            optimizer = initialize_optimizer(
                network.parameters(),
                use_lbfgs=self.use_lbfgs,
                learning_rate=self.learning_rate,
                new_opt=not self.global_opt,
                optimizer=optimizer,
            )

            logging.info("Starting local optimization process")
            if self.deviation:

                def closure() -> torch.Tensor:
                    # Optimizing deviation
                    optimizer.zero_grad()
                    # Calculate amplitudes i and amplitudes j
                    # When including outside, amplitudes j should be calculated individually, otherwise, it equals to amplitudes i
                    # It should be notices that sometimes we do not want to optimize small configurations
                    # So we calculate amplitudes j in no grad mode
                    # but the first several configurations in amplitudes j are duplicated with those in amplitudes i
                    # So cat them manually
                    amplitudes_i = network(configs_i)
                    if self.exclude_outside:
                        amplitudes_j = amplitudes_i
                    else:
                        if self.fix_outside:
                            with torch.no_grad():
                                amplitudes_j = network(configs_j)
                            amplitudes_j = torch.cat([amplitudes_i[:unique_sampling_count], amplitudes_j[unique_sampling_count:]])
                        else:
                            amplitudes_j = network(configs_j)
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
                    deviation.backward()  # type: ignore[no-untyped-call]
                    # As we have already calculated energy, embed it in deviation for logging
                    deviation.energy = energy.real  # type: ignore[attr-defined]
                    return deviation

                for i in range(self.local_step):
                    deviation: torch.Tensor = optimizer.step(closure)  # type: ignore[assignment,arg-type]
                    logging.info("Local optimization in progress, step: %d, energy: %.10f, deviation: %.10f", i, deviation.energy.item(), deviation.item())  # type: ignore[attr-defined]
            else:

                def closure() -> torch.Tensor:
                    # Optimizing energy
                    optimizer.zero_grad()
                    # Calculate amplitudes i and amplitudes j
                    # When including outside, amplitudes j should be calculated individually, otherwise, it equals to amplitudes i
                    # Because of gradient formula, we always calculate amplitudes j in no grad mode
                    amplitudes_i = network(configs_i)
                    if self.exclude_outside:
                        amplitudes_j = amplitudes_i.detach()
                    else:
                        with torch.no_grad():
                            amplitudes_j = network(configs_j)
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
                    energy.backward()  # type: ignore[no-untyped-call]
                    # Embed the deviation which has been calculated in energy for logging
                    energy.deviation = deviation  # type: ignore[attr-defined]
                    return energy

                for i in range(self.local_step):
                    energy: torch.Tensor = optimizer.step(closure)  # type: ignore[assignment,arg-type]
                    logging.info("Local optimization in progress, step: %d, energy: %.10f, deviation: %.10f", i, energy.item(), energy.deviation.item())  # type: ignore[attr-defined]

            logging.info("Local optimization process completed")

            logging.info("Saving model checkpoint")
            torch.save(network.state_dict(), self.common.checkpoint_path / f"{self.common.job_name}.pt")
            torch.save(optimizer.state_dict(), self.common.checkpoint_path / f"{self.common.job_name}.opt.pt")
            logging.info("Checkpoint successfully saved")

            logging.info("Current optimization cycle completed")


subcommand_dict["vmc"] = VmcConfig

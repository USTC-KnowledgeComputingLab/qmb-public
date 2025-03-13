"""
This file implements the subspace chopping for the result of the imag script.
"""

import logging
import typing
import dataclasses
import tyro
import torch.utils.tensorboard
from .common import CommonConfig
from .subcommand_dict import subcommand_dict


@dataclasses.dataclass
class ChopImagConfig:
    """
    The subspace chopping for the result of the imag script.
    """

    common: typing.Annotated[CommonConfig, tyro.conf.OmitArgPrefixes]

    # The number of configurations to eliminate every iteration
    chop_size: typing.Annotated[int, tyro.conf.arg(aliases=["-c"])] = 10000
    # The estimated magnitude of the second order term
    second_order_magnitude: typing.Annotated[float, tyro.conf.arg(aliases=["-s"])] = 0.0

    def main(self) -> None:
        """
        The main function for the subspace chopping.
        """

        model, _, data = self.common.main()

        logging.info(
            "Arguments Summary: "
            "Chop Size: %d"
            "Second Order Magnitude: %.10f",
            self.chop_size,
            self.second_order_magnitude,
        )

        configs, psi = data["imag"]["pool"]
        configs = configs.to(device=self.common.device)
        psi = psi.to(device=self.common.device)

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=self.common.folder())  # type: ignore[no-untyped-call]

        i = 0
        while True:
            logging.info("The number of configurations: %d", len(configs))
            writer.add_scalar("chop_imag/num_configs", len(configs), i)  # type: ignore[no-untyped-call]
            psi = psi / psi.norm()
            hamiltonian_psi = model.apply_within(configs, psi, configs)
            psi_hamiltonian_psi = (psi.conj() @ hamiltonian_psi).real
            energy = psi_hamiltonian_psi
            logging.info("The energy: %.10s, The energy error is %.10f", energy.item(), energy.item() - model.ref_energy)
            writer.add_scalar("chop_imag/energy", energy.item(), i)  # type: ignore[no-untyped-call]
            writer.add_scalar("chop_imag/error", energy.item() - model.ref_energy, i)  # type: ignore[no-untyped-call]
            grad = hamiltonian_psi - psi_hamiltonian_psi * psi
            delta = -psi.conj() * grad
            real_delta = 2 * delta.real
            second_order = (psi.conj() * psi).real * self.second_order_magnitude
            selected = (real_delta + second_order).argsort()[self.chop_size:]
            if len(selected) == 0:
                break
            configs = configs[selected]
            psi = psi[selected]
            writer.flush()  # type: ignore[no-untyped-call]
            i += 1


subcommand_dict["chop_imag"] = ChopImagConfig

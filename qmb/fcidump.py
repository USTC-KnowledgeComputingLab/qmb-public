"""
This file provides an interface to work with FCIDUMP files and reuses utilities from openfermion.
"""

import re
import logging
import pathlib
import torch
import openfermion
from .hamiltonian import Hamiltonian
from .openfermion import Model as OpenFermionModel
from .model_dict import model_dict


def _read_fcidump(file_name: str) -> openfermion.FermionOperator:
    # pylint: disable=too-many-locals
    with open(file_name, "rt", encoding="utf-8") as file:
        n_orbit: int = -1
        for line in file:
            data: str = line.lower()
            match = re.search(r"norb\s*=\s*(\d+)", data)
            if match is not None:
                n_orbit = int(match.group(1))
            if "&end" in data:
                break
        energy_0: float = 0.0
        energy_1: torch.Tensor = torch.zeros([n_orbit, n_orbit], dtype=torch.float64)
        energy_2: torch.Tensor = torch.zeros([n_orbit, n_orbit, n_orbit, n_orbit], dtype=torch.float64)
        for line in file:
            pieces: list[str] = line.split()
            coefficient: float = float(pieces[0])
            sites: tuple[int, ...] = tuple(int(i) - 1 for i in pieces[1:])
            match sites:
                case (-1, -1, -1, -1):
                    energy_0 = coefficient
                case (i, j, -1, -1):
                    energy_1[i, j] = coefficient
                    energy_1[j, i] = coefficient
                case (i, j, k, l):
                    energy_2[i, j, k, l] = coefficient
                    energy_2[i, j, l, k] = coefficient
                    energy_2[j, i, k, l] = coefficient
                    energy_2[j, i, l, k] = coefficient
                    energy_2[l, k, j, i] = coefficient
                    energy_2[k, l, j, i] = coefficient
                    energy_2[l, k, i, j] = coefficient
                    energy_2[k, l, i, j] = coefficient
                case _:
                    raise ValueError(f"Invalid FCIDUMP format: {sites}")

    energy_2 = energy_2.permute(0, 2, 3, 1).contiguous() / 2
    energy_1_b: torch.Tensor = torch.zeros([n_orbit * 2, n_orbit * 2], dtype=torch.float64)
    energy_2_b: torch.Tensor = torch.zeros([n_orbit * 2, n_orbit * 2, n_orbit * 2, n_orbit * 2], dtype=torch.float64)
    energy_1_b[0::2, 0::2] = energy_1
    energy_1_b[1::2, 1::2] = energy_1
    energy_2_b[0::2, 0::2, 0::2, 0::2] = energy_2
    energy_2_b[0::2, 1::2, 1::2, 0::2] = energy_2
    energy_2_b[1::2, 0::2, 0::2, 1::2] = energy_2
    energy_2_b[1::2, 1::2, 1::2, 1::2] = energy_2

    interaction_operator: openfermion.InteractionOperator = openfermion.InteractionOperator(energy_0, energy_1_b.numpy(), energy_2_b.numpy())  # type: ignore[no-untyped-call]
    fermion_operator: openfermion.FermionOperator = openfermion.get_fermion_operator(interaction_operator)  # type: ignore[no-untyped-call]
    return openfermion.normal_ordered(fermion_operator)  # type: ignore[no-untyped-call]


class Model(OpenFermionModel):
    """
    This class handles the models from FCIDUMP files.
    """

    def __init__(self, model_name: str, model_path: pathlib.Path) -> None:
        # pylint: disable=super-init-not-called
        model_file_name: str = f"{model_path}/{model_name}.FCIDUMP"
        logging.info("Loading FCIDUMP Hamiltonian '%s' from file: %s", model_name, model_file_name)
        openfermion_hamiltonian: openfermion.FermionOperator = _read_fcidump(model_file_name)
        logging.info("FCIDUMP Hamiltonian '%s' successfully loaded", model_name)

        match = re.match(r"\w*_(\d*)_(\d*)", model_name)
        assert match is not None
        n_electrons, n_qubits = match.groups()
        self.n_qubits: int = int(n_qubits)
        self.n_electrons: int = int(n_electrons)
        logging.info("Identified %d qubits and %d electrons for model '%s'", self.n_qubits, self.n_electrons, model_name)

        self.ref_energy: float = torch.nan
        logging.info("Reference energy for model '%s' is currently undetermined", model_name)

        logging.info("Converting OpenFermion Hamiltonian to internal Hamiltonian representation")
        self.hamiltonian: Hamiltonian = Hamiltonian(openfermion_hamiltonian.terms, kind="fermi")
        logging.info("Internal Hamiltonian representation for model '%s' has been successfully created", model_name)


model_dict["fcidump"] = Model

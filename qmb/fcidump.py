"""
This file provides an interface to work with FCIDUMP files and reuses utilities from openfermion.
"""

import re
import json
import gzip
import logging
import pathlib
import hashlib
import torch
import openfermion
import platformdirs
from .hamiltonian import Hamiltonian
from .openfermion import Model as OpenFermionModel
from .model_dict import model_dict


def _read_fcidump(file_name: pathlib.Path) -> dict[tuple[tuple[int, int], ...], complex]:
    # pylint: disable=too-many-locals
    with gzip.open(file_name, "rt", encoding="utf-8") as file:
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
                case (_, -1, -1, -1):
                    # Psi4 writes additional non-standard one-electron integrals in this format, which we omit.
                    pass
                case (i, j, -1, -1):
                    energy_1[i, j] = coefficient
                    energy_1[j, i] = coefficient
                case (_, _, _, -1):
                    # In the standard FCIDUMP format, there is no such term.
                    raise ValueError(f"Invalid FCIDUMP format: {sites}")
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
    return {k: complex(v) for k, v in openfermion.normal_ordered(fermion_operator).terms.items()}  # type: ignore[no-untyped-call]


class Model(OpenFermionModel):
    """
    This class handles the models from FCIDUMP files.
    """

    def __init__(self, model_name: str, model_path: pathlib.Path) -> None:
        # pylint: disable=super-init-not-called
        model_file_name = model_path / f"{model_name}.FCIDUMP.gz"

        checksum = hashlib.sha256(model_file_name.read_bytes()).hexdigest() + "v5"
        cache_file = platformdirs.user_cache_path("qmb", "kclab") / checksum
        if cache_file.exists():
            logging.info("Loading FCIDUMP Hamiltonian '%s' from cache", model_name)
            openfermion_hamiltonian_data = torch.load(cache_file, map_location="cpu", weights_only=True)
            logging.info("FCIDUMP Hamiltonian '%s' successfully loaded", model_name)

            logging.info("Recovering internal Hamiltonian representation for model '%s'", model_name)
            self.hamiltonian = Hamiltonian(openfermion_hamiltonian_data, kind="fermi")
            logging.info("Internal Hamiltonian representation for model '%s' successfully recovered", model_name)
        else:
            logging.info("Loading FCIDUMP Hamiltonian '%s' from file: %s", model_name, model_file_name)
            openfermion_hamiltonian_dict = _read_fcidump(model_file_name)
            logging.info("FCIDUMP Hamiltonian '%s' successfully loaded", model_name)

            logging.info("Converting OpenFermion Hamiltonian to internal Hamiltonian representation")
            self.hamiltonian = Hamiltonian(openfermion_hamiltonian_dict, kind="fermi")
            logging.info("Internal Hamiltonian representation for model '%s' has been successfully created", model_name)

            logging.info("Caching OpenFermion Hamiltonian for model '%s'", model_name)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save((self.hamiltonian.site, self.hamiltonian.kind, self.hamiltonian.coef), cache_file)
            logging.info("OpenFermion Hamiltonian for model '%s' successfully cached", model_name)

        match = re.match(r"\w*_(\d*)_(\d*)(?:/.*)?", model_name)
        assert match is not None
        n_electrons, n_qubits = match.groups()
        self.n_qubits: int = int(n_qubits)
        self.n_electrons: int = int(n_electrons)
        logging.info("Identified %d qubits and %d electrons for model '%s'", self.n_qubits, self.n_electrons, model_name)

        with open(model_file_name.parent / "FCIDUMP.json", "rt", encoding="utf-8") as file:
            fcidump_ref_energy = json.load(file)
        self.ref_energy: float = fcidump_ref_energy.get(model_name.split("/")[-1], torch.nan)
        logging.info("Reference energy for model '%s' is %.10f", model_name, self.ref_energy)


model_dict["fcidump"] = Model

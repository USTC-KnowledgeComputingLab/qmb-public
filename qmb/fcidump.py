# This file implements interface to FCIDUMP, openfermion utilities reused.

import re
import logging
import torch
import openfermion
from . import _openfermion
from .openfermion import Model as OpenFermionModel


def read_fcidump(file_name):
    with open(file_name, "rt", encoding="utf-8") as file:
        for line in file:
            data = line.lower()
            match = re.search(r"norb\s*=\s*(\d+)", data)
            if match is not None:
                n_orbit = int(match.group(1))
            if "&end" in data:
                break
        energy_1 = torch.zeros([n_orbit, n_orbit])
        energy_2 = torch.zeros([n_orbit, n_orbit, n_orbit, n_orbit])
        for line in file:
            data = line.split()
            coefficient = float(data[0])
            sites = tuple(int(i) - 1 for i in data[1:])
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

    H = openfermion.FermionOperator()
    H += energy_0
    for p in range(n_orbit):
        for q in range(n_orbit):
            coef = energy_1[p, q].item()
            for i, j in ((2 * p, 2 * q), (2 * p + 1, 2 * q + 1)):
                H += openfermion.FermionOperator(((i, 1), (j, 0)), coef)
    for p in range(n_orbit):
        for q in range(n_orbit):
            for r in range(n_orbit):
                for s in range(n_orbit):
                    coef = energy_2[p, s, q, r].item() / 2
                    for i, j, k, l in (
                        ((2 * p), (2 * q), (2 * r), (2 * s)),
                        ((2 * p + 1), (2 * q + 1), (2 * r + 1), (2 * s + 1)),
                        ((2 * p + 1), (2 * q), (2 * r), (2 * s + 1)),
                        ((2 * p), (2 * q + 1), (2 * r + 1), (2 * s)),
                    ):
                        H += openfermion.FermionOperator(((i, 1), (j, 1), (k, 0), (l, 0)), coef)

    H = openfermion.normal_ordered(H)
    return H


class Model(OpenFermionModel):

    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.model_file_name = f"{self.model_path}/{self.model_name}.FCIDUMP"
        logging.info("loading operator of fcidump model %s from %s", self.model_name, self.model_file_name)
        self.openfermion = read_fcidump(self.model_file_name)
        logging.info("operator of fcidump model %s loaded", self.model_name)

        n_electrons, n_qubits = re.match(r"\w*_(\d*)_(\d*)", model_name).groups()
        self.n_qubits = int(n_qubits)
        self.n_electrons = int(n_electrons)
        logging.info("n_qubits: %d, n_electrons: %d", self.n_qubits, self.n_electrons)

        self.ref_energy = torch.nan
        logging.info("reference energy is unknown")

        logging.info("converting openfermion handle to hamiltonian handle")
        self.hamiltonian = _openfermion.Hamiltonian(list(self.openfermion.terms.items()))
        logging.info("hamiltonian handle has been created")

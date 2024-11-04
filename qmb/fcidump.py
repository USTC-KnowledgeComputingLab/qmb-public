# This file implements interface to FCIDUMP, openfermion utilities reused.

import re
import logging
import torch
import openfermion
from . import hamiltonian
from .openfermion import Model as OpenFermionModel
from .model_dict import model_dict


def read_fcidump(file_name):
    with open(file_name, "rt", encoding="utf-8") as file:
        for line in file:
            data = line.lower()
            match = re.search(r"norb\s*=\s*(\d+)", data)
            if match is not None:
                n_orbit = int(match.group(1))
            if "&end" in data:
                break
        energy_1 = torch.zeros([n_orbit, n_orbit], dtype=torch.float64)
        energy_2 = torch.zeros([n_orbit, n_orbit, n_orbit, n_orbit], dtype=torch.float64)
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

    energy_2 = energy_2.permute(0, 2, 3, 1).contiguous() / 2
    energy_1_b = torch.zeros([n_orbit * 2, n_orbit * 2], dtype=torch.float64)
    energy_2_b = torch.zeros([n_orbit * 2, n_orbit * 2, n_orbit * 2, n_orbit * 2], dtype=torch.float64)
    energy_1_b[0::2, 0::2] = energy_1
    energy_1_b[1::2, 1::2] = energy_1
    energy_2_b[0::2, 0::2, 0::2, 0::2] = energy_2
    energy_2_b[0::2, 1::2, 1::2, 0::2] = energy_2
    energy_2_b[1::2, 0::2, 0::2, 1::2] = energy_2
    energy_2_b[1::2, 1::2, 1::2, 1::2] = energy_2

    interaction_operator = openfermion.ops.InteractionOperator(energy_0, energy_1_b.numpy(), energy_2_b.numpy())
    fermion_operator = openfermion.transforms.get_fermion_operator(interaction_operator)
    return openfermion.normal_ordered(fermion_operator)


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
        self.hamiltonian = hamiltonian.FermiHamiltonian(self.openfermion.terms)
        logging.info("hamiltonian handle has been created")


model_dict["fcidump"] = Model

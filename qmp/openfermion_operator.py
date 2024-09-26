# This file implements interface to openfermion model, but read the fermion operators directly.

import re
import logging
import numpy
import torch
from .openfermion import Model as OpenFermionModel
from . import _openfermion


class Model(OpenFermionModel):

    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.model_file_name = f"{self.model_path}/{self.model_name}.npy"
        logging.info("loading operator of openfermion model %s from %s", self.model_name, self.model_file_name)
        self.openfermion = numpy.load(self.model_file_name, allow_pickle=True).item()
        logging.info("operator of openfermion model %s loaded", self.model_name)

        n_electrons, n_qubits = re.match(r"\w*_(\d*)_(\d*)", model_name).groups()
        self.n_qubits = int(n_qubits)
        self.n_electrons = int(n_electrons)
        logging.info("n_qubits: %d, n_electrons: %d", self.n_qubits, self.n_electrons)

        self.ref_energy = torch.nan
        logging.info("reference energy is unknown")

        logging.info("converting openfermion handle to hamiltonian handle")
        self.hamiltonian = _openfermion.Hamiltonian(list(self.openfermion.terms.items()))
        logging.info("hamiltonian handle has been created")

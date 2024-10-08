# This file implements interface to openfermion model.

import typing
import logging
import pathlib
import dataclasses
import torch
import tyro
import openfermion
from . import _openfermion
from . import naqs as naqs_m
from . import attention as attention_m


@dataclasses.dataclass
class ModelConfig:
    # The openfermion model name
    model_name: typing.Annotated[str, tyro.conf.Positional, tyro.conf.arg(metavar="MODEL")]
    # The path of models folder
    model_path: typing.Annotated[pathlib.Path, tyro.conf.arg(aliases=["-M"])] = pathlib.Path("models")


@dataclasses.dataclass
class NaqsConfig:
    # The hidden widths of the network
    hidden: typing.Annotated[tuple[int, ...], tyro.conf.arg(aliases=["-w"])] = (512,)


@dataclasses.dataclass
class AttentionConfig:
    # Embedding dimension
    embedding_dim: typing.Annotated[int, tyro.conf.arg(aliases=["-e"])] = 512
    # Heads number
    heads_num: typing.Annotated[int, tyro.conf.arg(aliases=["-m"])] = 8
    # Feedforward dimension
    feed_forward_dim: typing.Annotated[int, tyro.conf.arg(aliases=["-f"])] = 2048
    # Network depth
    depth: typing.Annotated[int, tyro.conf.arg(aliases=["-d"])] = 6


class Model:

    @classmethod
    def preparse(cls, input_args):
        args = tyro.cli(ModelConfig, args=input_args)
        return args.model_name

    @classmethod
    def parse(cls, input_args):
        logging.info("parsing args %a by openfermion model", input_args)
        args = tyro.cli(ModelConfig, args=input_args)
        logging.info("model name: %s, model path: %s", args.model_name, args.model_path)

        return cls(args.model_name, args.model_path)

    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.model_file_name = f"{self.model_path}/{self.model_name}.hdf5"
        logging.info("loading openfermion model %s from %s", self.model_name, self.model_file_name)
        self.openfermion = openfermion.MolecularData(filename=self.model_file_name)
        logging.info("openfermion model %s loaded", self.model_name)

        self.n_qubits = self.openfermion.n_qubits
        self.n_electrons = self.openfermion.n_electrons
        logging.info("n_qubits: %d, n_electrons: %d", self.n_qubits, self.n_electrons)

        self.ref_energy = self.openfermion.fci_energy.item()
        logging.info("reference energy in openfermion data is %.10f", self.ref_energy)

        logging.info("converting openfermion handle to hamiltonian handle")
        self.hamiltonian = _openfermion.Hamiltonian(list(openfermion.transforms.get_fermion_operator(self.openfermion.get_molecular_hamiltonian()).terms.items()))
        logging.info("hamiltonian handle has been created")

    def inside(self, configs):
        return self.hamiltonian.inside(configs)

    def outside(self, configs):
        return self.hamiltonian.outside(configs)

    def naqs(self, input_args):
        logging.info("parsing args %a by network naqs", input_args)
        args = tyro.cli(NaqsConfig, args=input_args)
        logging.info("hidden: %a", args.hidden)

        network = naqs_m.WaveFunction(
            double_sites=self.n_qubits,
            physical_dim=2,
            is_complex=True,
            spin_up=self.n_electrons // 2,
            spin_down=self.n_electrons // 2,
            hidden_size=args.hidden,
            ordering=+1,
        ).double()

        return torch.jit.script(network)

    def attention(self, input_args):
        logging.info("parsing args %a by network attention", input_args)
        args = tyro.cli(AttentionConfig, args=input_args)
        logging.info("embedding dim: %d, heads_num: %d, feed forward dim: %d, depth: %d", args.embedding_dim, args.heads_num, args.feed_forward_dim, args.depth)

        network = attention_m.WaveFunction(
            double_sites=self.n_qubits,
            physical_dim=2,
            is_complex=True,
            spin_up=self.n_electrons // 2,
            spin_down=self.n_electrons // 2,
            embedding_dim=args.embedding_dim,
            heads_num=args.heads_num,
            feed_forward_dim=args.feed_forward_dim,
            depth=args.depth,
            ordering=+1,
        ).double()

        return torch.jit.script(network)

"""
This file provides an interface to work with openfermion models.
"""

import typing
import logging
import pathlib
import dataclasses
import torch
import tyro
import openfermion
from . import naqs as naqs_m
from . import attention as attention_m
from . import hamiltonian
from .model_dict import model_dict, ModelProto, NetworkProto


@dataclasses.dataclass
class ModelConfig:
    """
    The configuration of the model.
    """

    # The openfermion model name
    model_name: typing.Annotated[str, tyro.conf.Positional, tyro.conf.arg(metavar="MODEL")]
    # The path of models folder
    model_path: typing.Annotated[pathlib.Path, tyro.conf.arg(aliases=["-M"])] = pathlib.Path("models")


class Model(ModelProto["Model"]):
    """
    This class handles the openfermion model.
    """

    network_dict: dict[str, typing.Callable[["Model", tuple[str, ...]], NetworkProto]] = {}

    @classmethod
    def preparse(cls, input_args: tuple[str, ...]) -> str:
        args: ModelConfig = tyro.cli(ModelConfig, args=input_args)
        return args.model_name

    @classmethod
    def parse(cls, input_args: tuple[str, ...]) -> "Model":
        logging.info("parsing args %a for openfermion model", input_args)
        args = tyro.cli(ModelConfig, args=input_args)
        logging.info("arguments parsed, model name: %s, model path: %s", args.model_name, args.model_path)

        return cls(args.model_name, args.model_path)

    def __init__(self, model_name: str, model_path: pathlib.Path) -> None:
        self.model_name: str = model_name
        self.model_path: pathlib.Path = model_path
        self.model_file_name: str = f"{self.model_path}/{self.model_name}.hdf5"
        logging.info("loading openfermion model %s from %s", self.model_name, self.model_file_name)
        openfermion_model: openfermion.MolecularData = openfermion.MolecularData(filename=self.model_file_name)  # type: ignore[no-untyped-call]
        logging.info("openfermion model %s has been loaded", self.model_name)

        self.n_qubits: int = typing.cast(int, openfermion_model.n_qubits)
        self.n_electrons: int = typing.cast(int, openfermion_model.n_electrons)
        logging.info("openfermion model parameter: n_qubits=%d, n_electrons=%d", self.n_qubits, self.n_electrons)

        self.ref_energy: float = float(openfermion_model.fci_energy)  # type: ignore[arg-type]
        logging.info("reference energy in openfermion data is %.10f", self.ref_energy)

        logging.info("converting openfermion handle to hamiltonian handle")
        self.hamiltonian: hamiltonian.Hamiltonian = hamiltonian.Hamiltonian(
            openfermion.transforms.get_fermion_operator(openfermion_model.get_molecular_hamiltonian()).terms,  # type: ignore[no-untyped-call]
            kind="fermi",
        )
        logging.info("hamiltonian handle has been created")

    def inside(self, configs_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hamiltonian.inside(configs_i)

    def outside(self, configs_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.hamiltonian.outside(configs_i)


model_dict["openfermion"] = Model


@dataclasses.dataclass
class NaqsConfig:
    """
    The configuration of the NAQS network.
    """

    # The hidden widths of the network
    hidden: typing.Annotated[tuple[int, ...], tyro.conf.arg(aliases=["-w"])] = (512,)

    @classmethod
    def create(cls, model: Model, input_args: tuple[str, ...]) -> NetworkProto:
        """
        Create a NAQS network for the model.
        """
        logging.info("parsing args %a by network naqs", input_args)
        args = tyro.cli(cls, args=input_args)
        logging.info("hidden: %a", args.hidden)

        network = naqs_m.WaveFunction(
            double_sites=model.n_qubits,
            physical_dim=2,
            is_complex=True,
            spin_up=model.n_electrons // 2,
            spin_down=model.n_electrons // 2,
            hidden_size=args.hidden,
            ordering=+1,
        ).double()

        return torch.jit.script(network)


Model.network_dict["naqs"] = NaqsConfig.create


@dataclasses.dataclass
class AttentionConfig:
    """
    The configuration of the attention network.
    """

    # Embedding dimension
    embedding_dim: typing.Annotated[int, tyro.conf.arg(aliases=["-e"])] = 512
    # Heads number
    heads_num: typing.Annotated[int, tyro.conf.arg(aliases=["-m"])] = 8
    # Feedforward dimension
    feed_forward_dim: typing.Annotated[int, tyro.conf.arg(aliases=["-f"])] = 2048
    # Network depth
    depth: typing.Annotated[int, tyro.conf.arg(aliases=["-d"])] = 6

    @classmethod
    def create(cls, model: Model, input_args: tuple[str, ...]) -> NetworkProto:
        """
        Create an attention network for the model.
        """
        logging.info("parsing args %a by network attention", input_args)
        args = tyro.cli(cls, args=input_args)
        logging.info("embedding dim: %d, heads_num: %d, feed forward dim: %d, depth: %d", args.embedding_dim, args.heads_num, args.feed_forward_dim, args.depth)

        network = attention_m.WaveFunction(
            double_sites=model.n_qubits,
            physical_dim=2,
            is_complex=True,
            spin_up=model.n_electrons // 2,
            spin_down=model.n_electrons // 2,
            embedding_dim=args.embedding_dim,
            heads_num=args.heads_num,
            feed_forward_dim=args.feed_forward_dim,
            depth=args.depth,
            ordering=+1,
        ).double()

        return torch.jit.script(network)


Model.network_dict["attention"] = AttentionConfig.create

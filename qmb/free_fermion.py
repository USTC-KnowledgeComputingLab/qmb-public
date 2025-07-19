"""
This file offers an interface for defining Free fermion models on a two-dimensional lattice.
"""

import typing
import logging
import dataclasses
import torch
import tyro
from .mlp import WaveFunctionNormal as MlpWaveFunction
from .attention import WaveFunctionNormal as AttentionWaveFunction
from .hamiltonian import Hamiltonian
from .model_dict import model_dict, ModelProto, NetworkProto, NetworkConfigProto


@dataclasses.dataclass
class ModelConfig:
    """
    The configuration for the Free fermion model.
    """

    # The width of the free fermion lattice
    m: typing.Annotated[int, tyro.conf.Positional]
    # The height of the free fermion lattice
    n: typing.Annotated[int, tyro.conf.Positional]

    # The electron number, left empty for half-filling
    electron_number: typing.Annotated[int, tyro.conf.arg(aliases=["-e"])]

    def __post_init__(self) -> None:
        if self.electron_number is None:
            raise ValueError("Electron number must be provided.")

        if self.m <= 0 or self.n <= 0:
            raise ValueError("The dimensions of the Free fermion model must be positive integers.")

        if self.electron_number < 0 or self.electron_number > self.m * self.n:
            raise ValueError(f"The electron number {self.electron_number} is out of bounds for a {self.m}x{self.n} lattice. Each site can host up to one electrons (all spin up or spin down).")


class Model(ModelProto[ModelConfig]):
    """
    This class handles the Free fermion model.
    """

    network_dict: dict[str, type[NetworkConfigProto["Model"]]] = {}

    config_t = ModelConfig

    @classmethod
    def default_group_name(cls, config: ModelConfig) -> str:
        return f"FreeFermion_{config.m}x{config.n}_e{config.electron_number}"

    @classmethod
    def _prepare_hamiltonian(cls, args: ModelConfig) -> dict[tuple[tuple[int, int], ...], complex]:

        def _index(i: int, j: int) -> int:
            return i + j * args.m

        hamiltonian_dict: dict[tuple[tuple[int, int], ...], complex] = {}
        for i in range(args.m):
            for j in range(args.n):
                # Nearest neighbor hopping
                if i != 0:
                    hamiltonian_dict[(_index(i, j), 1), (_index(i - 1, j), 0)] = 1
                    hamiltonian_dict[(_index(i - 1, j), 1), (_index(i, j), 0)] = 1
                if j != 0:
                    hamiltonian_dict[(_index(i, j), 1), (_index(i, j - 1), 0)] = 1
                    hamiltonian_dict[(_index(i, j - 1), 1), (_index(i, j), 0)] = 1

        return hamiltonian_dict

    @classmethod
    def _calculate_ref_energy(cls, args: ModelConfig, hamiltonian_dict: dict[tuple[tuple[int, int], ...], complex]) -> float:
        sites = args.m * args.n
        hamiltonian = torch.zeros((sites,) * 2, dtype=torch.complex128)  # device?
        for operators, intensity in hamiltonian_dict.items():
            if len(operators) == 2:
                coordinate = tuple({n: j for j, n in operators}[i] for i in [0, 1])
                hamiltonian[coordinate] = torch.tensor(intensity, dtype=torch.complex128)  # device?
        return sum(torch.linalg.eigh(hamiltonian).eigenvalues[:args.electron_number])  # pylint: disable=not-callable

    def __init__(
        self,
        args: ModelConfig,
    ):
        logging.info("Input arguments successfully parsed")

        assert args.electron_number is not None
        self.m: int = args.m
        self.n: int = args.n
        self.electron_number: int = args.electron_number
        logging.info("Constructing Free fermion model with dimensions: width = %d, height = %d", self.m, self.n)
        logging.info("The parameters of the model are: N = %d", args.electron_number)

        logging.info("Initializing Hamiltonian for the lattice")
        hamiltonian_dict: dict[tuple[tuple[int, int], ...], complex] = self._prepare_hamiltonian(args)
        logging.info("Hamiltonian initialization complete")

        self.ref_energy: float = self._calculate_ref_energy(args, hamiltonian_dict)
        logging.info("The ref energy is %.10f", self.ref_energy)

        logging.info("Converting the Hamiltonian to internal Hamiltonian representation")
        self.hamiltonian: Hamiltonian = Hamiltonian(hamiltonian_dict, kind="fermi")
        logging.info("Internal Hamiltonian representation for model has been successfully created")

    def apply_within(self, configs_i: torch.Tensor, psi_i: torch.Tensor, configs_j: torch.Tensor) -> torch.Tensor:
        return self.hamiltonian.apply_within(configs_i, psi_i, configs_j)

    def find_relative(self, configs_i: torch.Tensor, psi_i: torch.Tensor, count_selected: int, configs_exclude: torch.Tensor | None = None) -> torch.Tensor:
        return self.hamiltonian.find_relative(configs_i, psi_i, count_selected, configs_exclude)

    def single_relative(self, configs: torch.Tensor) -> torch.Tensor:
        return self.hamiltonian.single_relative(configs)

    def show_config(self, config: torch.Tensor) -> str:
        trans_table = {ord("0"): " ", ord("1"): "x"}
        string = "".join(f"{i:08b}"[::-1] for i in config.cpu().numpy()).translate(trans_table)
        return "[" + ".".join(string[(j * self.m):((j + 1) * self.m)] for j in range(self.n)) + "]"


model_dict["free_fermion"] = Model


@dataclasses.dataclass
class MlpConfig:
    """
    The configuration of the MLP network.
    """

    # The hidden widths of the network
    hidden: typing.Annotated[tuple[int, ...], tyro.conf.arg(aliases=["-w"])] = (512,)

    def create(self, model: Model) -> NetworkProto:
        """
        Create a MLP network for the model.
        """
        logging.info("Hidden layer widths: %a", self.hidden)

        network = MlpWaveFunction(
            sites=model.m * model.n,
            physical_dim=2,
            is_complex=True,
            hidden_size=self.hidden,
            ordering=+1,
        )

        return network


Model.network_dict["mlp"] = MlpConfig


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
    # Shared expert number
    shared_expert_num: typing.Annotated[int, tyro.conf.arg(aliases=["-s"])] = 1
    # Routed expert number
    routed_expert_num: typing.Annotated[int, tyro.conf.arg(aliases=["-r"])] = 0
    # Selected expert number
    selected_expert_num: typing.Annotated[int, tyro.conf.arg(aliases=["-c"])] = 0
    # Network depth
    depth: typing.Annotated[int, tyro.conf.arg(aliases=["-d"])] = 6

    def create(self, model: Model) -> NetworkProto:
        """
        Create an attention network for the model.
        """
        logging.info(
            "Attention network configuration: "
            "embedding dimension: %d, "
            "number of heads: %d, "
            "feed-forward dimension: %d, "
            "shared expert number: %d, "
            "routed expert number: %d, "
            "selected expert number: %d, "
            "depth: %d",
            self.embedding_dim,
            self.heads_num,
            self.feed_forward_dim,
            self.shared_expert_num,
            self.routed_expert_num,
            self.selected_expert_num,
            self.depth,
        )

        network = AttentionWaveFunction(
            sites=model.m * model.n,
            physical_dim=2,
            is_complex=True,
            embedding_dim=self.embedding_dim,
            heads_num=self.heads_num,
            feed_forward_dim=self.feed_forward_dim,
            shared_num=self.shared_expert_num,
            routed_num=self.routed_expert_num,
            selected_num=self.selected_expert_num,
            depth=self.depth,
            ordering=+1,
        )

        return network


Model.network_dict["attention"] = AttentionConfig

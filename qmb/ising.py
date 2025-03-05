"""
This file offers a interface for defining Ising-like models on a two-dimensional lattice.
"""

import typing
import logging
import dataclasses
import collections
import torch
import tyro
from .naqs import WaveFunctionNormal as NaqsWaveFunction
from .attention import WaveFunctionNormal as AttentionWaveFunction
from .hamiltonian import Hamiltonian
from .model_dict import model_dict, ModelProto, NetworkProto


@dataclasses.dataclass
class ModelConfig:
    """
    The configuration for the Ising-like model.
    """

    # pylint: disable=too-many-instance-attributes

    # The width of the ising lattice
    m: typing.Annotated[int, tyro.conf.Positional]
    # The height of the ising lattice
    n: typing.Annotated[int, tyro.conf.Positional]

    # The coefficient of X
    x: typing.Annotated[float, tyro.conf.arg(aliases=["-xe"])] = 0
    # The coefficient of Y
    y: typing.Annotated[float, tyro.conf.arg(aliases=["-ye"])] = 0
    # The coefficient of Z
    z: typing.Annotated[float, tyro.conf.arg(aliases=["-ze"])] = 0
    # The coefficient of XX for horizontal bond
    xh: typing.Annotated[float, tyro.conf.arg(aliases=["-xh"])] = 0
    # The coefficient of YY for horizontal bond
    yh: typing.Annotated[float, tyro.conf.arg(aliases=["-yh"])] = 0
    # The coefficient of ZZ for horizontal bond
    zh: typing.Annotated[float, tyro.conf.arg(aliases=["-zh"])] = 0
    # The coefficient of XX for vertical bond
    xv: typing.Annotated[float, tyro.conf.arg(aliases=["-xv"])] = 0
    # The coefficient of YY for vertical bond
    yv: typing.Annotated[float, tyro.conf.arg(aliases=["-yv"])] = 0
    # The coefficient of ZZ for vertical bond
    zv: typing.Annotated[float, tyro.conf.arg(aliases=["-zv"])] = 0
    # The coefficient of XX for diagonal bond
    xd: typing.Annotated[float, tyro.conf.arg(aliases=["-xd"])] = 0
    # The coefficient of YY for diagonal bond
    yd: typing.Annotated[float, tyro.conf.arg(aliases=["-yd"])] = 0
    # The coefficient of ZZ for diagonal bond
    zd: typing.Annotated[float, tyro.conf.arg(aliases=["-zd"])] = 0
    # The coefficient of XX for antidiagonal bond
    xa: typing.Annotated[float, tyro.conf.arg(aliases=["-xa"])] = 0
    # The coefficient of YY for antidiagonal bond
    ya: typing.Annotated[float, tyro.conf.arg(aliases=["-ya"])] = 0
    # The coefficient of ZZ for antidiagonal bond
    za: typing.Annotated[float, tyro.conf.arg(aliases=["-za"])] = 0


class Model(ModelProto):
    """
    This class handles the Ising-like model.
    """

    network_dict: dict[str, typing.Callable[["Model", tuple[str, ...]], NetworkProto]] = {}

    @classmethod
    def preparse(cls, input_args: tuple[str, ...]) -> str:
        # pylint: disable=too-many-locals
        args = tyro.cli(ModelConfig, args=input_args)
        x = f"_x{args.x}" if args.x != 0 else ""
        y = f"_y{args.y}" if args.y != 0 else ""
        z = f"_z{args.z}" if args.z != 0 else ""
        xh = f"_xh{args.xh}" if args.xh != 0 else ""
        yh = f"_yh{args.yh}" if args.yh != 0 else ""
        zh = f"_zh{args.zh}" if args.zh != 0 else ""
        xv = f"_xv{args.xv}" if args.xv != 0 else ""
        yv = f"_yv{args.yv}" if args.yv != 0 else ""
        zv = f"_zv{args.zv}" if args.zv != 0 else ""
        xd = f"_xd{args.xd}" if args.xd != 0 else ""
        yd = f"_yd{args.yd}" if args.yd != 0 else ""
        zd = f"_zd{args.zd}" if args.zd != 0 else ""
        xa = f"_xa{args.xa}" if args.xa != 0 else ""
        ya = f"_ya{args.ya}" if args.ya != 0 else ""
        za = f"_za{args.za}" if args.za != 0 else ""
        desc = x + y + z + xh + yh + zh + xv + yv + zv + xd + yd + zd + xa + ya + za
        return f"Ising_{args.m}_{args.n}" + desc

    @classmethod
    def parse(cls, input_args: tuple[str, ...]) -> "Model":
        logging.info("Parsing arguments for the model: %a", input_args)
        args = tyro.cli(ModelConfig, args=input_args)
        logging.info("Input arguments successfully parsed")
        logging.info("Grid dimensions: width = %d, height = %d", args.m, args.n)
        logging.info("Element-wise coefficients: X = %.10f, Y = %.10f, Z = %.10f", args.x, args.y, args.z)
        logging.info("Horizontal bond coefficients: X = %.10f, Y = %.10f, Z = %.10f", args.xh, args.yh, args.zh)
        logging.info("Vertical bond coefficients: X = %.10f, Y = %.10f, Z = %.10f", args.xv, args.yv, args.zv)
        logging.info("Diagonal bond coefficients: X = %.10f, Y = %.10f, Z = %.10f", args.xd, args.yd, args.zd)
        logging.info("Anti-diagonal bond coefficients: X = %.10f, Y = %.10f, Z = %.10f", args.xa, args.ya, args.za)

        return cls(args)

    @classmethod
    def _prepare_hamiltonian(cls, args: ModelConfig) -> dict[tuple[tuple[int, int], ...], complex]:
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-nested-blocks

        def _index(i: int, j: int) -> int:
            return i + j * args.m

        def _x(i: int, j: int) -> tuple[tuple[tuple[tuple[int, int], ...], complex], ...]:
            return (
                (((_index(i, j), 1),), +1),
                (((_index(i, j), 0),), +1),
            )

        def _y(i: int, j: int) -> tuple[tuple[tuple[tuple[int, int], ...], complex], ...]:
            return (
                (((_index(i, j), 1),), -1j),
                (((_index(i, j), 0),), +1j),
            )

        def _z(i: int, j: int) -> tuple[tuple[tuple[tuple[int, int], ...], complex], ...]:
            return (
                (((_index(i, j), 1), (_index(i, j), 0)), +1),
                (((_index(i, j), 0), (_index(i, j), 1)), -1),
            )

        hamiltonian: dict[tuple[tuple[int, int], ...], complex] = collections.defaultdict(lambda: 0)
        # Express spin pauli matrix in hard core boson language.
        for i in range(args.m):
            for j in range(args.n):
                k: tuple[tuple[int, int], ...]
                k1: tuple[tuple[int, int], ...]
                k2: tuple[tuple[int, int], ...]
                v: complex
                v1: complex
                v2: complex
                if True:  # pylint: disable=using-constant-test
                    if args.x != 0:
                        for k, v in _x(i, j):
                            hamiltonian[k] += v * args.x
                    if args.y != 0:
                        for k, v in _y(i, j):
                            hamiltonian[k] += v * args.y
                    if args.z != 0:
                        for k, v in _z(i, j):
                            hamiltonian[k] += v * args.z
                if i != 0:
                    if args.xh != 0:
                        for k1, v1 in _x(i, j):
                            for k2, v2 in _x(i - 1, j):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.xh
                    if args.yh != 0:
                        for k1, v1 in _y(i, j):
                            for k2, v2 in _y(i - 1, j):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.yh
                    if args.zh != 0:
                        for k1, v1 in _z(i, j):
                            for k2, v2 in _z(i - 1, j):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.zh
                if j != 0:
                    if args.xv != 0:
                        for k1, v1 in _x(i, j):
                            for k2, v2 in _x(i, j - 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.xv
                    if args.yv != 0:
                        for k1, v1 in _y(i, j):
                            for k2, v2 in _y(i, j - 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.yv
                    if args.zv != 0:
                        for k1, v1 in _z(i, j):
                            for k2, v2 in _z(i, j - 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.zv
                if i != 0 and j != 0:
                    if args.xd != 0:
                        for k1, v1 in _x(i, j):
                            for k2, v2 in _x(i - 1, j - 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.xd
                    if args.yd != 0:
                        for k1, v1 in _y(i, j):
                            for k2, v2 in _y(i - 1, j - 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.yd
                    if args.zd != 0:
                        for k1, v1 in _z(i, j):
                            for k2, v2 in _z(i - 1, j - 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.zd
                if i != 0 and j != args.n - 1:
                    if args.xa != 0:
                        for k1, v1 in _x(i, j):
                            for k2, v2 in _x(i - 1, j + 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.xa
                    if args.ya != 0:
                        for k1, v1 in _y(i, j):
                            for k2, v2 in _y(i - 1, j + 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.ya
                    if args.za != 0:
                        for k1, v1 in _z(i, j):
                            for k2, v2 in _z(i - 1, j + 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.za
        return hamiltonian

    def __init__(self, args: ModelConfig) -> None:
        self.m: int = args.m
        self.n: int = args.n
        logging.info("Constructing Ising model with dimensions: width = %d, height = %d", self.m, self.n)
        logging.info("Element-wise coefficients: X = %.10f, Y = %.10f, Z = %.10f", args.x, args.y, args.z)
        logging.info("Horizontal bond coefficients: X = %.10f, Y = %.10f, Z = %.10f", args.xh, args.yh, args.zh)
        logging.info("Vertical bond coefficients: X = %.10f, Y = %.10f, Z = %.10f", args.xv, args.yv, args.zv)
        logging.info("Diagonal bond coefficients: X = %.10f, Y = %.10f, Z = %.10f", args.xd, args.yd, args.zd)
        logging.info("Anti-diagonal bond coefficients: X = %.10f, Y = %.10f, Z = %.10f", args.xa, args.ya, args.za)

        logging.info("Initializing Hamiltonian for the lattice")
        hamiltonian_dict: dict[tuple[tuple[int, int], ...], complex] = self._prepare_hamiltonian(args)
        logging.info("Hamiltonian initialization complete")

        self.ref_energy: float = torch.nan

        logging.info("Converting the Hamiltonian to internal Hamiltonian representation")
        self.hamiltonian: Hamiltonian = Hamiltonian(hamiltonian_dict, kind="bose2")
        logging.info("Internal Hamiltonian representation for model has been successfully created")

    def apply_within(self, configs_i: torch.Tensor, psi_i: torch.Tensor, configs_j: torch.Tensor) -> torch.Tensor:
        return self.hamiltonian.apply_within(configs_i, psi_i, configs_j)

    def find_relative(self, configs_i: torch.Tensor, psi_i: torch.Tensor, count_selected: int, configs_exclude: torch.Tensor | None = None) -> torch.Tensor:
        return self.hamiltonian.find_relative(configs_i, psi_i, count_selected, configs_exclude)

    def show_config(self, config: torch.Tensor) -> str:
        string = "".join(f"{i:08b}"[::-1] for i in config.cpu().numpy())
        return "[" + ".".join("".join("↑" if string[i + j * self.m] == "0" else "↓" for i in range(self.m)) for j in range(self.n)) + "]"


model_dict["ising"] = Model


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
        logging.info("Parsing arguments for NAQS network: %a", input_args)
        args = tyro.cli(cls, args=input_args)
        logging.info("Hidden layer widths: %a", args.hidden)

        network = NaqsWaveFunction(
            sites=model.m * model.n,
            physical_dim=2,
            is_complex=True,
            hidden_size=args.hidden,
            ordering=+1,
        )

        return network


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
    # Shared expert number
    shared_expert_num: typing.Annotated[int, tyro.conf.arg(aliases=["-s"])] = 1
    # Routed expert number
    routed_expert_num: typing.Annotated[int, tyro.conf.arg(aliases=["-r"])] = 0
    # Selected expert number
    selected_expert_num: typing.Annotated[int, tyro.conf.arg(aliases=["-c"])] = 0
    # Network depth
    depth: typing.Annotated[int, tyro.conf.arg(aliases=["-d"])] = 6

    @classmethod
    def create(cls, model: Model, input_args: tuple[str, ...]) -> NetworkProto:
        """
        Create an attention network for the model.
        """
        logging.info("Parsing arguments for attention network: %a", input_args)
        args = tyro.cli(cls, args=input_args)
        logging.info(
            "Attention network configuration: "
            "embedding dimension: %d, "
            "number of heads: %d, "
            "feed-forward dimension: %d, "
            "shared expert number: %d, "
            "routed expert number: %d, "
            "selected expert number: %d, "
            "depth: %d",
            args.embedding_dim,
            args.heads_num,
            args.feed_forward_dim,
            args.shared_expert_num,
            args.routed_expert_num,
            args.selected_expert_num,
            args.depth,
        )

        network = AttentionWaveFunction(
            sites=model.m * model.n,
            physical_dim=2,
            is_complex=True,
            embedding_dim=args.embedding_dim,
            heads_num=args.heads_num,
            feed_forward_dim=args.feed_forward_dim,
            shared_num=args.shared_expert_num,
            routed_num=args.routed_expert_num,
            selected_num=args.selected_expert_num,
            depth=args.depth,
            ordering=+1,
        )

        return network


Model.network_dict["attention"] = AttentionConfig.create

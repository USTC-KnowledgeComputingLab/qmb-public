# This file declare ising-like model on two dimensional lattice.

from collections import defaultdict
import typing
import logging
import dataclasses
import torch
import tyro
from . import naqs as naqs_m
from . import hamiltonian
from .model_dict import model_dict, ModelProto, NetworkProto


@dataclasses.dataclass
class ModelConfig:
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
    hx: typing.Annotated[float, tyro.conf.arg(aliases=["-xh"])] = 0
    # The coefficient of YY for horizontal bond
    hy: typing.Annotated[float, tyro.conf.arg(aliases=["-yh"])] = 0
    # The coefficient of ZZ for horizontal bond
    hz: typing.Annotated[float, tyro.conf.arg(aliases=["-zh"])] = 0
    # The coefficient of XX for vertical bond
    vx: typing.Annotated[float, tyro.conf.arg(aliases=["-xv"])] = 0
    # The coefficient of YY for vertical bond
    vy: typing.Annotated[float, tyro.conf.arg(aliases=["-yv"])] = 0
    # The coefficient of ZZ for vertical bond
    vz: typing.Annotated[float, tyro.conf.arg(aliases=["-zv"])] = 0
    # The coefficient of XX for diagonal bond
    dx: typing.Annotated[float, tyro.conf.arg(aliases=["-xd"])] = 0
    # The coefficient of YY for diagonal bond
    dy: typing.Annotated[float, tyro.conf.arg(aliases=["-yd"])] = 0
    # The coefficient of ZZ for diagonal bond
    dz: typing.Annotated[float, tyro.conf.arg(aliases=["-zd"])] = 0
    # The coefficient of XX for antidiagonal bond
    ax: typing.Annotated[float, tyro.conf.arg(aliases=["-xa"])] = 0
    # The coefficient of YY for antidiagonal bond
    ay: typing.Annotated[float, tyro.conf.arg(aliases=["-ya"])] = 0
    # The coefficient of ZZ for antidiagonal bond
    az: typing.Annotated[float, tyro.conf.arg(aliases=["-za"])] = 0


class Model(ModelProto["Model"]):

    network_dict: dict[str, typing.Callable[["Model", tuple[str, ...]], NetworkProto]] = {}

    @classmethod
    def preparse(cls, input_args):
        args = tyro.cli(ModelConfig, args=input_args)
        x = f"_hx{args.x}" if args.x != 0 else ""
        y = f"_hy{args.y}" if args.y != 0 else ""
        z = f"_hz{args.z}" if args.z != 0 else ""
        hx = f"_hx{args.hx}" if args.hx != 0 else ""
        hy = f"_hy{args.hy}" if args.hy != 0 else ""
        hz = f"_hz{args.hz}" if args.hz != 0 else ""
        vx = f"_vx{args.vx}" if args.vx != 0 else ""
        vy = f"_vy{args.vy}" if args.vy != 0 else ""
        vz = f"_vz{args.vz}" if args.vz != 0 else ""
        dx = f"_dx{args.dx}" if args.dx != 0 else ""
        dy = f"_dy{args.dy}" if args.dy != 0 else ""
        dz = f"_dz{args.dz}" if args.dz != 0 else ""
        ax = f"_ax{args.ax}" if args.ax != 0 else ""
        ay = f"_ay{args.ay}" if args.ay != 0 else ""
        az = f"_az{args.az}" if args.az != 0 else ""
        desc = x + y + z + hx + hy + hz + vx + vy + vz + dx + dy + dz + ax + ay + az
        return f"Ising_{args.m}_{args.n}" + desc

    @classmethod
    def parse(cls, input_args):
        logging.info("parsing args %a by ising model", input_args)
        args = tyro.cli(ModelConfig, args=input_args)
        logging.info("width: %d, height: %d", args.m, args.n)
        logging.info("elementwise x: %.10f, y: %.10f, z: %.10f", args.x, args.y, args.z)
        logging.info("horizontal x: %.10f, y: %.10f, z: %.10f", args.hx, args.hy, args.hz)
        logging.info("vertical x: %.10f, y: %.10f, z: %.10f", args.vx, args.vy, args.vz)
        logging.info("diagonal x: %.10f, y: %.10f, z: %.10f", args.dx, args.dy, args.dz)
        logging.info("antidiagonal x: %.10f, y: %.10f, z: %.10f", args.ax, args.ay, args.az)

        return cls(args)

    @classmethod
    def _prepare_hamiltonian(cls, args: ModelConfig):

        def _index(i, j):
            return i + j * args.m

        def _x(i, j):
            return (((_index(i, j), 1),), +1), (((_index(i, j), 0),), +1)

        def _y(i, j):
            return (((_index(i, j), 1),), +1j), (((_index(i, j), 0),), +1j)

        def _z(i, j):
            return (((_index(i, j), 1), (_index(i, j), 0)), +1), (((_index(i, j), 0), (_index(i, j), 1)), -1)

        hamiltonian = defaultdict(lambda: 0)
        # Express spin pauli matrix in hard core boson language.
        for i in range(args.m):
            for j in range(args.n):
                if True:
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
                    if args.hx != 0:
                        for k1, v1 in _x(i, j):
                            for k2, v2 in _x(i - 1, j):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.hx
                    if args.hy != 0:
                        for k1, v1 in _y(i, j):
                            for k2, v2 in _y(i - 1, j):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.hy
                    if args.hz != 0:
                        for k1, v1 in _z(i, j):
                            for k2, v2 in _z(i - 1, j):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.hz
                if j != 0:
                    if args.vx != 0:
                        for k1, v1 in _x(i, j):
                            for k2, v2 in _x(i, j - 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.vx
                    if args.vy != 0:
                        for k1, v1 in _y(i, j):
                            for k2, v2 in _y(i, j - 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.vy
                    if args.vz != 0:
                        for k1, v1 in _z(i, j):
                            for k2, v2 in _z(i, j - 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.vz
                if i != 0 and j != 0:
                    if args.dx != 0:
                        for k1, v1 in _x(i, j):
                            for k2, v2 in _x(i - 1, j - 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.dx
                    if args.dy != 0:
                        for k1, v1 in _y(i, j):
                            for k2, v2 in _y(i - 1, j - 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.dy
                    if args.dz != 0:
                        for k1, v1 in _z(i, j):
                            for k2, v2 in _z(i - 1, j - 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.dz
                if i != 0 and j != args.n - 1:
                    if args.ax != 0:
                        for k1, v1 in _x(i, j):
                            for k2, v2 in _x(i - 1, j + 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.ax
                    if args.ay != 0:
                        for k1, v1 in _y(i, j):
                            for k2, v2 in _y(i - 1, j + 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.ay
                    if args.az != 0:
                        for k1, v1 in _z(i, j):
                            for k2, v2 in _z(i - 1, j + 1):
                                hamiltonian[(*k1, *k2)] += v1 * v2 * args.az
        return hamiltonian

    def __init__(self, args: ModelConfig):
        self.args = args
        self.m = self.args.m
        self.n = self.args.n
        logging.info("creating Ising model with width = %d, height = %d", self.m, self.n)
        logging.info("elementwise coefficients: x = %.10f, y = %.10f, z = %.10f", self.args.x, self.args.y, self.args.z)
        logging.info("horizontal bond coefficients: x = %.10f, y = %.10f, z = %.10f", self.args.hx, self.args.hy, self.args.hz)
        logging.info("vertical bond coefficients: x = %.10f, y = %.10f, z = %.10f", self.args.vx, self.args.vy, self.args.vz)
        logging.info("diagonal bond coefficients: x = %.10f, y = %.10f, z = %.10f", self.args.dx, self.args.dy, self.args.dz)
        logging.info("antidiagonal bond coefficients: x = %.10f, y = %.10f, z = %.10f", self.args.ax, self.args.ay, self.args.az)

        logging.info("preparing hamiltonian on the lattice")
        hamiltonian_dict = self._prepare_hamiltonian(args)
        logging.info("the hamiltonian has been prepared")

        self.ref_energy = torch.nan

        logging.info("creating ising hamiltonian handle")
        self.hamiltonian = hamiltonian.Hamiltonian(hamiltonian_dict, kind="bose2")
        logging.info("hamiltonian handle has been created")

    def inside(self, configs_i):
        return self.hamiltonian.inside(configs_i)

    def outside(self, configs_i):
        return self.hamiltonian.outside(configs_i)


model_dict["ising"] = Model


@dataclasses.dataclass
class NaqsConfig:
    # The hidden widths of the network
    hidden: typing.Annotated[tuple[int, ...], tyro.conf.arg(aliases=["-w"])] = (512,)

    @classmethod
    def create(cls, model, input_args):
        logging.info("parsing args %a by network naqs", input_args)
        args = tyro.cli(NaqsConfig, args=input_args)
        logging.info("hidden: %a", args.hidden)

        network = naqs_m.WaveFunctionNormal(
            sites=model.m * model.n,
            physical_dim=2,
            is_complex=True,
            hidden_size=args.hidden,
            ordering=+1,
        ).double()

        return torch.jit.script(network)


Model.network_dict["naqs"] = NaqsConfig.create

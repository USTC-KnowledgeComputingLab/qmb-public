# This file declare transverse field ising model.

import typing
import logging
import dataclasses
import torch
import tyro
from . import _ising
from . import naqs as naqs_m


@dataclasses.dataclass
class ModelConfig:
    # The length of the ising chain
    length: typing.Annotated[int, tyro.conf.Positional]
    # The coefficient of X
    X: typing.Annotated[float, tyro.conf.arg(aliases=["-x"])] = 0
    # The coefficient of Y
    Y: typing.Annotated[float, tyro.conf.arg(aliases=["-y"])] = 0
    # The coefficient of Z
    Z: typing.Annotated[float, tyro.conf.arg(aliases=["-z"])] = 0
    # The coefficient of XX
    XX: typing.Annotated[float, tyro.conf.arg(aliases=["-X"])] = 0
    # The coefficient of YY
    YY: typing.Annotated[float, tyro.conf.arg(aliases=["-Y"])] = 0
    # The coefficient of ZZ
    ZZ: typing.Annotated[float, tyro.conf.arg(aliases=["-Z"])] = 0


@dataclasses.dataclass
class NaqsConfig:
    # The hidden widths of the network
    hidden: typing.Annotated[tuple[int, ...], tyro.conf.arg(aliases=["-w"])] = (512,)


class Model:

    @classmethod
    def preparse(cls, input_args):
        args = tyro.cli(ModelConfig, args=input_args)
        return f"Ising_L{args.length}_X{args.X}_Y{args.Y}_Z{args.Z}_XX{args.XX}_YY{args.YY}_ZZ{args.ZZ}"

    @classmethod
    def parse(cls, input_args):
        logging.info("parsing args %a by ising model", input_args)
        args = tyro.cli(ModelConfig, args=input_args)
        logging.info("length: %d, X: %.10f, Y: %.10f, Z: %.10f, XX: %.10f, YY: %.10f, ZZ: %.10f", args.length, args.X, args.Y, args.Z, args.XX, args.YY, args.ZZ)

        return cls(args.length, args.X, args.Y, args.Z, args.XX, args.YY, args.ZZ)

    def __init__(self, length, X, Y, Z, XX, YY, ZZ):
        self.length = length
        self.X = X
        self.Y = Y
        self.Z = Z
        self.XX = XX
        self.YY = YY
        self.ZZ = ZZ
        logging.info("creating ising model with length = %d, X = %.10f, Y = %.10f, Z = %.10f, XX = %.10f, YY = %.10f, ZZ = %.10f", self.length, self.X, self.Y, self.Z, self.XX, self.YY, self.ZZ)

        self.ref_energy = torch.nan

        logging.info("creating ising hamiltonian handle")
        self.hamiltonian = _ising.Hamiltonian(self.X, self.Y, self.Z, self.XX, self.YY, self.ZZ)
        logging.info("hamiltonian handle has been created")

    def inside(self, configs):
        return self.hamiltonian.inside(configs)

    def outside(self, configs):
        return self.hamiltonian.outside(configs)

    def naqs(self, input_args):
        logging.info("parsing args %a by network naqs", input_args)
        args = tyro.cli(NaqsConfig, args=input_args)
        logging.info("hidden: %a", args.hidden)

        network = naqs_m.WaveFunctionNormal(
            sites=self.length,
            physical_dim=2,
            is_complex=True,
            hidden_size=args.hidden,
            ordering=+1,
        ).double()

        return torch.jit.script(network)

# This file declare transverse field ising model.

import argparse
import logging
import torch
from . import _ising
from . import naqs as naqs_m


class Model:

    @classmethod
    def preparse(cls, input_args):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("length", type=int, help="length of the ising chain")
        parser.add_argument("-x", dest="X", type=float, default=0, help="coefficient of X")
        parser.add_argument("-y", dest="Y", type=float, default=0, help="coefficient of Y")
        parser.add_argument("-z", dest="Z", type=float, default=0, help="coefficient of Z")
        parser.add_argument("-X", dest="XX", type=float, default=0, help="coefficient of XX")
        parser.add_argument("-Y", dest="YY", type=float, default=0, help="coefficient of YY")
        parser.add_argument("-Z", dest="ZZ", type=float, default=0, help="coefficient of ZZ")
        args = parser.parse_args(input_args)

        return f"Ising_L{args.length}_X{args.X}_Y{args.Y}_Z{args.Z}_XX{args.XX}_YY{args.YY}_ZZ{args.ZZ}"

    @classmethod
    def parse(cls, input_args):
        logging.info("parsing args %a by ising model", input_args)
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("length", type=int, help="length of the ising chain")
        parser.add_argument("-x", dest="X", type=float, default=0, help="coefficient of X")
        parser.add_argument("-y", dest="Y", type=float, default=0, help="coefficient of Y")
        parser.add_argument("-z", dest="Z", type=float, default=0, help="coefficient of Z")
        parser.add_argument("-X", dest="XX", type=float, default=0, help="coefficient of XX")
        parser.add_argument("-Y", dest="YY", type=float, default=0, help="coefficient of YY")
        parser.add_argument("-Z", dest="ZZ", type=float, default=0, help="coefficient of ZZ")
        args = parser.parse_args(input_args)
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
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("-w", "--hidden-width", dest="hidden", type=int, default=[512], nargs="+", help="hidden width of the network")
        args = parser.parse_args(input_args)
        logging.info("hidden: %a", args.hidden)

        network = naqs_m.WaveFunctionNormal(
            sites=self.length,
            physical_dim=2,
            is_complex=True,
            hidden_size=args.hidden,
            ordering=+1,
        ).double()

        return torch.jit.script(network)

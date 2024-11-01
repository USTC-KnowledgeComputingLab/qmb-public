# This file implements interface to openfermion model.

import os
import typing
import logging
import pathlib
import dataclasses
import torch
import torch.utils.cpp_extension
import tyro
import openfermion
from . import naqs as naqs_m
from . import attention as attention_m


def get_openfermion_extension():
    return torch.utils.cpp_extension.load(name="_openfermion", sources=f"{os.path.dirname(__file__)}/_openfermion.cu")


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

        logging.info("compiling torch extension")
        _openfermion = get_openfermion_extension()
        logging.info("torch extension compiled")

        logging.info("converting openfermion handle to hamiltonian handle")
        self.hamiltonian = _openfermion.Hamiltonian(openfermion.transforms.get_fermion_operator(self.openfermion.get_molecular_hamiltonian()).terms)
        logging.info("hamiltonian handle has been created")

    def inside(self, configs_i):
        configs_i = configs_i.cuda().to(dtype=torch.bool)
        # Parameters
        # configs_i : bool[batch_size, n_qubits]
        # Returns
        # index_i_and_j : int64[..., 2]
        # coefs : complex128[...]

        batch_size = configs_i.shape[0]
        valid_index_i, valid_configs_j, valid_coefs = self.hamiltonian.relative(configs_i)
        # configs_i : bool[batch_size, n_qubits]
        # valid_configs_j : bool[valid_size, n_qubits]
        # valid_index_i : int64[valid_size]
        # valid_coefs : float64[valid_size, 2]

        # map from valid to pool first, and then map pool to target.

        configs_i_and_j = torch.cat([configs_i, valid_configs_j], dim=0)
        # pool : bool[pool_size, n_qubits]
        # v_to_p : int64[batch_size + valid_size]
        pool, v_to_p = torch.unique(configs_i_and_j, dim=0, sorted=False, return_inverse=True, return_counts=False)
        pool_size = pool.shape[0]

        # p_to_v : int64[pool_size]
        p_to_t = torch.full([pool_size], -1, dtype=torch.int64, device=configs_i.device)
        p_to_t[v_to_p[:batch_size]] = torch.arange(batch_size, device=configs_i.device)

        # usable data
        # valid_index_j : int64[valid_size] -> -1 or 0...batch_size-1
        # it is v_to_t in fact
        valid_index_j = p_to_t[v_to_p[batch_size:]]

        # usable : int64[]
        usable = valid_index_j >= 0

        index_i_target = valid_index_i[usable]
        index_j_target = valid_index_j[usable]
        coefs_target = valid_coefs[usable]

        return torch.stack([index_i_target, index_j_target], dim=1), torch.view_as_complex(coefs_target)

    def outside(self, configs_i):
        configs_i = configs_i.cuda().to(dtype=torch.bool)
        # Parameters
        # configs_i : bool[batch_size, n_qubits]
        # Returns
        # index_i_and_j : int64[..., 2]
        # coefs : complex128[...]

        batch_size = configs_i.shape[0]
        valid_index_i, valid_configs_j, valid_coefs = self.hamiltonian.relative(configs_i)
        # configs_i : bool[batch_size, n_qubits]
        # valid_configs_j : bool[valid_size, n_qubits]
        # valid_index_i : int64[valid_size]
        # valid_coefs : float64[valid_size, 2]

        # map from valid to pool first, and then map pool to target.

        configs_i_and_j = torch.cat([configs_i, valid_configs_j], dim=0)
        # pool : bool[pool_size, n_qubits]
        # v_to_p : int64[batch_size + valid_size]
        pool, v_to_p = torch.unique(configs_i_and_j, dim=0, sorted=False, return_inverse=True, return_counts=False)
        pool_size = pool.shape[0]

        # p_to_v : int64[pool_size]
        p_to_t = torch.full([pool_size], -1, dtype=torch.int64, device=configs_i.device)
        p_to_t[v_to_p[:batch_size]] = torch.arange(batch_size, device=configs_i.device)
        p_to_t[p_to_t == -1] = torch.arange(batch_size, pool_size, device=configs_i.device)

        # usable data
        # valid_index_j : int64[valid_size] -> -1 or 0...batch_size-1
        # it is v_to_t in fact
        valid_index_j = p_to_t[v_to_p[batch_size:]]

        index_i_target = valid_index_i
        index_j_target = valid_index_j
        coefs_target = valid_coefs

        configs_target = torch.empty_like(pool)
        configs_target[p_to_t] = pool

        return torch.stack([index_i_target, index_j_target], dim=1), torch.view_as_complex(coefs_target), configs_target.to(dtype=torch.int64)

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

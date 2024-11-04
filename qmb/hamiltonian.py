import os
import typing
import torch
import torch.utils.cpp_extension

extension: object = None


def get_extension() -> object:
    global extension
    if extension is None:
        extension = torch.utils.cpp_extension.load(name="_hamiltonian", sources=f"{os.path.dirname(__file__)}/_hamiltonian.cu")
    return extension


@typing.runtime_checkable
class RelativeProto(typing.Protocol):

    def relative(self, configs_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...


class Hamiltonian:

    def __init__(self, hamiltonian: dict[tuple[tuple[int, int], ...], complex], *, kind: typing.Literal["fermi", "bose2"]) -> None:
        self.hamiltonian: RelativeProto = getattr(get_extension(), kind)(hamiltonian)

    def relative(
        self,
        configs_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # this function is equivalent to return self.hamiltonian.relative(configs_i)
        # but split configs_i into pieces to avoid out of memory
        batch_size: int = configs_i.shape[0]
        index_i_pool: list[torch.Tensor] = []
        configs_j_pool: list[torch.Tensor] = []
        coefs_pool: list[torch.Tensor] = []
        for i in range(batch_size):
            index_i: torch.Tensor
            configs_j: torch.Tensor
            coefs: torch.Tensor
            index_i, configs_j, coefs = self.hamiltonian.relative(configs_i[i:i + 1])
            index_i_pool.append(index_i + i)
            configs_j_pool.append(configs_j)
            coefs_pool.append(coefs)
        return torch.cat(index_i_pool, dim=0), torch.cat(configs_j_pool, dim=0), torch.cat(coefs_pool, dim=0)

    def inside(self, configs_i_int64: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        configs_i: torch.Tensor = configs_i_int64.cuda().to(dtype=torch.int8)
        # Parameters
        # configs_i : bool[batch_size, n_qubits]
        # Returns
        # index_i_and_j : int64[..., 2]
        # coefs : complex128[...]

        batch_size: int = configs_i.shape[0]
        valid_index_i: torch.Tensor
        valid_configs_j: torch.Tensor
        valid_coefs: torch.Tensor
        valid_index_i, valid_configs_j, valid_coefs = self.relative(configs_i)
        # configs_i : bool[batch_size, n_qubits]
        # valid_configs_j : bool[valid_size, n_qubits]
        # valid_index_i : int64[valid_size]
        # valid_coefs : float64[valid_size, 2]

        # map from valid to pool first, and then map pool to target.

        configs_i_and_j: torch.Tensor = torch.cat([configs_i, valid_configs_j], dim=0)
        # pool : bool[pool_size, n_qubits]
        # v_to_p : int64[batch_size + valid_size]
        pool: torch.Tensor
        v_to_p: torch.Tensor
        pool, v_to_p = torch.unique(configs_i_and_j, dim=0, sorted=False, return_inverse=True, return_counts=False)
        pool_size: int = pool.shape[0]

        # p_to_v : int64[pool_size]
        p_to_t: torch.Tensor = torch.full([pool_size], -1, dtype=torch.int64, device=configs_i.device)
        p_to_t[v_to_p[:batch_size]] = torch.arange(batch_size, device=configs_i.device)

        # usable data
        # valid_index_j : int64[valid_size] -> -1 or 0...batch_size-1
        # it is v_to_t in fact
        valid_index_j: torch.Tensor = p_to_t[v_to_p[batch_size:]]

        # usable : int64[]
        usable: torch.Tensor = valid_index_j >= 0

        index_i_target: torch.Tensor = valid_index_i[usable]
        index_j_target: torch.Tensor = valid_index_j[usable]
        coefs_target: torch.Tensor = valid_coefs[usable]

        return torch.stack([index_i_target, index_j_target], dim=1), torch.view_as_complex(coefs_target)

    def outside(self, configs_i_int64: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        configs_i: torch.Tensor = configs_i_int64.cuda().to(dtype=torch.int8)
        # Parameters
        # configs_i : bool[batch_size, n_qubits]
        # Returns
        # index_i_and_j : int64[..., 2]
        # coefs : complex128[...]

        batch_size: int = configs_i.shape[0]
        valid_index_i: torch.Tensor
        valid_configs_j: torch.Tensor
        valid_coefs: torch.Tensor
        valid_index_i, valid_configs_j, valid_coefs = self.relative(configs_i)
        # configs_i : bool[batch_size, n_qubits]
        # valid_configs_j : bool[valid_size, n_qubits]
        # valid_index_i : int64[valid_size]
        # valid_coefs : float64[valid_size, 2]

        # map from valid to pool first, and then map pool to target.

        configs_i_and_j: torch.Tensor = torch.cat([configs_i, valid_configs_j], dim=0)
        # pool : bool[pool_size, n_qubits]
        # v_to_p : int64[batch_size + valid_size]
        pool: torch.Tensor
        v_to_p: torch.Tensor
        pool, v_to_p = torch.unique(configs_i_and_j, dim=0, sorted=False, return_inverse=True, return_counts=False)
        pool_size: int = pool.shape[0]

        # p_to_v : int64[pool_size]
        p_to_t: torch.Tensor = torch.full([pool_size], -1, dtype=torch.int64, device=configs_i.device)
        p_to_t[v_to_p[:batch_size]] = torch.arange(batch_size, device=configs_i.device)
        p_to_t[p_to_t == -1] = torch.arange(batch_size, pool_size, device=configs_i.device)

        # usable data
        # valid_index_j : int64[valid_size] -> -1 or 0...batch_size-1
        # it is v_to_t in fact
        valid_index_j: torch.Tensor = p_to_t[v_to_p[batch_size:]]

        index_i_target: torch.Tensor = valid_index_i
        index_j_target: torch.Tensor = valid_index_j
        coefs_target: torch.Tensor = valid_coefs

        configs_target: torch.Tensor = torch.empty_like(pool)
        configs_target[p_to_t] = pool

        return torch.stack([index_i_target, index_j_target], dim=1), torch.view_as_complex(coefs_target), configs_target.to(dtype=torch.int64)

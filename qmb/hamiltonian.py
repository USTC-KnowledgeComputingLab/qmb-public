"""
This file contains the Hamiltonian class, which is used to store the Hamiltonian and process iteration over each term in the Hamiltonian for given configurations.
"""

import os
import typing
import torch
import torch.utils.cpp_extension


@torch.jit.script
def _merge_inside(
    configs_i: torch.Tensor,
    result: tuple[
        torch.Tensor,  # index_i
        torch.Tensor,  # index_j
        torch.Tensor,  # coefs
    ] | None,
    batch: tuple[
        torch.Tensor,  # index_i
        torch.Tensor,  # configs_j
        torch.Tensor,  # coefs
    ],
) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
]:
    # pylint: disable=too-many-locals
    device = configs_i.device
    if result is not None:
        index_i, index_j, coefs = result
    else:
        index_i = torch.empty([0], dtype=torch.int64, device=device)
        index_j = torch.empty([0], dtype=torch.int64, device=device)
        coefs = torch.empty([0, 2], dtype=torch.float64, device=device)
    batch_index_i, batch_configs_j, batch_coefs = batch

    pool, both_to_pool = torch.unique(torch.cat([configs_i, batch_configs_j], dim=0), dim=0, sorted=False, return_inverse=True, return_counts=False)

    src_size = configs_i.size(0)
    pool_size = pool.size(0)

    src_to_pool = both_to_pool[:src_size]
    pool_to_src = torch.full([pool_size], -1, dtype=torch.int64, device=device)
    pool_to_src[src_to_pool] = torch.arange(src_size, device=device)

    dst_to_pool = both_to_pool[src_size:]
    dst_to_src = pool_to_src[dst_to_pool]

    usable = dst_to_src != -1

    usable_index_i = batch_index_i[usable]
    usable_index_j = dst_to_src[usable]
    usable_coefs = batch_coefs[usable]

    return (
        torch.cat([index_i, usable_index_i], dim=0),
        torch.cat([index_j, usable_index_j], dim=0),
        torch.cat([coefs, usable_coefs], dim=0),
    )


@torch.jit.script
def _merge_outside(
    configs_i: torch.Tensor,
    result: tuple[
        torch.Tensor,  # index_i
        torch.Tensor,  # index_j
        torch.Tensor,  # coefs
        torch.Tensor,  # configs_j
    ] | None,
    batch: tuple[
        torch.Tensor,  # index_i
        torch.Tensor,  # configs_j
        torch.Tensor,  # coefs
    ],
) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
]:
    # pylint: disable=too-many-locals
    device = configs_i.device
    if result is not None:
        index_i, index_j, coefs, configs_j = result
    else:
        index_i = torch.empty([0], dtype=torch.int64, device=device)
        index_j = torch.empty([0], dtype=torch.int64, device=device)
        coefs = torch.empty([0, 2], dtype=torch.float64, device=device)
        configs_j = configs_i
    batch_index_i, batch_configs_j, batch_coefs = batch
    src_size = configs_j.shape[0]

    pool, both_to_pool = torch.unique(torch.cat([configs_j, batch_configs_j], dim=0), dim=0, sorted=False, return_inverse=True, return_counts=False)
    result_index_j = torch.cat([both_to_pool[index_j], both_to_pool[src_size:]], dim=0)
    del both_to_pool  # Memory is crucial here, release this tensor immediately to free up resources.

    return (
        torch.cat([index_i, batch_index_i], dim=0),
        result_index_j,
        torch.cat([coefs, batch_coefs], dim=0),
        pool,
    )


class Hamiltonian:
    """
    The Hamiltonian type, which stores the Hamiltonian and processes iteration over each term in the Hamiltonian for given configurations.
    """

    _extension: object = None

    @classmethod
    def _get_extension(cls) -> object:
        if cls._extension is None:
            folder = os.path.dirname(__file__)
            cls._extension = torch.utils.cpp_extension.load(
                name="_hamiltonian",
                sources=[
                    f"{folder}/_hamiltonian.cpp",
                    f"{folder}/_hamiltonian_cuda.cu",
                ],
            )
        return cls._extension

    def __init__(self, hamiltonian: dict[tuple[tuple[int, int], ...], complex], *, kind: str) -> None:
        self.site: torch.Tensor
        self.kind: torch.Tensor
        self.coef: torch.Tensor
        self.site, self.kind, self.coef = getattr(self._get_extension(), "prepare")(hamiltonian)
        self._relative_impl: typing.Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        self._relative_impl = getattr(torch.ops._hamiltonian, kind)

    def _prepare_data(self, device: torch.device) -> None:
        self.site = self.site.to(device=device).contiguous()
        self.kind = self.kind.to(device=device).contiguous()
        self.coef = self.coef.to(device=device).contiguous()

    def _relative_kernel(
        self,
        configs_i: torch.Tensor,
        *,
        term_group_size: int,
        batch_group_size: int,
    ) -> typing.Iterable[tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
    ]]:
        batch_size = configs_i.size(0)
        term_number = self.site.size(0)
        if batch_group_size == -1:
            batch_group_size = batch_size
        if term_group_size == -1:
            term_group_size = term_number

        for i in range(0, term_number, term_group_size):
            for j in range(0, batch_size, batch_group_size):
                index_i, configs_j, coefs = self._relative_impl(
                    configs_i[j:j + batch_group_size],
                    self.site[i:i + term_group_size],
                    self.kind[i:i + term_group_size],
                    self.coef[i:i + term_group_size],
                )
                yield index_i + j, configs_j, coefs

    def _relative_group(
        self,
        configs_i: torch.Tensor,
        *,
        term_group_size: int = 1024,
        batch_group_size: int = -1,
        group_size: int = 1 << 30,
    ) -> typing.Iterable[tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
    ]]:
        index_i_pool = []
        configs_j_pool = []
        coefs_pool = []
        total_size = 0
        for index_i, configs_j, coefs in self._relative_kernel(configs_i, term_group_size=term_group_size, batch_group_size=batch_group_size):
            index_i_pool.append(index_i)
            configs_j_pool.append(configs_j)
            coefs_pool.append(coefs)
            total_size += index_i.nelement() * index_i.element_size() + configs_j.nelement() * configs_j.element_size() + coefs.nelement() * coefs.element_size()
            if total_size >= group_size:
                yield torch.cat(index_i_pool, dim=0), torch.cat(configs_j_pool, dim=0), torch.cat(coefs_pool, dim=0)
                index_i_pool.clear()
                configs_j_pool.clear()
                coefs_pool.clear()
                total_size = 0
        yield torch.cat(index_i_pool, dim=0), torch.cat(configs_j_pool, dim=0), torch.cat(coefs_pool, dim=0)

    def inside(
        self,
        configs_i: torch.Tensor,
    ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
    ]:
        """
        Applies the Hamiltonian to the given configuration and obtains the resulting sparse Hamiltonian matrix block within the configuration subspace.
        This function only considers the terms that are within the configuration subspace.

        Parameters
        ----------
        configs_i : torch.Tensor
            A tensor of shape [batch_size, n_qubits] representing the input configurations.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing two tensors:
                - index_i: A tensor of shape [...] representing the indices i for the non-zero Hamiltonian terms.
                - index_j: A tensor of shape [...] representing the indices j for the non-zero Hamiltonian terms.
                - coefs: A tensor of shape [...] representing the corresponding complex coefficients for the indices.
        """
        device: torch.device = configs_i.device
        self._prepare_data(device)
        # Parameters
        # configs_i : bool[src_size, n_qubits]
        # Returns
        # index_i_and_j : int64[..., 2]
        # coefs : complex128[...]

        result: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        for batch in self._relative_group(configs_i):
            result = _merge_inside(configs_i, result, batch)
        assert result is not None
        index_i, index_j, coefs = result
        return index_i, index_j, torch.view_as_complex(coefs)

    def outside(
        self,
        configs_i: torch.Tensor,
    ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
    ]:
        """
        Applies the Hamiltonian to the given configuration and obtains the resulting sparse Hamiltonian matrix block within the configuration subspace.
        This function considers both the inside and outside configurations, ensuring that the input configurations are the first `batch_size` configurations in the result.

        Parameters
        ----------
        configs_i : torch.Tensor
            A tensor of shape [batch_size, n_qubits] representing the input configurations.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing three tensors:
                - index_i: A tensor of shape [...] representing the indices i for the non-zero Hamiltonian terms.
                - index_j: A tensor of shape [...] representing the indices j for the non-zero Hamiltonian terms.
                - coefs: A tensor of shape [...] representing the corresponding complex coefficients for the indices.
                - configs_target: A tensor of shape [pool_size, n_qubits] representing the target configurations after processing.
                  The first `batch_size` configurations are guaranteed to be identical to the input `configs_i`.
        """
        # pylint: disable=too-many-locals
        device: torch.device = configs_i.device
        self._prepare_data(device)
        # Parameters
        # configs_i : bool[src_size, n_qubits]
        # Returns
        # index_i : int64[...]
        # index_j : int64[...]
        # coefs : complex128[...]
        # configs_j : bool[dst_size, n_qubits]

        result: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        for batch in self._relative_group(configs_i):
            result = _merge_outside(configs_i, result, batch)
        assert result is not None
        index_i, index_j, coefs, configs_j = result

        pool, both_to_pool = torch.unique(torch.cat([configs_i, configs_j], dim=0), dim=0, sorted=False, return_inverse=True, return_counts=False)

        del configs_j

        src_size = configs_i.size(0)
        pool_size = pool.size(0)

        pool_to_target = torch.full([pool_size], -1, dtype=torch.int64, device=device)
        pool_to_target[both_to_pool[:src_size]] = torch.arange(src_size, device=device)
        pool_to_target[pool_to_target == -1] = torch.arange(src_size, pool_size, device=device)

        target = torch.empty_like(pool)
        target[pool_to_target] = pool

        del pool

        target_index_j = pool_to_target[both_to_pool[src_size:]][index_j]

        return index_i, target_index_j, torch.view_as_complex(coefs), target

"""
This file contains the Hamiltonian class, which is used to store the Hamiltonian and process iteration over each term in the Hamiltonian for given configurations.
"""

import os
import gc
import functools
import typing
import torch
import torch.utils.cpp_extension

_Raw = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_Inside = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_Outside = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
_Sparse = tuple[torch.Tensor, torch.Tensor]

_P = typing.ParamSpec("_P")
_R = typing.TypeVar("_R")


def _collect_and_empty_cache(func: typing.Callable[_P, _R]) -> typing.Callable[_P, _R]:

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        gc.collect()
        torch.cuda.empty_cache()
        return func(*args, **kwargs)

    return wrapper


class Hamiltonian:
    """
    The Hamiltonian type, which stores the Hamiltonian and processes iteration over each term in the Hamiltonian for given configurations.
    """

    _hamiltonian_module: object = None
    _collection_module: dict[int, object] = {}

    @classmethod
    def _get_hamiltonian_module(cls) -> object:
        if cls._hamiltonian_module is None:
            folder = os.path.dirname(__file__)
            cls._hamiltonian_module = torch.utils.cpp_extension.load(
                name="qmb_hamiltonian",
                sources=[
                    f"{folder}/_hamiltonian.cpp",
                    f"{folder}/_hamiltonian_cuda.cu",
                ],
            )
        return cls._hamiltonian_module

    @classmethod
    def _get_collection_module(cls, n_qubytes: int = 0) -> object:
        if n_qubytes not in cls._collection_module:
            folder = os.path.dirname(__file__)
            cls._collection_module[n_qubytes] = torch.utils.cpp_extension.load(
                name="qmb_collection" if n_qubytes == 0 else f"qmb_collection_{n_qubytes}",
                sources=[
                    f"{folder}/_collection.cpp",
                    f"{folder}/_collection_cuda.cu",
                ],
                is_python_module=n_qubytes == 0,
                extra_cflags=[f"-DNQUBYTES={n_qubytes}"],
                extra_cuda_cflags=[f"-DNQUBYTES={n_qubytes}"],
            )
        return cls._collection_module[n_qubytes]

    @classmethod
    def _load_hamiltonian(cls) -> object:
        cls._get_hamiltonian_module()
        return torch.ops.qmb_hamiltonian

    @classmethod
    def _load_collection(cls, n_qubytes: int) -> object:
        cls._get_collection_module(n_qubytes=n_qubytes)
        return getattr(torch.ops, f"qmb_collection_{n_qubytes}")

    @classmethod
    def _prepare(cls, hamiltonian: dict[tuple[tuple[int, int], ...], complex]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return getattr(cls._get_hamiltonian_module(), "prepare")(hamiltonian)

    @classmethod
    def _merge_raw(cls, raws: list[_Raw]) -> _Raw:
        return getattr(cls._get_collection_module(), "merge_raw")(raws)

    @classmethod
    def _raw_to_inside(cls, raw: _Raw, configs_i: torch.Tensor) -> _Inside:
        return getattr(cls._get_collection_module(), "raw_to_inside")(raw, configs_i)

    @classmethod
    def _merge_inside(cls, insides: list[_Inside]) -> _Inside:
        return getattr(cls._get_collection_module(), "merge_inside")(insides)

    @classmethod
    def _raw_to_outside(cls, raw: _Raw, configs_i: torch.Tensor) -> _Outside:
        return getattr(cls._get_collection_module(), "raw_to_outside")(raw, configs_i)

    @classmethod
    def _merge_outside(cls, outsides: list[_Outside], configs_i: torch.Tensor | None) -> _Outside:
        return getattr(cls._get_collection_module(), "merge_outside")(outsides, configs_i)

    @classmethod
    def _raw_apply_outside(cls, raw: _Raw, psi_i: torch.Tensor, configs_i: torch.Tensor, squared: bool) -> _Sparse:
        return getattr(cls._get_collection_module(), "raw_apply_outside")(raw, psi_i, configs_i, squared)

    @classmethod
    def _merge_apply_outside(cls, sparses: list[_Sparse], configs_i: torch.Tensor | None) -> _Sparse:
        return getattr(cls._get_collection_module(), "merge_apply_outside")(sparses, configs_i)

    def __init__(self, hamiltonian: dict[tuple[tuple[int, int], ...], complex] | tuple[torch.Tensor, torch.Tensor, torch.Tensor], *, kind: str) -> None:
        self.site: torch.Tensor
        self.kind: torch.Tensor
        self.coef: torch.Tensor
        if isinstance(hamiltonian, dict):
            self.site, self.kind, self.coef = self._prepare(hamiltonian)
        else:
            self.site, self.kind, self.coef = hamiltonian
        self._relative_impl: typing.Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        self._relative_impl = getattr(self._load_hamiltonian(), kind)

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
    ) -> typing.Iterable[_Raw]:
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
    ) -> typing.Iterable[_Raw]:
        pool = []
        total_size = 0
        for batch in self._relative_kernel(configs_i, term_group_size=term_group_size, batch_group_size=batch_group_size):
            pool.append(batch)
            total_size += sum(tensor.nelement() * tensor.element_size() for tensor in batch)
            if total_size >= group_size:
                yield self._merge_raw(pool)
                pool.clear()
                total_size = 0
        if pool:
            yield self._merge_raw(pool)

    def inside(
        self,
        configs_i: torch.Tensor,
    ) -> _Inside:
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

        result: list[_Inside] = []
        for batch in (self._raw_to_inside(raw, configs_i) for raw in self._relative_group(configs_i)):
            if len(result) >= 2:
                result = [self._merge_inside(result)]
            result.append(batch)
        index_i, index_j, coefs = self._merge_inside(result)
        return index_i, index_j, torch.view_as_complex(coefs)

    def outside(
        self,
        configs_i: torch.Tensor,
    ) -> _Outside:
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

        result: list[_Outside] = []
        for batch in (self._raw_to_outside(raw, configs_i) for raw in self._relative_group(configs_i)):
            if len(result) >= 2:
                result = [self._merge_outside(result, None)]
            result.append(batch)
        index_i, index_j, coefs, configs_j = self._merge_outside(result, configs_i)
        return index_i, index_j, torch.view_as_complex(coefs), configs_j

    def apply_outside(self, psi_i: torch.Tensor, configs_i: torch.Tensor, squared: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the outside Hamiltonian to the given vector.

        This method is equivalent to the following code:
        ```
        indices_i, indices_j, values, configs_j = self.outside(configs_i)
        hamiltonian = torch.sparse_coo_tensor(torch.stack([indices_i, indices_j], dim=0), values, [count_i, count_j], dtype=torch.complex128)
        psi_j = psi_i.conj() @ hamiltonian
        return psi_j, configs_j
        ```
        """
        return self._apply_outside_kernel(psi_i, configs_i, squared)

    def _apply_outside_kernel(self, psi_i: torch.Tensor, configs_i: torch.Tensor, squared: bool) -> tuple[torch.Tensor, torch.Tensor]:
        device: torch.device = configs_i.device
        self._prepare_data(device)

        module = self._load_collection(configs_i.size(1))
        op_sort_ = _collect_and_empty_cache(getattr(module, "sort_"))
        op_merge = _collect_and_empty_cache(getattr(module, "merge"))
        op_reduce = _collect_and_empty_cache(getattr(module, "reduce"))
        op_ensure_ = _collect_and_empty_cache(getattr(module, "ensure_"))

        configs_j: torch.Tensor | None = None
        psi_j: torch.Tensor | None = None
        for batch_psi_j, batch_configs_j in (self._raw_apply_outside(raw, torch.view_as_real(psi_i), configs_i, squared) for raw in self._relative_group(configs_i)):
            batch_configs_j, batch_psi_j = op_sort_(batch_configs_j, batch_psi_j)
            batch_configs_j, batch_psi_j = op_reduce(batch_configs_j, batch_psi_j)
            if configs_j is None or psi_j is None:
                configs_j, psi_j = batch_configs_j, batch_psi_j
            else:
                configs_j, psi_j = op_merge(configs_j, psi_j, batch_configs_j, batch_psi_j)
                configs_j, psi_j = op_reduce(configs_j, psi_j)
        assert configs_j is not None and psi_j is not None
        configs_j = torch.cat([configs_i, configs_j])
        psi_j = torch.cat([torch.zeros([configs_i.size(0), psi_j.size(1)], dtype=psi_j.dtype, device=psi_j.device), psi_j])
        configs_j, psi_j = op_ensure_(configs_j, psi_j, configs_i.size(0))
        assert configs_j is not None and psi_j is not None
        if squared:
            psi_j = psi_j[:, 0]
        else:
            psi_j = torch.view_as_complex(psi_j)
        return psi_j, configs_j

    def _apply_outside_libtorch(self, psi_i: torch.Tensor, configs_i: torch.Tensor, squared: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Libtorch implementation of `apply_outside`.
        """
        device: torch.device = configs_i.device
        self._prepare_data(device)

        result: list[_Sparse] = []
        for batch in (self._raw_apply_outside(raw, torch.view_as_real(psi_i), configs_i, squared) for raw in self._relative_group(configs_i)):
            if len(result) >= 2:
                result = [self._merge_apply_outside(result, None)]
            result.append(batch)
        psi_j, configs_j = self._merge_apply_outside(result, configs_i)
        if squared:
            psi_j = psi_j[:, 0]
        else:
            psi_j = torch.view_as_complex(psi_j)
        return psi_j, configs_j

    def _apply_outside_ref(self, psi_i: torch.Tensor, configs_i: torch.Tensor, squared: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The reference implementation of `apply_outside`.
        """
        indices_i, indices_j, values, configs_j = self.outside(configs_i)
        count_i = configs_i.size(0)
        count_j = configs_j.size(0)
        hamiltonian = torch.sparse_coo_tensor(torch.stack([indices_i, indices_j], dim=0), values, [count_i, count_j], dtype=torch.complex128)
        if squared:
            psi_j = (psi_i.conj() * psi_i).abs() @ (hamiltonian.conj() * hamiltonian).abs()
        else:
            psi_j = psi_i.conj() @ hamiltonian
        return psi_j, configs_j

"""
This file contains the Hamiltonian class, which is used to store the Hamiltonian and process iteration over each term in the Hamiltonian for given configurations.
"""

import os
import typing
import torch
import torch.utils.cpp_extension


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

    def __init__(self, hamiltonian: dict[tuple[tuple[int, int], ...], complex], *, kind: typing.Literal["fermi", "bose2"]) -> None:
        self.site: torch.Tensor
        self.kind: torch.Tensor
        self.coef: torch.Tensor
        self.site, self.kind, self.coef = getattr(self._get_extension(), "prepare")(hamiltonian)
        self._relative_impl: typing.Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        self._relative_impl = getattr(torch.ops._hamiltonian, kind)

    def _relative(
        self,
        configs_i: torch.Tensor,
        early_drop: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device: torch.device = configs_i.device
        self.site = self.site.to(device=device)
        self.kind = self.kind.to(device=device)
        self.coef = self.coef.to(device=device)
        assert configs_i.is_contiguous() and self.site.is_contiguous() and self.kind.is_contiguous() and self.coef.is_contiguous()
        return self._relative_impl(configs_i, self.site, self.kind, self.coef, early_drop)

    def inside(self, configs_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the Hamiltonian to the given configuration and obtains the resulting sparse Hamiltonian matrix block within the configuration subspace.
        This function only considers the terms that are within the configuration subspace.

        Parameters
        ----------
        configs_i : torch.Tensor
            A tensor of shape [batch_size, n_qubits] representing the input configurations.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing two tensors:
                - index_i_and_j: A tensor of shape [..., 2] representing the indices (i, j) for the non-zero Hamiltonian terms.
                - coefs: A tensor of shape [...] representing the corresponding complex coefficients for the indices.

        The function performs the following steps:
        1. Converts the input configurations to dtype `torch.int8`.
        2. Calls the `_relative` method to obtain valid indices (`valid_index_i`), valid configurations (`valid_configs_j`), and valid coefficients (`valid_coefs`).
        3. Concatenates the input configurations and valid configurations to form a pool of configurations.
        4. Uses `torch.unique` to find the unique configurations in the pool and their corresponding indices.
        5. Maps the valid configurations to the target configurations, updating the indices accordingly.
        6. Filters out the usable configurations and their corresponding coefficients.
        7. Returns the valid indices and coefficients.
        """
        # pylint: disable=too-many-locals
        device: torch.device = configs_i.device
        # Parameters
        # configs_i : bool[batch_size, n_qubits]
        # Returns
        # index_i_and_j : int64[..., 2]
        # coefs : complex128[...]

        batch_size: int = configs_i.shape[0]
        valid_index_i: torch.Tensor
        valid_configs_j: torch.Tensor
        valid_coefs: torch.Tensor
        valid_index_i, valid_configs_j, valid_coefs = self._relative(configs_i, early_drop=True)
        # configs_i : bool[batch_size, n_qubits]
        # valid_configs_j : bool[valid_size, n_qubits]
        # valid_index_i : int64[valid_size]
        # valid_coefs : float64[valid_size, 2]

        # map from valid to pool first, and then map pool to target.

        configs_i_and_j: torch.Tensor = torch.cat([configs_i, valid_configs_j], dim=0)
        del configs_i
        del valid_configs_j
        # pool : bool[pool_size, n_qubits]
        # both_to_pool : int64[batch_size + valid_size]
        pool: torch.Tensor
        both_to_pool: torch.Tensor
        pool, both_to_pool = torch.unique(configs_i_and_j, dim=0, sorted=False, return_inverse=True, return_counts=False)
        pool_size: int = pool.shape[0]
        del configs_i_and_j
        del pool

        # pool_to_source : int64[pool_size]
        pool_to_source: torch.Tensor = torch.full([pool_size], -1, dtype=torch.int64, device=device)
        source_to_pool: torch.Tensor = both_to_pool[:batch_size]
        pool_to_source[source_to_pool] = torch.arange(batch_size, device=device)
        del source_to_pool

        # destination_to_source : int64[valid_size] -> -1 or 0...batch_size-1
        destination_to_pool: torch.Tensor = both_to_pool[batch_size:]
        del both_to_pool
        destination_to_source: torch.Tensor = pool_to_source[destination_to_pool]
        del pool_to_source
        del destination_to_pool

        # usable : int64[]
        usable: torch.Tensor = destination_to_source >= 0

        return torch.stack([valid_index_i[usable], destination_to_source[usable]], dim=1), torch.view_as_complex(valid_coefs[usable])

    def outside(self, configs_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies the Hamiltonian to the given configuration and obtains the resulting sparse Hamiltonian matrix block within the configuration subspace.
        This function considers both the inside and outside configurations, ensuring that the input configurations are the first `batch_size` configurations in the result.

        Parameters
        ----------
        configs_i : torch.Tensor
            A tensor of shape [batch_size, n_qubits] representing the input configurations.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing three tensors:
                - index_i_and_j: A tensor of shape [..., 2] representing the indices (i, j) for the non-zero Hamiltonian terms.
                - coefs: A tensor of shape [...] representing the corresponding complex coefficients for the indices.
                - configs_target: A tensor of shape [pool_size, n_qubits] representing the target configurations after processing.
                  The first `batch_size` configurations are guaranteed to be identical to the input `configs_i`.

        The function performs the following steps:
        1. Converts the input configurations to dtype `torch.int8`.
        2. Calls the `_relative` method to obtain valid indices (`valid_index_i`), valid configurations (`valid_configs_j`), and valid coefficients (`valid_coefs`).
        3. Concatenates the input configurations and valid configurations to form a pool of configurations.
        4. Uses `torch.unique` to find the unique configurations in the pool and their corresponding indices.
        5. Maps the valid configurations to the target configurations, updating the indices accordingly.
        6. Reorder the configurations to obtain the target configuration.
        7. Returns the valid indices, coefficients, and target configurations.
        """
        # pylint: disable=too-many-locals
        device: torch.device = configs_i.device
        # Parameters
        # configs_i : bool[batch_size, n_qubits]
        # Returns
        # index_i_and_j : int64[..., 2]
        # coefs : complex128[...]

        batch_size: int = configs_i.shape[0]
        valid_index_i: torch.Tensor
        valid_configs_j: torch.Tensor
        valid_coefs: torch.Tensor
        valid_index_i, valid_configs_j, valid_coefs = self._relative(configs_i)
        # configs_i : bool[batch_size, n_qubits]
        # valid_configs_j : bool[valid_size, n_qubits]
        # valid_index_i : int64[valid_size]
        # valid_coefs : float64[valid_size, 2]

        # map from valid to pool first, and then map pool to target.

        configs_i_and_j: torch.Tensor = torch.cat([configs_i, valid_configs_j], dim=0)
        del configs_i
        del valid_configs_j
        # pool : bool[pool_size, n_qubits]
        # both_to_pool : int64[batch_size + valid_size]
        pool: torch.Tensor
        both_to_pool: torch.Tensor
        pool, both_to_pool = torch.unique(configs_i_and_j, dim=0, sorted=False, return_inverse=True, return_counts=False)
        pool_size: int = pool.shape[0]
        del configs_i_and_j

        # pool_to_source : int64[pool_size]
        pool_to_source: torch.Tensor = torch.full([pool_size], -1, dtype=torch.int64, device=device)
        source_to_pool: torch.Tensor = both_to_pool[:batch_size]
        pool_to_source[source_to_pool] = torch.arange(batch_size, device=device)
        del source_to_pool
        pool_to_source[pool_to_source == -1] = torch.arange(batch_size, pool_size, device=device)

        # destination_to_source : int64[valid_size] -> -1 or 0...batch_size-1
        destination_to_pool: torch.Tensor = both_to_pool[batch_size:]
        del both_to_pool
        destination_to_source: torch.Tensor = pool_to_source[destination_to_pool]
        del destination_to_pool

        configs_target: torch.Tensor = torch.empty_like(pool)
        configs_target[pool_to_source] = pool
        del pool
        del pool_to_source

        return torch.stack([valid_index_i, destination_to_source], dim=1), torch.view_as_complex(valid_coefs), configs_target

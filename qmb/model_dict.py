"""
This module is used to store a dictionary that maps model names to their corresponding models.
Other packages can register their models by adding entries to this dictionary.
"""

import typing
import torch


class Model(typing.Protocol):
    """
    The Model protocol defines the interface that all models must implement.
    """

    @classmethod
    def preparse(cls, args: tuple[str, ...]) -> str:
        """
        The `preparse` method is used to obtain the job name that will be used for logging purposes.
        This is essential and should be generated first before any other operations.
        """

    @classmethod
    def parse(cls, args: tuple[str, ...]) -> "Model":
        """
        The `parse` method is used to parse the arguments and return an instance of the model.
        """

    def inside(self, configs_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the Hamiltonian to the given configuration and obtains the resulting sparse Hamiltonian matrix block within the configuration subspace.
        This function only considers the terms that are within the configuration subspace.
        """

    def outside(self, configs_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies the Hamiltonian to the given configuration and obtains the resulting sparse Hamiltonian matrix block within the configuration subspace.
        This function considers both the inside and outside configurations, ensuring that the input configurations are the first `batch_size` configurations in the result.
        """


model_dict: dict[str, typing.Type[Model]] = {}

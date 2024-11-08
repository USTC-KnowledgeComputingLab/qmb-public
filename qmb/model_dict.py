"""
This module is used to store a dictionary that maps model names to their corresponding models.
Other packages can register their models by adding entries to this dictionary.
"""

import typing
import torch


class NetworkProto(typing.Protocol):
    """
    The Network protocol defines the interface that all networks must implement.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the amplitude for the given configurations.
        """

    def generate_unique(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        Generate a batch of unique configurations.
        """


_Model = typing.TypeVar('_Model')


class ModelProto(typing.Protocol[_Model]):
    """
    The Model protocol defines the interface that all models must implement.
    """

    network_dict: dict[str, typing.Callable[[_Model, tuple[str, ...]], NetworkProto]]

    @classmethod
    def preparse(cls, input_args: tuple[str, ...]) -> str:
        """
        The `preparse` method is used to obtain the job name that will be used for logging purposes.
        This is essential and should be generated first before any other operations.
        """

    @classmethod
    def parse(cls, input_args: tuple[str, ...]) -> _Model:
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


model_dict: dict[str, typing.Type[ModelProto]] = {}

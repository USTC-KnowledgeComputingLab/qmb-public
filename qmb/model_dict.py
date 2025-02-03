"""
This module is used to store a dictionary that maps model names to their corresponding models.

Other packages or subpackages can register their models by adding entries to this dictionary.
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

    def generate_unique(self, batch_size: int, block_num: int = 1) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        Generate a batch of unique configurations.
        """

    def load_state_dict(self, data: dict[str, torch.Tensor]) -> typing.Any:
        """torch.nn.Module function"""

    def state_dict(self) -> dict[str, torch.Tensor]:
        """torch.nn.Module function"""

    def to(self, device: torch.device) -> typing.Any:
        """torch.nn.Module function"""

    def parameters(self) -> typing.Iterable[torch.Tensor]:
        """torch.nn.Module function"""

    def bfloat16(self) -> typing.Self:
        """torch.nn.Module function"""

    def half(self) -> typing.Self:
        """torch.nn.Module function"""

    def float(self) -> typing.Self:
        """torch.nn.Module function"""

    def double(self) -> typing.Self:
        """torch.nn.Module function"""


class ModelProto(typing.Protocol):
    """
    The Model protocol defines the interface that all models must implement.
    """

    network_dict: dict[str, typing.Callable[[typing.Self, tuple[str, ...]], NetworkProto]]

    ref_energy: float

    @classmethod
    def preparse(cls, input_args: tuple[str, ...]) -> str:
        """
        Preparse the arguments to obtain the group name for logging perposes

        Parameters
        ----------
        input_args : tuple[str, ...]
            The input arguments to the model.

        Returns
        -------
        str
            The group name for logging purposes.
        """

    @classmethod
    def parse(cls, input_args: tuple[str, ...]) -> typing.Self:
        """
        Parse the arguments and return an instance of the model.

        Parameters
        ----------
        input_args : tuple[str, ...]
            The input arguments to the model.

        Returns
        -------
        Self
            An instance of the model.
        """

    def apply_within(self, configs_i: torch.Tensor, psi_i: torch.Tensor, configs_j: torch.Tensor) -> torch.Tensor:
        """
        Applies the Hamiltonian to the given vector.

        Parameters
        ----------
        configs_i : torch.Tensor
            The configurations to apply the Hamiltonian to.
        psi_i : torch.Tensor
            The amplitudes of the configurations.
        configs_j : torch.Tensor
            The configurations subspace for the result of the Hamiltonian application.

        Returns
        -------
        torch.Tensor
            The result of the Hamiltonian application on the selected configurations subspace.
        """

    def find_relative(self, configs_i: torch.Tensor, psi_i: torch.Tensor, count_selected: int) -> torch.Tensor:
        """
        Find relative configurations to the given configurations.

        Parameters
        ----------
        configs_i : torch.Tensor
            The configurations to find relative configurations for.
        psi_i : torch.Tensor
            The amplitudes of the configurations.
        count_selected : int
            The number of relative configurations to find.

        Returns
        -------
        torch.Tensor
            The relative configurations.
        """

    def show_config(self, config: torch.Tensor) -> str:
        """
        Converts a configuration tensor to a string representation.

        Parameters
        ----------
        config : torch.Tensor
            The configuration tensor to convert.

        Returns
        -------
        str
            The string representation of the configuration tensor.
        """


model_dict: dict[str, typing.Type[ModelProto]] = {}

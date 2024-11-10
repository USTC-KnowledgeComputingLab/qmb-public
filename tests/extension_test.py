"""
Test module for PyTorch C++ extension.
"""

import torch
import qmb.hamiltonian


def test_import() -> None:
    """
    Test the import and availability of the PyTorch C++ extension operations.
    """
    # pylint: disable=protected-access
    extension = qmb.hamiltonian.Hamiltonian._get_extension()
    _ = getattr(extension, "prepare")
    _ = torch.ops._hamiltonian.bose2
    _ = torch.ops._hamiltonian.fermi

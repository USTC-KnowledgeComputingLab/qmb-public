"""
Test module for the collection extension of configurations.
"""

import torch
import qmb.hamiltonian

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access


def test_node() -> None:
    node = getattr(qmb.hamiltonian.Hamiltonian._get_collection_module(), "Node")()

    count = 0

    count += node.add_tensor(
        torch.view_as_real(torch.tensor([1 + 2j, 3 + 4j, 5 + 6j, 3 + 2j], dtype=torch.complex128)),
        torch.tensor([[234, 2], [2, 1], [234, 2], [2, 0]], dtype=torch.uint8),
    )

    count += node.add_tensor(
        torch.view_as_real(torch.tensor([1 + 2j, 3 + 4j, 5 + 6j], dtype=torch.complex128)),
        torch.tensor([[255, 2], [2, 0], [2, 0]], dtype=torch.uint8),
    )

    psi, config = node.get_tensor(
        torch.tensor([[255, 2], [2, 0]], dtype=torch.uint8),
        count,
        True,
    )

    assert torch.allclose(psi, torch.tensor([[1, 2], [11, 12], [3, 4], [6, 8]], dtype=torch.float64))
    assert torch.allclose(config, torch.tensor([[255, 2], [2, 0], [2, 1], [234, 2]], dtype=torch.uint8))

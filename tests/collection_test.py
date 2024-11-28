"""
Test module for the collection extension of configurations.
"""

import torch
import qmb.hamiltonian

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access


def test_node() -> None:
    qmb.hamiltonian.Hamiltonian._get_collection_module()

    key1 = torch.tensor([[1, 3], [4, 1], [1, 2], [3, 1]], dtype=torch.uint8).cuda()
    value1 = torch.tensor([[2.], [1.], [3.], [7.]], dtype=torch.float64).cuda()
    key2 = torch.tensor([[4, 1], [4, 2], [3, 1], [2, 4]], dtype=torch.uint8).cuda()
    value2 = torch.tensor([[4.], [5.], [3.], [6.]], dtype=torch.float64).cuda()

    key1, value1 = torch.ops.qmb_collection.sort_(key1, value1)
    key2, value2 = torch.ops.qmb_collection.sort_(key2, value2)

    assert torch.allclose(key1, torch.tensor([[1, 2], [1, 3], [3, 1], [4, 1]], dtype=torch.uint8).cuda())
    assert torch.allclose(value1, torch.tensor([[3.], [2.], [7.], [1.]], dtype=torch.float64).cuda())
    assert torch.allclose(key2, torch.tensor([[2, 4], [3, 1], [4, 1], [4, 2]], dtype=torch.uint8).cuda())
    assert torch.allclose(value2, torch.tensor([[6.], [3.], [4.], [5.]], dtype=torch.float64).cuda())

    key, value = torch.ops.qmb_collection.merge(key1, value1, key2, value2)
    assert torch.allclose(key, torch.tensor([[1, 2], [1, 3], [2, 4], [3, 1], [3, 1], [4, 1], [4, 1], [4, 2]], dtype=torch.uint8).cuda())
    assert torch.allclose(value, torch.tensor([[3.], [2.], [6.], [7.], [3.], [1.], [4.], [5.]], dtype=torch.float64).cuda())

    key, value = torch.ops.qmb_collection.reduce(key, value)
    assert torch.allclose(key, torch.tensor([[1, 2], [1, 3], [2, 4], [3, 1], [4, 1], [4, 2]], dtype=torch.uint8).cuda())
    assert torch.allclose(value, torch.tensor([[3.], [2.], [6.], [10.], [5.], [5.]], dtype=torch.float64).cuda())

    config = torch.tensor([[3, 1], [1, 3]], dtype=torch.uint8).cuda()
    key, value = torch.ops.qmb_collection.ensure_(key, value, config)
    assert torch.allclose(key, torch.tensor([[3, 1], [1, 3], [1, 2], [2, 4], [4, 1], [4, 2]], dtype=torch.uint8).cuda())
    assert torch.allclose(value, torch.tensor([[10.], [2.], [3.], [6.], [5.], [5.]], dtype=torch.float64).cuda())

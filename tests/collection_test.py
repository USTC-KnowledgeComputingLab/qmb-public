"""
Test module for the collection extension of configurations.
"""

import torch
import qmb.hamiltonian

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access


def test_collection() -> None:
    module = qmb.hamiltonian.Hamiltonian._load_collection(n_qubytes=2)
    op_sort = getattr(module, "sort")
    op_merge = getattr(module, "merge")
    op_reduce = getattr(module, "reduce")
    op_ensure = getattr(module, "ensure")

    key1 = torch.tensor([[1, 3], [4, 1], [1, 2], [3, 1]], dtype=torch.uint8).cuda()
    value1 = torch.tensor([[2.], [1.], [3.], [7.]], dtype=torch.float64).cuda()
    key2 = torch.tensor([[4, 1], [4, 2], [3, 1], [2, 4]], dtype=torch.uint8).cuda()
    value2 = torch.tensor([[4.], [5.], [3.], [6.]], dtype=torch.float64).cuda()

    key1, value1 = op_sort(key1, value1)
    key2, value2 = op_sort(key2, value2)

    assert torch.allclose(key1, torch.tensor([[1, 2], [1, 3], [3, 1], [4, 1]], dtype=torch.uint8).cuda())
    assert torch.allclose(value1, torch.tensor([[3.], [2.], [7.], [1.]], dtype=torch.float64).cuda())
    assert torch.allclose(key2, torch.tensor([[2, 4], [3, 1], [4, 1], [4, 2]], dtype=torch.uint8).cuda())
    assert torch.allclose(value2, torch.tensor([[6.], [3.], [4.], [5.]], dtype=torch.float64).cuda())

    key, value = op_merge(key1, value1, key2, value2)
    assert torch.allclose(key, torch.tensor([[1, 2], [1, 3], [2, 4], [3, 1], [3, 1], [4, 1], [4, 1], [4, 2]], dtype=torch.uint8).cuda())
    assert torch.allclose(value, torch.tensor([[3.], [2.], [6.], [7.], [3.], [1.], [4.], [5.]], dtype=torch.float64).cuda())

    key, value = op_reduce(key, value)
    assert torch.allclose(key, torch.tensor([[1, 2], [1, 3], [2, 4], [3, 1], [4, 1], [4, 2]], dtype=torch.uint8).cuda())
    assert torch.allclose(value, torch.tensor([[3.], [2.], [6.], [10.], [5.], [5.]], dtype=torch.float64).cuda())

    config = torch.tensor([[3, 1], [1, 3]], dtype=torch.uint8).cuda()
    key, value = op_ensure(key, value, config)
    assert torch.allclose(key, torch.tensor([[3, 1], [1, 3], [2, 4], [4, 1], [4, 2], [1, 2]], dtype=torch.uint8).cuda())
    assert torch.allclose(value, torch.tensor([[10.], [2.], [6.], [5.], [5.], [3.]], dtype=torch.float64).cuda())

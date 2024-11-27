"""
This module provides tools for PyTorch optimizers.
"""

import typing
import pathlib
import torch


def _migrate_tensor(tensor: torch.Tensor, device: torch.device) -> None:
    """
    Migrates the tensor to the specified device.
    """
    tensor.data = tensor.data.to(device=device)
    if tensor.grad is not None:
        tensor.grad.data = tensor.grad.data.to(device=device)


def _migrate_optimizer(optimizer: torch.optim.Optimizer) -> None:
    """
    Migrates the optimizer to the device of the model parameters.
    """
    # pylint: disable=too-many-nested-blocks
    # pylint: disable=too-many-branches
    device: torch.device = optimizer.param_groups[0]["params"][0].device
    for param in optimizer.state.values():
        if isinstance(param, torch.Tensor):
            _migrate_tensor(param, device)
        elif isinstance(param, list):
            for subparam in param:
                if isinstance(subparam, torch.Tensor):
                    _migrate_tensor(subparam, device)
        elif isinstance(param, int | float):
            pass
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    _migrate_tensor(subparam, device)
                elif isinstance(subparam, list):
                    for subsubparam in subparam:
                        if isinstance(subsubparam, torch.Tensor):
                            _migrate_tensor(subsubparam, device)
                elif isinstance(subparam, int | float):
                    pass
                else:
                    raise ValueError(f"Unexpected parameter type: {type(subparam)}")
        else:
            raise ValueError(f"Unexpected parameter type: {type(param)}")


def initialize_optimizer(  # pylint: disable=too-many-arguments
    params: typing.Iterable[torch.Tensor],
    *,
    use_lbfgs: bool,
    learning_rate: float,
    new_opt: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
    optimizer_path: pathlib.Path | None = None,
) -> torch.optim.Optimizer:
    """
    Initialize an optimizer.
    """
    if new_opt or optimizer is None:
        if use_lbfgs:
            optimizer = torch.optim.LBFGS(params, lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(params, lr=learning_rate)
    if optimizer_path is not None:
        if optimizer_path.exists():
            state_dict = torch.load(optimizer_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(state_dict)
            _migrate_optimizer(optimizer)
    return optimizer


def scale_learning_rate(optimizer: torch.optim.Optimizer, scale: float) -> None:
    """
    Scales the learning rate of all parameter groups in the optimizer by a given factor.
    """
    for param in optimizer.param_groups:
        param["lr"] *= scale

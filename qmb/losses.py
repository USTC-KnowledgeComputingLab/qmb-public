"""
This file contains various loss functions used in the learning script.
These functions help calculate the difference between the target wave function and the current state wave function.
"""

import torch


@torch.jit.script
def log(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Calculate the loss based on the difference between the log of the current state wave function and the target wave function.
    """
    s_abs = torch.sqrt(s.real**2 + s.imag**2)
    t_abs = torch.sqrt(t.real**2 + t.imag**2)

    s_angle = torch.atan2(s.imag, s.real)
    t_angle = torch.atan2(t.imag, t.real)

    s_magnitude = torch.log(s_abs)
    t_magnitude = torch.log(t_abs)

    error_real = (s_magnitude - t_magnitude) / (2 * torch.pi)
    error_imag = (s_angle - t_angle) / (2 * torch.pi)
    error_imag = error_imag - error_imag.round()

    loss = error_real**2 + error_imag**2
    return loss.mean()


@torch.jit.script
def target_reweighted_log(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Calculate the loss based on the difference between the log of the current state wave function and the target wave function,
    but reweighted by the abs of the target wave function.
    """
    s_abs = torch.sqrt(s.real**2 + s.imag**2)
    t_abs = torch.sqrt(t.real**2 + t.imag**2)

    s_angle = torch.atan2(s.imag, s.real)
    t_angle = torch.atan2(t.imag, t.real)

    s_magnitude = torch.log(s_abs)
    t_magnitude = torch.log(t_abs)

    error_real = (s_magnitude - t_magnitude) / (2 * torch.pi)
    error_imag = (s_angle - t_angle) / (2 * torch.pi)
    error_imag = error_imag - error_imag.round()

    loss = error_real**2 + error_imag**2
    loss = loss * t_abs
    return loss.mean()


@torch.jit.script
def target_filtered_log(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Calculate the loss based on the difference between the log of the current state wave function and the target wave function,
    but filtered by the abs of the target wave function.
    """
    s_abs = torch.sqrt(s.real**2 + s.imag**2)
    t_abs = torch.sqrt(t.real**2 + t.imag**2)

    s_angle = torch.atan2(s.imag, s.real)
    t_angle = torch.atan2(t.imag, t.real)

    s_magnitude = torch.log(s_abs)
    t_magnitude = torch.log(t_abs)

    error_real = (s_magnitude - t_magnitude) / (2 * torch.pi)
    error_imag = (s_angle - t_angle) / (2 * torch.pi)
    error_imag = error_imag - error_imag.round()

    loss = error_real**2 + error_imag**2
    loss = loss / (1 + 1e-10 / t_abs)
    # This function scale only for very small abs value.
    # I think we could ignore those definitly for amplitude less than 1e-10.
    return loss.mean()


@torch.jit.script
def sum_reweighted_log(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Calculate the loss based on the difference between the log of the current state wave function and the target wave function,
    but reweighted by the sum of the abs of the current state wave function and the target wave function.
    """
    s_abs = torch.sqrt(s.real**2 + s.imag**2)
    t_abs = torch.sqrt(t.real**2 + t.imag**2)

    s_angle = torch.atan2(s.imag, s.real)
    t_angle = torch.atan2(t.imag, t.real)

    s_magnitude = torch.log(s_abs)
    t_magnitude = torch.log(t_abs)

    error_real = (s_magnitude - t_magnitude) / (2 * torch.pi)
    error_imag = (s_angle - t_angle) / (2 * torch.pi)
    error_imag = error_imag - error_imag.round()

    loss = error_real**2 + error_imag**2
    loss = loss * (t_abs + s_abs)
    return loss.mean()


@torch.jit.script
def sum_filtered_log(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Calculate the loss based on the difference between the log of the current state wave function and the target wave function,
    but filtered by the sum of the abs of the current state wave function and the target wave function.
    """
    s_abs = torch.sqrt(s.real**2 + s.imag**2)
    t_abs = torch.sqrt(t.real**2 + t.imag**2)

    s_angle = torch.atan2(s.imag, s.real)
    t_angle = torch.atan2(t.imag, t.real)

    s_magnitude = torch.log(s_abs)
    t_magnitude = torch.log(t_abs)

    error_real = (s_magnitude - t_magnitude) / (2 * torch.pi)
    error_imag = (s_angle - t_angle) / (2 * torch.pi)
    error_imag = error_imag - error_imag.round()

    loss = error_real**2 + error_imag**2
    loss = loss / (1 + 1e-10 / (t_abs + s_abs))
    # This function scale only for very small abs value.
    # I think we could ignore those definitly for amplitude less than 1e-10.
    return loss.mean()


@torch.jit.script
def direct(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Calculate the loss based on the difference between the current state wave function and the target wave function directly.
    """
    error = s - t
    loss = error.real**2 + error.imag**2
    return loss.mean()

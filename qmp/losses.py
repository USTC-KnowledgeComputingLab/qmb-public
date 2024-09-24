import torch


def log(a, b):
    # b is target, a is learnable
    error = (a.log() - b.log()) / (2 * torch.pi)
    error_real = error.real
    error_imag = error.imag - error.imag.round()
    loss = error_real**2 + error_imag**2
    return loss.mean()


def target_reweighted_log(a, b):
    # b is target, a is learnable
    error = (a.log() - b.log()) / (2 * torch.pi)
    error_real = error.real
    error_imag = error.imag - error.imag.round()
    loss = error_real**2 + error_imag**2
    abs_b = b.abs()
    loss = loss * abs_b
    return loss.mean()


def target_filtered_log(a, b):
    # b is target, a is learnable
    error = (a.log() - b.log()) / (2 * torch.pi)
    error_real = error.real
    error_imag = error.imag - error.imag.round()
    loss = error_real**2 + error_imag**2
    abs_b = b.abs()
    loss = loss / (1 + 1e-10 / abs_b)
    # This function scale only for very small abs value.
    # I think we could ignore those definitly for amplitude less than 1e-10.
    return loss.mean()


def sum_filtered_log(a, b):
    # b is target, a is learnable
    error = (a.log() - b.log()) / (2 * torch.pi)
    error_real = error.real
    error_imag = error.imag - error.imag.round()
    loss = error_real**2 + error_imag**2
    sum_a_b = a.abs() + b.abs()
    loss = loss / (1 + 1e-10 / sum_a_b)
    # This function scale only for very small abs value.
    # I think we could ignore those definitly for amplitude less than 1e-10.
    return loss.mean()


def direct(a, b):
    # b is target, a is learnable
    error = a - b
    loss = (error.conj() * error).real
    return loss.mean()

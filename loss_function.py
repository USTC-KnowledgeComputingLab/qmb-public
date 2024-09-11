import torch


@torch.jit.script
def target_reweighted_log(a, b):
    # b is target, a is learnable
    log_a = a.log()
    log_b = b.log()
    error_real = (log_a.real - log_b.real) / (2 * torch.pi)
    error_imag = (log_a.imag - log_b.imag) / (2 * torch.pi)
    error_imag = error_imag - error_imag.round()
    loss = (error_real**2) + (error_imag**2)
    loss = loss * b.abs()
    return loss.mean()


@torch.jit.script
def log(a, b):
    # b is target, a is learnable
    log_a = a.log()
    log_b = b.log()
    error_real = (log_a.real - log_b.real) / (2 * torch.pi)
    error_imag = (log_a.imag - log_b.imag) / (2 * torch.pi)
    error_imag = error_imag - error_imag.round()
    loss = (error_real**2) + (error_imag**2)
    return loss.mean()


@torch.jit.script
def direct(a, b):
    # b is target, a is learnable
    error = a - b
    loss = (error.conj() * error).real
    return loss.mean()

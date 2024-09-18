import torch
import logging


def _cg(matrix, vector, max_step=None, threshold=None):

    def D(v):
        return matrix @ v

    def DT(v):
        return matrix.H @ v

    def A(v):
        return DT(D(v))

    logging.info("conjugate gradient starting")

    b = vector
    b_square = b.conj() @ b
    x = torch.zeros_like(b)
    r = b
    r_square = r.conj() @ r
    p = r
    t = 0
    while True:
        error_square = r_square / b_square
        logging.info("conjugate gradient step %d error %f", t, error_square.sqrt())
        if max_step is not None and t == max_step:
            logging.info("conjugate gradient step stop because of max step reached")
            break
        if threshold is not None and error_square < threshold**2:
            logging.info("conjugate gradient step stop because of threshold reached")
            break
        Dp = D(p)
        pAp = Dp.conj() @ Dp
        alpha = r_square / pAp
        x = x + alpha * p
        r = r - alpha * DT(Dp)
        new_r_square = r.conj() @ r
        beta = new_r_square / r_square
        r_square = new_r_square
        p = r + beta * p
        t += 1

    logging.info("conjugate gradient finished")
    return x


def cg(matrix, vector, max_step=None, threshold=None):
    logging.info("reshaping gradient and matric to vector and matrix")
    _vector = torch.cat([tensor.view([-1]) for tensor in vector])
    _matrix = torch.stack([torch.cat([tensor.view([-1]) for tensor in line]) for line in matrix])
    _x = _cg(_matrix, _vector, max_step=max_step, threshold=threshold)
    logging.info("reshaping inversion result back to tensors")
    x = []
    index = 0
    for tensor in vector:
        size = tensor.nelement()
        x.append(_x[index:index + size].view(tensor.shape))
        index += size
    logging.info("natural gradient method direction calculated")
    return x

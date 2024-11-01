# The LOBPCG implementation in Torch has a bug, so we will implement it again here.
# See https://github.com/pytorch/pytorch/issues/135860
# This file is copied from SciPy 1.14.1.

import warnings
import scipy
import torch

__all__ = ["lobpcg"]


@torch.jit.ignore
def _eigh(A: torch.Tensor, B: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    # PyTorch does not natively support the generalized eigenvalue problem.
    # This function handles the generalized eigenvalue problem by converting the tensors to NumPy arrays and using SciPy's `scipy.linalg.eigh` function.
    # The results are then converted back to PyTorch tensors and returned.
    device = A.device
    A = A.cpu().numpy()
    B = B.cpu().numpy() if B is not None else None
    try:
        eigvals, eigvecs = scipy.linalg.eigh(A, B)
    except scipy.linalg.LinAlgError:
        return torch.empty(0, device=device), torch.empty(0, device=device)
    return torch.tensor(eigvals).to(device=device), torch.tensor(eigvecs).to(device=device)


@torch.jit.ignore
def _eps(A: torch.Tensor) -> float:
    return torch.finfo(A.dtype).eps


@torch.jit.ignore
def _warn(msg: str) -> None:
    warnings.warn(msg, UserWarning, stacklevel=3)


@torch.jit.script
def lobpcg(A: torch.Tensor, X: torch.Tensor, tol: float | None = None, maxiter: int = 20, restartControl: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    device = A.device
    dtype = A.dtype
    eps = _eps(A)
    myeps = eps**(1 / 2)
    bestIterationNumber = maxiter

    blockVectorX = X
    bestblockVectorX = blockVectorX

    n = blockVectorX.shape[0]
    sizeX = blockVectorX.shape[1]

    if tol is None:
        residualTolerance = myeps * n
    else:
        residualTolerance = tol

    blockVectorX = blockVectorX / torch.norm(blockVectorX)
    blockVectorAX = A @ blockVectorX

    gramXAX = blockVectorX.T.conj() @ blockVectorAX

    _lambda, eigBlockVector = _eigh(gramXAX, None)
    ii = torch.argsort(_lambda)[:sizeX]
    _lambda = _lambda[ii]

    eigBlockVector = eigBlockVector[:, ii]
    blockVectorX = blockVectorX @ eigBlockVector
    blockVectorAX = blockVectorAX @ eigBlockVector

    activeMask = torch.ones([sizeX], dtype=torch.bool, device=device)

    blockVectorP = blockVectorAP = activeBlockVectorP = activeBlockVectorAP = torch.empty([0, 0], device=device)
    smallestResidualNorm = 0.0

    iterationNumber = -1
    restart = True
    forcedRestart = False
    explicitGramFlag = False
    while iterationNumber < maxiter:
        iterationNumber += 1

        blockVectorR = blockVectorAX - blockVectorX * _lambda[None, :]
        residualNorms = torch.sqrt(torch.abs(torch.sum(blockVectorR.conj() * blockVectorR, dim=0)))
        residualNorm = torch.sum(torch.abs(residualNorms)) / sizeX

        if iterationNumber == 0 or residualNorm < smallestResidualNorm:
            smallestResidualNorm = residualNorm
            bestIterationNumber = iterationNumber
            bestblockVectorX = blockVectorX
        elif residualNorm > 2**restartControl * smallestResidualNorm:
            forcedRestart = True
            blockVectorAX = A @ blockVectorX

        ii = residualNorms > residualTolerance
        activeMask = activeMask & ii
        currentBlockSize = activeMask.sum()

        if currentBlockSize == 0:
            break

        activeBlockVectorR = blockVectorR[:, activeMask]

        if iterationNumber > 0:
            activeBlockVectorP = blockVectorP[:, activeMask]
            activeBlockVectorAP = blockVectorAP[:, activeMask]

        activeBlockVectorR = activeBlockVectorR - (blockVectorX @ (blockVectorX.T.conj() @ activeBlockVectorR))

        aux = torch.norm(activeBlockVectorR)
        if aux == 0:
            _warn(f"Failed at iteration {iterationNumber} with accuracies "
                  f"{residualNorms}\n not reaching the requested "
                  f"tolerance {residualTolerance}.")
            break
        activeBlockVectorR = activeBlockVectorR / aux
        activeBlockVectorAR = A @ activeBlockVectorR

        if iterationNumber > 0:
            aux = torch.norm(activeBlockVectorP)
            if aux == 0:
                restart = True
            else:
                aux = 1 / aux
                activeBlockVectorP = activeBlockVectorP * aux
                activeBlockVectorAP = activeBlockVectorAP * aux
                restart = forcedRestart

        if residualNorms.max() > myeps and not explicitGramFlag:
            explicitGramFlag = False
        else:
            explicitGramFlag = True

        gramXAR = blockVectorX.T.conj() @ activeBlockVectorAR
        gramRAR = activeBlockVectorR.T.conj() @ activeBlockVectorAR

        if explicitGramFlag:
            gramRAR = (gramRAR + gramRAR.T.conj()) / 2
            gramXAX = blockVectorX.T.conj() @ blockVectorAX
            gramXAX = (gramXAX + gramXAX.T.conj()) / 2
            gramXBX = blockVectorX.T.conj() @ blockVectorX
            gramRBR = activeBlockVectorR.T.conj() @ activeBlockVectorR
            gramXBR = blockVectorX.T.conj() @ activeBlockVectorR
        else:
            gramXAX = torch.diag(_lambda).to(dtype)
            gramXBX = torch.eye(sizeX, dtype=dtype, device=device)
            gramRBR = torch.eye(currentBlockSize, dtype=dtype, device=device)
            gramXBR = torch.zeros([sizeX, int(currentBlockSize)], dtype=dtype, device=device)

        if not restart:
            gramXAP = blockVectorX.T.conj() @ activeBlockVectorAP
            gramRAP = activeBlockVectorR.T.conj() @ activeBlockVectorAP
            gramPAP = activeBlockVectorP.T.conj() @ activeBlockVectorAP
            gramXBP = blockVectorX.T.conj() @ activeBlockVectorP
            gramRBP = activeBlockVectorR.T.conj() @ activeBlockVectorP
            if explicitGramFlag:
                gramPAP = (gramPAP + gramPAP.T.conj()) / 2
                gramPBP = activeBlockVectorP.T.conj() @ activeBlockVectorP
            else:
                gramPBP = torch.eye(currentBlockSize, dtype=dtype, device=device)

            gramA = torch.cat([torch.cat([gramXAX, gramXAR, gramXAP], dim=1),
                               torch.cat([gramXAR.T.conj(), gramRAR, gramRAP], dim=1),
                               torch.cat([gramXAP.T.conj(), gramRAP.T.conj(), gramPAP], dim=1)],
                              dim=0)
            gramB = torch.cat([torch.cat([gramXBX, gramXBR, gramXBP], dim=1),
                               torch.cat([gramXBR.T.conj(), gramRBR, gramRBP], dim=1),
                               torch.cat([gramXBP.T.conj(), gramRBP.T.conj(), gramPBP], dim=1)],
                              dim=0)

            _lambda, eigBlockVector = _eigh(gramA, gramB)
            if _lambda.numel() == 0:
                restart = True

        if restart:
            gramA = torch.cat([torch.cat([gramXAX, gramXAR], dim=1), torch.cat([gramXAR.T.conj(), gramRAR], dim=1)], dim=0)

            gramB = torch.cat([torch.cat([gramXBX, gramXBR], dim=1), torch.cat([gramXBR.T.conj(), gramRBR], dim=1)], dim=0)

            _lambda, eigBlockVector = _eigh(gramA, gramB)
            if _lambda.numel() == 0:
                _warn(f"eigh failed at iteration {iterationNumber} with error\n")
                break

        ii = torch.argsort(_lambda)[:sizeX]
        _lambda = _lambda[ii]
        eigBlockVector = eigBlockVector[:, ii]

        if not restart:
            eigBlockVectorX = eigBlockVector[:sizeX]
            eigBlockVectorR = eigBlockVector[sizeX:sizeX + currentBlockSize]
            eigBlockVectorP = eigBlockVector[sizeX + currentBlockSize:]

            pp = activeBlockVectorR @ eigBlockVectorR
            pp = pp + activeBlockVectorP @ eigBlockVectorP

            app = activeBlockVectorAR @ eigBlockVectorR
            app = app + activeBlockVectorAP @ eigBlockVectorP
        else:
            eigBlockVectorX = eigBlockVector[:sizeX]
            eigBlockVectorR = eigBlockVector[sizeX:]

            pp = activeBlockVectorR @ eigBlockVectorR
            app = activeBlockVectorAR @ eigBlockVectorR

        blockVectorX = blockVectorX @ eigBlockVectorX + pp
        blockVectorAX = blockVectorAX @ eigBlockVectorX + app

        blockVectorP, blockVectorAP = pp, app

    blockVectorR = blockVectorAX - blockVectorX * _lambda[None, :]
    residualNorms = torch.sqrt(torch.abs(torch.sum(blockVectorR.conj() * blockVectorR, dim=0)))
    residualNorm = torch.sum(torch.abs(residualNorms)) / sizeX

    if residualNorm < smallestResidualNorm:
        smallestResidualNorm = residualNorm
        bestIterationNumber = iterationNumber + 1
        bestblockVectorX = blockVectorX

    if torch.max(torch.abs(residualNorms)) > residualTolerance:
        _warn(f"Exited at iteration {iterationNumber} with accuracies \n"
              f"{residualNorms}\n"
              f"not reaching the requested tolerance {residualTolerance}.\n"
              f"Use iteration {bestIterationNumber} instead with accuracy \n"
              f"{smallestResidualNorm}.\n")

    blockVectorX = bestblockVectorX
    blockVectorAX = A @ blockVectorX
    gramXAX = blockVectorX.T.conj() @ blockVectorAX
    gramXBX = blockVectorX.T.conj() @ blockVectorX
    gramXAX = (gramXAX + gramXAX.T.conj()) / 2
    gramXBX = (gramXBX + gramXBX.T.conj()) / 2
    _lambda, eigBlockVector = _eigh(gramXAX, gramXBX)

    ii = torch.argsort(_lambda)[:sizeX]
    _lambda = _lambda[ii]
    eigBlockVector = eigBlockVector[:, ii]

    blockVectorX = blockVectorX @ eigBlockVector

    return _lambda, blockVectorX

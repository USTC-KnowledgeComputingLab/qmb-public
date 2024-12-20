import os
import sys
import gc
import functools
import typing
import torch
import torch.utils.cpp_extension
from datetime import datetime


def _get_collection_module(n_qubytes: int = 0, max_op_number: int = 4) -> object:
    folder = os.path.dirname(__file__)
    torch.utils.cpp_extension.load(
        name=f"qmb_test",
        sources=[
            f"{folder}/test.cu",
        ],
        is_python_module=False,
        extra_cflags=["-O3", "-ffast-math", "-march=native", f"-DN_QUBYTES={n_qubytes}", f"-DMAX_OP_NUMBER={max_op_number}"],
        extra_cuda_cflags=["-O3", "--use_fast_math", f"-DN_QUBYTES={n_qubytes}", f"-DMAX_OP_NUMBER={max_op_number}"],
    )
    return getattr(torch.ops, f"qmb_test")

torch.set_grad_enabled(False)

configs, psi, site, kind, coef = torch.load(sys.argv[1], weights_only=True)
site = site.cuda()
kind = kind.cuda()
coef = coef.cuda()
print(configs)
print(configs.shape)
configs = configs[:33550000]
psi     = psi    [:33550000]


batch_size = configs.size(0)
n_qubytes = configs.size(1)
term_size = site.size(0)

m = _get_collection_module(n_qubytes, 4)
result = None

block = 512
block_size = site.size(0) // block + 1
for i in range(block):
    print(i, datetime.now().strftime("%H:%M:%S"))
    s = slice(i*block_size, (i+1)*block_size)
    result_block = m.test(
            configs, psi,
            site[s], kind[s], coef[s])
    if result is None:
        result = result_block
    else:
        result = result + result_block

print(result.shape)
print(result)

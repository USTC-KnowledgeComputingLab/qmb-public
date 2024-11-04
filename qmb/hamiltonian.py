import os
import torch.utils.cpp_extension


def get_extension():
    return torch.utils.cpp_extension.load(name="_hamiltonian", sources=f"{os.path.dirname(__file__)}/_hamiltonian.cu")


extension = None


def __getattr__(name):
    global extension
    if extension is None:
        extension = get_extension()
    return getattr(extension, name)

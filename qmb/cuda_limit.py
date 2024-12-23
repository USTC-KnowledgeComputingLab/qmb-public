"""
This module is used to initialize the heap limit of the GPU.
"""

import platform
import ctypes

if platform.system() == "Linux":
    _cudart = ctypes.CDLL("libcudart.so")
else:
    raise RuntimeError("Unsupported platform")


def _set_heap_limit(expect: int) -> int:
    _cudart.cudaDeviceSetLimit(2, ctypes.c_size_t(expect))
    value = ctypes.c_size_t()
    _cudart.cudaDeviceGetLimit(ctypes.pointer(value), 2)
    return value.value


def _initialize_heap_limit() -> None:
    expect = 1 << 60
    while True:
        expect >>= 1
        if _set_heap_limit(expect) == expect:
            break
    expect >>= 1
    _set_heap_limit(expect)


_initialize_heap_limit()

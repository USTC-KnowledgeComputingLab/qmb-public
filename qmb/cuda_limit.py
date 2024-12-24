"""
This module is used to initialize the heap limit of the GPU.
"""

import warnings
import platform
import ctypes


def _set_heap_limit(cudart: object, expect: int) -> int:
    getattr(cudart, "cudaDeviceSetLimit")(2, ctypes.c_size_t(expect))
    value = ctypes.c_size_t()
    getattr(cudart, "cudaDeviceGetLimit")(ctypes.pointer(value), 2)
    return value.value


def _initialize_heap_limit(cudart: object) -> None:
    expect = 1 << 60
    while True:
        expect >>= 1
        if _set_heap_limit(cudart, expect) == expect:
            break
    expect >>= 1
    _set_heap_limit(cudart, expect)


if platform.system() == "Linux":
    _cudart: object = None
    try:
        _cudart = ctypes.CDLL("libcudart.so")
    except OSError:
        warnings.warn(
            "Error initializing heap limit. Unable to proceed due to missing CUDA runtime library."
            "You may need to manually set the heap limit to ensure the program runs correctly.",
            RuntimeWarning,
        )
    if _cudart:
        _initialize_heap_limit(_cudart)
else:
    warnings.warn(
        "Error initializing heap limit. Unable to proceed due to unknown platform."
        "You may need to manually set the heap limit to ensure the program runs correctly.",
        RuntimeWarning,
    )

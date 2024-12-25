"""
This module initializes the heap limit of the GPU using CUDA runtime API.

When importing this module, it attempts to set the heap limit to a high value and then adjusts it to ensure availability.
"""

import warnings
import platform
import ctypes


def _set_heap_limit(cudart: object, expect: int) -> int:
    """
    Sets the GPU heap limit to the specified value and returns the actual set value.
    """
    cuda_limit_malloc_heap_size: int = 2
    getattr(cudart, "cudaDeviceSetLimit")(cuda_limit_malloc_heap_size, ctypes.c_size_t(expect))
    value = ctypes.c_size_t()
    getattr(cudart, "cudaDeviceGetLimit")(ctypes.pointer(value), cuda_limit_malloc_heap_size)
    return value.value


def _initialize_heap_limit(cudart: object) -> None:
    """
    Initializes the GPU heap limit by attempting to set it to a high value and adjusting it for availability.
    """
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
            "Error initializing heap limit of the GPU using CUDA runtime API."
            "Unable to proceed due to missing CUDA runtime library."
            "You may need to manually set the heap limit to ensure the program runs correctly.",
            RuntimeWarning,
        )
    if _cudart:
        _initialize_heap_limit(_cudart)
else:
    warnings.warn(
        "Error initializing heap limit of the GPU using CUDA runtime API."
        "Unable to proceed due to unknown platform."
        "You may need to manually set the heap limit to ensure the program runs correctly.",
        RuntimeWarning,
    )

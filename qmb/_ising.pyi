import numpy as np

__all__ = ['Hamiltonian']


class Hamiltonian:

    def __init__(self, X: complex, Y: complex, Z: complex, XX: complex, YY: complex, ZZ: complex) -> None:
        ...

    def inside(self, configs: np.ndarray[np.int64]) -> tuple[np.ndarray[np.int64], np.ndarray[np.complex128]]:
        ...

    def outside(self, configs: np.ndarray[np.int64]) -> tuple[np.ndarray[np.int64], np.ndarray[np.complex128], np.ndarray[np.int64]]:
        ...

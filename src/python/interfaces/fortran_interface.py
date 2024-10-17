# src/python/interfaces/fortran_interface.py

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import os


class FortranInterface:
    def __init__(self, lib_path):
        # Determine the correct library extension based on the operating system
        if os.name == "nt":  # Windows
            lib_extension = ".dll"
        else:  # Unix/Linux
            lib_extension = ".so"
        full_lib_path = lib_path + lib_extension
        self.lib = ctypes.CDLL(full_lib_path)
        # Define argument and return types
        self.lib.nose_hoover_chain.argtypes = [
            ctypes.c_double,  # dt
            ctypes.c_int,  # num_atoms
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # positions
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # velocities
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # forces
            ctypes.c_double,  # Q
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # xi
            ctypes.c_double,  # temperature
        ]
        self.lib.nose_hoover_chain.restype = None

    def nose_hoover_chain(
        self, dt, num_atoms, masses, positions, velocities, forces, Q, xi, temperature
    ):
        self.lib.nose_hoover_chain(
            ctypes.c_double(dt),
            ctypes.c_int(num_atoms),
            masses,
            positions,
            velocities,
            forces,
            ctypes.c_double(Q),
            xi,
            ctypes.c_double(temperature),
        )

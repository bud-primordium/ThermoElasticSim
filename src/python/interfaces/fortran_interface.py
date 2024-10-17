# src/python/interfaces/fortran_interface.py

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import os


class FortranInterface:
    def __init__(self, lib_name):
        # Determine the correct library extension based on the operating system
        if os.name == "nt":  # Windows
            lib_extension = ".dll"
        else:  # Unix/Linux
            lib_extension = ".so"
        lib_path = os.path.join("lib", lib_name + lib_extension)
        self.lib = ctypes.CDLL(lib_path)
        # Define argument and return types
        self.lib.nose_hoover.argtypes = [
            ctypes.c_double,  # dt
            ctypes.c_int,  # num_atoms
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # velocities
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # forces
            ctypes.POINTER(ctypes.c_double),  # xi
            ctypes.c_double,  # Q
            ctypes.c_double,  # target_temperature
        ]
        self.lib.nose_hoover.restype = None

    def nose_hoover(
        self, dt, num_atoms, masses, velocities, forces, xi, Q, target_temperature
    ):
        xi_c = ctypes.c_double(xi)
        self.lib.nose_hoover(
            ctypes.c_double(dt),
            ctypes.c_int(num_atoms),
            masses,
            velocities,
            forces,
            ctypes.byref(xi_c),
            ctypes.c_double(Q),
            ctypes.c_double(target_temperature),
        )
        return xi_c.value

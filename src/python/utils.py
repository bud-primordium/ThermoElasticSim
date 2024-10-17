# src/python/utils.py

import numpy as np
from .structure import Atom, Cell


class IOHandler:
    def read_structure(self, filename):
        # Placeholder for reading structure from file
        pass

    def write_structure(self, filename, cell):
        # Placeholder for writing structure to file
        pass


# src/python/utils.py

import numpy as np


class TensorConverter:
    @staticmethod
    def to_voigt(tensor):
        if tensor.shape != (3, 3):
            raise ValueError("Input tensor must be a 3x3 matrix.")
        if not np.allclose(tensor, tensor.T):
            raise ValueError("Input tensor must be symmetric.")

        voigt = np.array(
            [
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                tensor[0, 1],
                tensor[1, 2],
                tensor[2, 0],
            ]
        )
        return voigt

    @staticmethod
    def from_voigt(voigt):
        if voigt.shape != (6,):
            raise ValueError("Voigt representation must be a 6-element array.")

        tensor = np.array(
            [
                [voigt[0], voigt[3], voigt[5]],
                [voigt[3], voigt[1], voigt[4]],
                [voigt[5], voigt[4], voigt[2]],
            ]
        )
        return tensor


class DataCollector:
    def __init__(self):
        self.data = []

    def collect(self, cell):
        # Collect required data from the cell
        positions = [atom.position.copy() for atom in cell.atoms]
        velocities = [atom.velocity.copy() for atom in cell.atoms]
        self.data.append({"positions": positions, "velocities": velocities})

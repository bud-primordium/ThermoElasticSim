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


class TensorConverter:
    @staticmethod
    def to_voigt(tensor):
        voigt = np.array(
            [
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                2 * tensor[1, 2],
                2 * tensor[0, 2],
                2 * tensor[0, 1],
            ]
        )
        return voigt

    @staticmethod
    def from_voigt(voigt):
        tensor = np.zeros((3, 3))
        tensor[0, 0] = voigt[0]
        tensor[1, 1] = voigt[1]
        tensor[2, 2] = voigt[2]
        tensor[1, 2] = tensor[2, 1] = voigt[3] / 2
        tensor[0, 2] = tensor[2, 0] = voigt[4] / 2
        tensor[0, 1] = tensor[1, 0] = voigt[5] / 2
        return tensor


class DataCollector:
    def __init__(self):
        self.data = []

    def collect(self, cell):
        # Collect required data from the cell
        positions = [atom.position.copy() for atom in cell.atoms]
        velocities = [atom.velocity.copy() for atom in cell.atoms]
        self.data.append({"positions": positions, "velocities": velocities})

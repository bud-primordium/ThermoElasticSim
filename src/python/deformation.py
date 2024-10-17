# src/python/deformation.py

import numpy as np


class Deformer:
    def __init__(self, delta):
        self.delta = delta

    def generate_deformation_matrices(self):
        delta = self.delta
        F_list = []

        # F1
        F1 = np.array([[1 + delta, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_list.append(F1)

        # F2
        F2 = np.array([[1, 0, 0], [0, 1 + delta, 0], [0, 0, 1]])
        F_list.append(F2)

        # F3
        F3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1 + delta]])
        F_list.append(F3)

        # F4
        F4 = np.array([[1, delta, 0], [0, 1, 0], [0, 0, 1]])
        F_list.append(F4)

        # F5
        F5 = np.array([[1, 0, delta], [0, 1, 0], [0, 0, 1]])
        F_list.append(F5)

        # F6
        F6 = np.array([[1, 0, 0], [0, 1, delta], [0, 0, 1]])
        F_list.append(F6)

        return F_list

    def apply_deformation(self, cell, deformation_matrix):
        cell.apply_deformation(deformation_matrix)

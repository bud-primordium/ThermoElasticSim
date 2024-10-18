# tests/test_elasticity.py

import pytest
import numpy as np
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.elasticity import ElasticConstantsCalculator
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_elastic_constants_calculator():
    """
    @brief 测试 ElasticConstantsCalculator 计算弹性常数
    """
    # Create a more complex cell, e.g., face-centered cubic (FCC) lattice, with repetitions to increase atom count
    atoms = []
    lattice_constant = 5.1  # Å
    repetitions = 2  # 2x2x2 unit cells
    for i in range(repetitions):
        for j in range(repetitions):
            for k in range(repetitions):
                base = np.array([i, j, k]) * lattice_constant
                positions = [
                    base + np.array([0.0, 0.0, 0.0]),
                    base + np.array([0.0, 0.5, 0.5]),
                    base + np.array([0.5, 0.0, 0.5]),
                    base + np.array([0.5, 0.5, 0.0]),
                ]
                for pos in positions:
                    atoms.append(Atom(id=len(atoms), mass=26.9815, position=pos, symbol="Al"))
    
    lattice_vectors = np.eye(3) * lattice_constant * repetitions  # Expanded lattice vectors
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    
    # Define Lennard-Jones potential
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 2.5 * sigma  # Å
    lj_potential = LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)
    
    # Create ElasticConstantsCalculator instance
    elastic_calculator = ElasticConstantsCalculator(
        cell=cell, potential=lj_potential, delta=1e-3, optimizer_type="BFGS"
    )
    
    # Calculate elastic constants
    C_in_GPa = elastic_calculator.calculate_elastic_constants()
    
    # Output computed elastic constants
    logger.debug("Computed Elastic Constants (GPa):")
    logger.debug(C_in_GPa)
    
    # Check that C_in_GPa is a 6x6 matrix
    assert C_in_GPa.shape == (6, 6), "弹性常数矩阵形状不匹配。"
    
    # Check symmetry
    symmetry = np.allclose(C_in_GPa, C_in_GPa.T, atol=1e-3)
    logger.debug(f"Symmetry check: {symmetry}")
    assert symmetry, "弹性常数矩阵不是对称的。"
    
    # Check that diagonal elements are positive
    for i in range(6):
        logger.debug(f"C_in_GPa[{i},{i}] = {C_in_GPa[i, i]}")
        assert C_in_GPa[i, i] > 0, f"弹性常数 C[{i},{i}] 不是正值。"
    
    # Optionally, check off-diagonal elements are within reasonable ranges
    for i in range(6):
        for j in range(i+1, 6):
            logger.debug(f"C_in_GPa[{i},{j}] = {C_in_GPa[i, j]}")
            assert 0.0 <= C_in_GPa[i, j] <= 100.0, f"弹性常数 C[{i},{j}] 不在合理范围内。"
    
    # Further check the elastic constants are within expected ranges
    # For example, diagonal elements (C11, C22, ...) typically 50-100 GPa
    # Shear moduli (C44, C55, C66) typically 10-30 GPa
    # Adjust these ranges based on your system's properties
    for i in range(3):
        logger.debug(f"C_in_GPa[{i},{i}] = {C_in_GPa[i, i]}")
        assert 50.0 <= C_in_GPa[i, i] <= 100.0, f"C[{i},{i}] = {C_in_GPa[i, i]} GPa 不在预期范围内。"
    for i in range(3,6):
        logger.debug(f"C_in_GPa[{i},{i}] = {C_in_GPa[i, i]}")
        assert 10.0 <= C_in_GPa[i, i] <= 30.0, f"C[{i},{i}] = {C_in_GPa[i, i]} GPa 不在预期范围内。"

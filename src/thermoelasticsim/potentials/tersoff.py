#!/usr/bin/env python3
r"""
ThermoElasticSim - Tersoff 势模块

.. moduleauthor:: Gilbert Young

Tersoff 多体势用于共价材料（Si、C 等），本实现提供 C(1988) 版本的参数化，
并通过 C++ 后端计算能量、力与维里张量（支持可选的平衡键长 shift=delta）。

References
----------
- Tersoff, J. (1988). Empirical Interatomic Potential for Carbon, with Applications to Amorphous Carbon.
  Phys. Rev. Lett. 61, 2879–2882.
- Tersoff, J. (1988). New empirical approach for the structure and energy of covalent systems.
  Phys. Rev. B 37, 6991–7000.
"""

import logging

import numpy as np

from thermoelasticsim.core.structure import Cell
from thermoelasticsim.interfaces.cpp_interface import CppInterface
from thermoelasticsim.utils.utils import NeighborList

from .base import Potential

logger = logging.getLogger(__name__)


class TersoffC1988Potential(Potential):
    r"""Tersoff (1988, C) 多体势实现（C++ 后端）。

    Parameters
    ----------
    params : dict | None, optional
        自定义参数；若为 None 则使用 C++ 内置的 1988 年碳默认参数。
    delta : float, optional
        LAMMPS 风格的可选键长 shift（r → r + delta），默认 0.0。

    Notes
    -----
    - 单位：能量 eV，长度 Å，力 eV/Å。
    - 截断半径自动取 :code:`R + D`（由参数决定）。
    - 该实现直接使用 C++ 多体解析力，维里按 :math:`-\sum_i r_i \otimes F_i` 计算。
    """

    def __init__(self, params: dict | None = None, delta: float = 0.0):
        self.params = params  # 自定义参数（可选）
        self.delta = float(delta)
        self.cpp_interface = CppInterface("tersoff_c1988")

        # 截断：若用户自定义参数，则用其 R+D；否则用 C++ 默认值 R+D=2.10 Å
        if isinstance(params, dict) and all(k in params for k in ("R", "D")):
            cutoff = float(params["R"]) + float(params["D"])  # type: ignore
        else:
            cutoff = 1.95 + 0.15  # C1988 默认
        super().__init__(parameters=(params or {}), cutoff=cutoff)

    def _cpp_args(self):
        # 自定义参数路径用通用接口；否则用 C(1988) 默认接口
        if isinstance(self.params, dict) and self.params:
            p = self.params
            return dict(
                A=float(p["A"]),
                B=float(p["B"]),
                lambda1=float(p["lambda1"]),
                lambda2=float(p["lambda2"]),
                lambda3=float(p.get("lambda3", 0.0)),
                beta=float(p["beta"]),
                n=float(p["n"]),
                c=float(p["c"]),
                d=float(p["d"]),
                h=float(p["h"]),
                R=float(p["R"]),
                D=float(p["D"]),
                m=int(p.get("m", 3)),
                shift_flag=(abs(self.delta) > 0.0),
                delta=self.delta,
            )
        else:
            # 标识：使用 C(1988) 内置参数
            return None

    def calculate_forces(
        self, cell: Cell, neighbor_list: NeighborList | None = None
    ) -> None:
        """计算并更新原子受力。

        Parameters
        ----------
        cell : Cell
            原子系统
        neighbor_list : NeighborList | None
            邻居列表（未使用）
        """
        num_atoms = cell.num_atoms
        positions = np.ascontiguousarray(
            cell.get_positions(), dtype=np.float64
        ).flatten()
        lattice_vectors = np.ascontiguousarray(
            cell.lattice_vectors, dtype=np.float64
        ).flatten()
        forces = np.zeros_like(positions, dtype=np.float64)

        args = self._cpp_args()
        if args is None:
            self.cpp_interface.calculate_tersoff_c1988_forces(
                num_atoms,
                positions,
                lattice_vectors,
                forces,
                shift_flag=(abs(self.delta) > 0.0),
                delta=self.delta,
            )
        else:
            self.cpp_interface.calculate_tersoff_forces(
                num_atoms, positions, lattice_vectors, forces, **args
            )

        f = forces.reshape((num_atoms, 3))
        for i, atom in enumerate(cell.atoms):
            atom.force = f[i]
        try:
            max_f = float(np.max(np.linalg.norm(f, axis=1))) if num_atoms > 0 else 0.0
            sum_f = np.sum(f, axis=0)
            norm_sum_f = float(np.linalg.norm(sum_f))
            logger.debug(
                "TersoffC1988: 力统计 max|F|=%.3e eV/Å, |ΣF|=%.3e eV/Å",
                max_f,
                norm_sum_f,
            )
        except Exception:
            pass

    def calculate_energy(
        self, cell: Cell, neighbor_list: NeighborList | None = None
    ) -> float:
        """计算系统总能量。

        Parameters
        ----------
        cell : Cell
            原子系统
        neighbor_list : NeighborList | None
            邻居列表（未使用）

        Returns
        -------
        float
            总能量(eV)
        """
        num_atoms = cell.num_atoms
        positions = np.ascontiguousarray(
            cell.get_positions(), dtype=np.float64
        ).flatten()
        lattice_vectors = np.ascontiguousarray(
            cell.lattice_vectors, dtype=np.float64
        ).flatten()

        args = self._cpp_args()
        if args is None:
            energy = self.cpp_interface.calculate_tersoff_c1988_energy(
                num_atoms,
                positions,
                lattice_vectors,
                shift_flag=(abs(self.delta) > 0.0),
                delta=self.delta,
            )
        else:
            energy = self.cpp_interface.calculate_tersoff_energy(
                num_atoms, positions, lattice_vectors, **args
            )
        e = float(energy)
        logger.debug("TersoffC1988: 总能量 E=%.6e eV", e)
        return e

    # 供 StressCalculator 专用（可选调用）
    def _calculate_virial_tensor(self, cell: Cell) -> np.ndarray:
        num_atoms = cell.num_atoms
        positions = np.ascontiguousarray(
            cell.get_positions(), dtype=np.float64
        ).flatten()
        lattice_vectors = np.ascontiguousarray(
            cell.lattice_vectors, dtype=np.float64
        ).flatten()
        args = self._cpp_args()
        if args is None:
            vir = self.cpp_interface.calculate_tersoff_c1988_virial(
                num_atoms,
                positions,
                lattice_vectors,
                shift_flag=(abs(self.delta) > 0.0),
                delta=self.delta,
            )
        else:
            vir = self.cpp_interface.calculate_tersoff_virial(
                num_atoms, positions, lattice_vectors, **args
            )
        try:
            frob = float(np.linalg.norm(vir))
            # Python 回退近似：-Σ r ⊗ F（用于诊断）
            try:
                self.calculate_forces(cell)
                pos = cell.get_positions()
                frc = cell.get_forces()
                vir_py = np.zeros((3, 3), dtype=np.float64)
                for i in range(pos.shape[0]):
                    vir_py -= np.outer(pos[i], frc[i])
                frob_py = float(np.linalg.norm(vir_py))
                logger.debug(
                    "TersoffC1988: 维里张量 Frobenius=%.3e | 近似(Σr⊗F)=%.3e",
                    frob,
                    frob_py,
                )
            except Exception:
                logger.debug("TersoffC1988: 维里诊断失败(Σr⊗F)")
            else:
                pass
        except Exception:
            pass
        return vir

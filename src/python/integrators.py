# src/python/integrators.py

import numpy as np


class Integrator:
    """
    @class Integrator
    @brief 积分器基类
    """

    def integrate(self, cell, potential, dt):
        raise NotImplementedError


class VelocityVerletIntegrator(Integrator):
    """
    @class VelocityVerletIntegrator
    @brief 速度 Verlet 积分器的实现
    """

    def integrate(self, cell, potential, dt):
        atoms = cell.atoms
        # 第一半步：更新位置
        for atom in atoms:
            atom.position += atom.velocity * dt + 0.5 * atom.force / atom.mass * dt**2
            # 应用周期性边界条件
            atom.position = cell.apply_periodic_boundary(atom.position)
        # 保存旧的力
        forces_old = [atom.force.copy() for atom in atoms]
        # 计算新的力
        potential.calculate_forces(cell)
        # 第二半步：更新速度
        for atom, force_old in zip(atoms, forces_old):
            atom.velocity += 0.5 * (atom.force + force_old) / atom.mass * dt


class RK4Integrator(Integrator):
    """
    @class RK4Integrator
    @brief 四阶 Runge-Kutta 积分器的实现
    """

    def integrate(self, cell, potential, dt):
        def get_state():
            return np.concatenate(
                [atom.position for atom in cell.atoms]
                + [atom.velocity for atom in cell.atoms]
            )

        def set_state(state):
            num_atoms = len(cell.atoms)
            positions = state[: 3 * num_atoms].reshape((num_atoms, 3))
            velocities = state[3 * num_atoms :].reshape((num_atoms, 3))
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i]
                atom.velocity = velocities[i]

        def compute_derivatives(state):
            # Set the state
            num_atoms = len(cell.atoms)
            positions = state[: 3 * num_atoms].reshape((num_atoms, 3))
            velocities = state[3 * num_atoms :].reshape((num_atoms, 3))
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i]
                atom.velocity = velocities[i]
            # Compute forces
            potential.calculate_forces(cell)
            # Compute derivatives
            derivatives = np.zeros_like(state)
            for i, atom in enumerate(cell.atoms):
                derivatives[3 * i : 3 * i + 3] = atom.velocity
                derivatives[3 * num_atoms + 3 * i : 3 * num_atoms + 3 * i + 3] = (
                    atom.force / atom.mass
                )
            return derivatives

        state = get_state()
        k1 = compute_derivatives(state)
        k2 = compute_derivatives(state + 0.5 * dt * k1)
        k3 = compute_derivatives(state + 0.5 * dt * k2)
        k4 = compute_derivatives(state + dt * k3)
        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        set_state(new_state)

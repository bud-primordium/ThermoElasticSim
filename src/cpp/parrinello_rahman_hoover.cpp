/**
 * @file parrinello_rahman_hoover.cpp
 * @brief Parrinello-Rahman-Hoover 恒压器的实现
 *
 * 该文件实现了 Parrinello-Rahman-Hoover 恒压器，主要用于分子动力学模拟中的压力控制。
 * 通过更新晶格矢量和原子速度，实现对系统压力的调节。
 *
 * @author Gilbert Young
 * @date 2024-10-20
 */

#include <cmath>
#include <vector>

extern "C"
{

    void remove_com_motion(
        int num_atoms,
        const double *masses,
        double *velocities)
    {
        // Calculate total mass and center of mass velocity
        double total_mass = 0.0;
        double com_vel[3] = {0.0, 0.0, 0.0};

        for (int i = 0; i < num_atoms; ++i)
        {
            total_mass += masses[i];
            for (int j = 0; j < 3; ++j)
            {
                com_vel[j] += masses[i] * velocities[3 * i + j];
            }
        }

        // Normalize COM velocity
        for (int j = 0; j < 3; ++j)
        {
            com_vel[j] /= total_mass;
        }

        // Remove COM motion
        for (int i = 0; i < num_atoms; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                velocities[3 * i + j] -= com_vel[j];
            }
        }
    }

    void calculate_kinetic_stress(
        int num_atoms,
        const double *masses,
        const double *velocities,
        const double volume,
        double *kinetic_stress)
    {
        // Initialize kinetic stress tensor
        for (int i = 0; i < 9; ++i)
        {
            kinetic_stress[i] = 0.0;
        }

        // Calculate kinetic contribution to stress
        for (int i = 0; i < num_atoms; ++i)
        {
            for (int a = 0; a < 3; ++a)
            {
                for (int b = 0; b < 3; ++b)
                {
                    kinetic_stress[3 * a + b] += masses[i] *
                                                 velocities[3 * i + a] * velocities[3 * i + b];
                }
            }
        }

        // Scale by volume
        for (int i = 0; i < 9; ++i)
        {
            kinetic_stress[i] /= volume;
        }
    }

    void parrinello_rahman_hoover(
        double dt,
        int num_atoms,
        const double *masses,
        double *velocities,
        const double *forces,
        double *lattice_vectors,       // 3x3 matrix, row-major
        double *xi,                    // Thermostat variable array length 6
        const double *Q,               // Thermostat mass parameters array length 6
        const double *target_pressure, // 3x3 matrix
        double *virial_stress,         // 3x3 matrix
        double W)                      // Cell mass parameter
    {
        double volume = lattice_vectors[0] * (lattice_vectors[4] * lattice_vectors[8] -
                                              lattice_vectors[5] * lattice_vectors[7]) -
                        lattice_vectors[1] * (lattice_vectors[3] * lattice_vectors[8] -
                                              lattice_vectors[5] * lattice_vectors[6]) +
                        lattice_vectors[2] * (lattice_vectors[3] * lattice_vectors[7] -
                                              lattice_vectors[4] * lattice_vectors[6]);

        // Calculate stress tensors
        double kinetic_stress[9];
        calculate_kinetic_stress(num_atoms, masses, velocities, volume, kinetic_stress);

        // Calculate pressure tensor: P = (K + V)/V
        double pressure[9];
        for (int i = 0; i < 9; ++i)
        {
            pressure[i] = (kinetic_stress[i] + virial_stress[i]) / volume;
        }

        // Update cell velocities (represented by xi)
        for (int i = 0; i < 6; ++i)
        {
            double delta_P = pressure[i] - target_pressure[i];
            xi[i] += (volume * delta_P / W) * dt;
        }

        // Update lattice vectors using cell velocities
        double new_lattice[9];
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                new_lattice[3 * i + j] = lattice_vectors[3 * i + j];
                for (int k = 0; k < 3; ++k)
                {
                    new_lattice[3 * i + j] += dt * xi[3 * i + k] * lattice_vectors[3 * k + j];
                }
            }
        }

        // Copy new lattice vectors
        for (int i = 0; i < 9; ++i)
        {
            lattice_vectors[i] = new_lattice[i];
        }

        // Update atomic velocities
        for (int i = 0; i < num_atoms; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                velocities[3 * i + j] += (forces[3 * i + j] / masses[i]) * dt;
                for (int k = 0; k < 3; ++k)
                {
                    velocities[3 * i + j] -= xi[3 * j + k] * velocities[3 * i + k] * dt;
                }
            }
        }

        // Remove center of mass motion
        remove_com_motion(num_atoms, masses, velocities);
    }

} // extern "C"
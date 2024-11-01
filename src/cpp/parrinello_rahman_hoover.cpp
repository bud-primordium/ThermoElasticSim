/**
 * @file parrinello_rahman_hoover.cpp
 * @brief Parrinello-Rahman-Hoover barostat implementation
 *
 * This implementation uses the total stress tensor calculated by StressCalculator
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

    void parrinello_rahman_hoover(
        double dt,
        int num_atoms,
        const double *masses,
        double *velocities,
        const double *forces,
        double *lattice_vectors,       // 3x3 matrix, row-major
        double *xi,                    // Thermostat variable array length 9
        const double *Q,               // Thermostat mass parameters array length 9
        const double *total_stress,    // Current total stress tensor from StressCalculator (9 components)
        const double *target_pressure, // Target pressure tensor (9 components)
        double W)                      // Cell mass parameter
    {
        double volume = lattice_vectors[0] * (lattice_vectors[4] * lattice_vectors[8] -
                                              lattice_vectors[5] * lattice_vectors[7]) -
                        lattice_vectors[1] * (lattice_vectors[3] * lattice_vectors[8] -
                                              lattice_vectors[5] * lattice_vectors[6]) +
                        lattice_vectors[2] * (lattice_vectors[3] * lattice_vectors[7] -
                                              lattice_vectors[4] * lattice_vectors[6]);

        // Update cell velocities (represented by xi)
        // Now using the total stress directly from StressCalculator
        for (int i = 0; i < 9; ++i)
        {
            double delta_P = total_stress[i] - target_pressure[i];
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

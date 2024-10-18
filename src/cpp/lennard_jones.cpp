// src/cpp/lennard_jones.cpp

#include <cmath>
#include <vector>
#include <iostream>

extern "C"
{
    /**
     * @brief 计算 Lennard-Jones 力，使用最小镜像法处理周期性边界条件
     *
     * @param num_atoms 原子数量
     * @param positions 原子位置数组（长度为 3*num_atoms）
     * @param forces 输出的力数组（长度为 3*num_atoms）
     * @param epsilon Lennard-Jones 势参数 ε，单位为 eV
     * @param sigma Lennard-Jones 势参数 σ，单位为 Å
     * @param cutoff 截断半径，单位为 Å
     * @param box_lengths 模拟盒子在每个维度的长度（长度为 3）
     */
    void calculate_forces(
        int num_atoms,
        const double *positions,
        double *forces,
        double epsilon,
        double sigma,
        double cutoff,
        const double *box_lengths // 长度为 3
    )
    {
        // 清零力数组
        for (int i = 0; i < 3 * num_atoms; ++i)
        {
            forces[i] = 0.0;
        }

        double cutoff_sq = cutoff * cutoff;
        // 计算力的截断修正
        double force_shift = 48.0 * epsilon * (pow(sigma / cutoff, 12) - 0.5 * pow(sigma / cutoff, 6));

        // 力的计算
        for (int i = 0; i < num_atoms; ++i)
        {
            const double *ri = &positions[3 * i];
            for (int j = i + 1; j < num_atoms; ++j)
            {
                const double *rj = &positions[3 * j];
                double rij[3];
                for (int k = 0; k < 3; ++k)
                {
                    rij[k] = ri[k] - rj[k];
                    // 最小镜像法
                    if (rij[k] > 0.5 * box_lengths[k])
                        rij[k] -= box_lengths[k];
                    else if (rij[k] < -0.5 * box_lengths[k])
                        rij[k] += box_lengths[k];
                }

                double r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];

                if (r2 < cutoff_sq)
                {
                    double r = sqrt(r2);
                    double r2_inv = 1.0 / r2;
                    double r6_inv = pow(sigma * sigma * r2_inv, 3); // (sigma/r)^6
                    double force_scalar = 48.0 * epsilon * r6_inv * (r6_inv - 0.5) * r2_inv;

                    // 力的截断修正
                    force_scalar -= force_shift;

                    for (int k = 0; k < 3; ++k)
                    {
                        double fij = force_scalar * rij[k];
                        forces[3 * i + k] += fij;
                        forces[3 * j + k] -= fij;
                    }
                }
            }
        }
    }

    /**
     * @brief 计算 Lennard-Jones 势能，使用最小镜像法处理周期性边界条件
     *
     * @param num_atoms 原子数量
     * @param positions 原子位置数组（长度为 3*num_atoms）
     * @param epsilon Lennard-Jones 势参数 ε，单位为 eV
     * @param sigma Lennard-Jones 势参数 σ，单位为 Å
     * @param cutoff 截断半径，单位为 Å
     * @param box_lengths 模拟盒子在每个维度的长度（长度为 3）
     * @return 总 Lennard-Jones 势能，单位为 eV
     */
    double calculate_energy(
        int num_atoms,
        const double *positions,
        double epsilon,
        double sigma,
        double cutoff,
        const double *box_lengths // 长度为 3
    )
    {
        double energy = 0.0;
        double cutoff_sq = cutoff * cutoff;
        double shift = 4.0 * epsilon * (pow(sigma / cutoff, 12) - pow(sigma / cutoff, 6));

        for (int i = 0; i < num_atoms; ++i)
        {
            const double *ri = &positions[3 * i];
            for (int j = i + 1; j < num_atoms; ++j)
            {
                const double *rj = &positions[3 * j];
                double rij[3];
                for (int k = 0; k < 3; ++k)
                {
                    rij[k] = ri[k] - rj[k];
                    // 最小镜像法
                    if (rij[k] > 0.5 * box_lengths[k])
                        rij[k] -= box_lengths[k];
                    else if (rij[k] < -0.5 * box_lengths[k])
                        rij[k] += box_lengths[k];
                }

                double r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];

                if (r2 < cutoff_sq)
                {
                    double r2_inv = 1.0 / r2;
                    double r6_inv = pow(sigma * sigma * r2_inv, 3); // (sigma/r)^6
                    double potential = 4.0 * epsilon * (r6_inv * r6_inv - r6_inv) - shift;
                    energy += potential;
                }
            }
        }

        return energy;
    }
}

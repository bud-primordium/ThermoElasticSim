/**
 * @file lennard_jones.cpp
 * @brief Lennard-Jones 势计算的实现，包括截断区域的多项式处理
 *
 * 该文件包含用于 Lennard-Jones 势力和能量计算的函数，包括处理周期性边界条件和截断修正。
 *
 * @author Gilbert Young
 * @date 2024-10-20
 */

#include <cmath>
#include <vector>

extern "C"
{
    /**
     * @brief 使用三次多项式计算截断区域的系数 a, b, c, d
     *
     * 满足以下条件：
     * 1. V(r0) = V0
     * 2. V'(r0) = F0
     * 3. V(cutoff) = 0
     * 4. V'(cutoff) = 0
     *
     * @param cutoff 截断半径
     * @param r0 过渡起点
     * @param sigma LJ势参数 σ
     * @param epsilon LJ势参数 ε
     * @param a 三次项系数
     * @param b 二次项系数
     * @param c 一次项系数
     * @param d 常数项
     */
    void solve_cutoff_cubic(
        double cutoff, double r0, double sigma, double epsilon,
        double &a, double &b, double &c, double &d)
    {
        // 计算 r0 处的势能 V0 和力 F0
        double r0_inv = 1.0 / r0;
        double sigma_r0_inv = sigma * r0_inv;
        double sigma_r0_inv_6 = pow(sigma_r0_inv, 6);
        double sigma_r0_inv_12 = pow(sigma_r0_inv, 12);
        double V0 = 4.0 * epsilon * (sigma_r0_inv_12 - sigma_r0_inv_6);
        double F0 = 24.0 * epsilon * (2.0 * sigma_r0_inv_12 - sigma_r0_inv_6) * r0_inv;

        // 按照 Mathematica 的解更新三次多项式系数 a, b, c, d
        double denom = pow(r0 - cutoff, 3);
        a = -(F0 * r0 - F0 * cutoff + 2.0 * V0) / denom;
        b = -((-F0 * pow(r0, 2)) - F0 * r0 * cutoff + 2.0 * F0 * pow(cutoff, 2) - 3.0 * r0 * V0 - 3.0 * cutoff * V0) / denom;
        c = -(cutoff * (2.0 * F0 * pow(r0, 2) - F0 * r0 * cutoff - F0 * pow(cutoff, 2) + 6.0 * r0 * V0)) / denom;
        d = -((-F0 * pow(r0, 2) * pow(cutoff, 2)) + F0 * r0 * pow(cutoff, 3) - 3.0 * r0 * pow(cutoff, 2) * V0 + pow(cutoff, 3) * V0) / denom;
    }

    void calculate_lj_forces(
        int num_atoms,
        const double *positions,
        double *forces,
        double epsilon,
        double sigma,
        double cutoff,
        const double *box_lengths,
        const int *neighbor_pairs,
        int num_pairs)
    {
        // 清零力数组
        for (int i = 0; i < 3 * num_atoms; ++i)
        {
            forces[i] = 0.0;
        }

        double cutoff_sq = cutoff * cutoff;
        double r0 = 0.9 * cutoff; // 统一设置平滑过渡的起点

        // 预计算三次多项式系数
        double a, b, c, d;
        solve_cutoff_cubic(cutoff, r0, sigma, epsilon, a, b, c, d);

        // 力的计算
        for (int p = 0; p < num_pairs; ++p)
        {
            int i = neighbor_pairs[2 * p];
            int j = neighbor_pairs[2 * p + 1];
            double rij[3];
            for (int k = 0; k < 3; ++k)
            {
                rij[k] = positions[3 * i + k] - positions[3 * j + k];
                // 最小镜像法已经在 Python/NeighborList 中处理（对LJ而言）
            }

            double r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];

            if (r2 < cutoff_sq)
            {
                double r = sqrt(r2);
                double force_scalar;

                if (r < r0)
                {
                    double r_inv = 1.0 / r;
                    double r6_inv = pow(sigma * r_inv, 6); // (σ/r)^6
                    force_scalar = 24.0 * epsilon * (2.0 * r6_inv * r6_inv - r6_inv) * r_inv;
                }
                else
                {
                    // 使用三次多项式过渡区的力 F(r) = -dV/dr = -(3a*r^2 + 2b*r + c)
                    force_scalar = -(3.0 * a * r * r + 2.0 * b * r + c);
                }

                for (int k = 0; k < 3; ++k)
                {
                    double fij = force_scalar * rij[k] / r;
                    forces[3 * i + k] += fij;
                    forces[3 * j + k] -= fij;
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
    double calculate_lj_energy(
        int num_atoms,
        const double *positions,
        double epsilon,
        double sigma,
        double cutoff,
        const double *box_lengths,
        const int *neighbor_pairs,
        int num_pairs)
    {
        double energy = 0.0;
        double cutoff_sq = cutoff * cutoff;
        double r0 = 0.9 * cutoff; // 统一设置平滑过渡的起点

        // 预计算三次多项式系数
        double a, b, c, d;
        solve_cutoff_cubic(cutoff, r0, sigma, epsilon, a, b, c, d);

        // 势能的计算
        for (int p = 0; p < num_pairs; ++p)
        {
            int i = neighbor_pairs[2 * p];
            int j = neighbor_pairs[2 * p + 1];
            double rij[3];
            for (int k = 0; k < 3; ++k)
            {
                rij[k] = positions[3 * i + k] - positions[3 * j + k];
                // 最小镜像法已经在 Python 中处理
            }

            double r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];

            if (r2 < cutoff_sq)
            {
                double r = sqrt(r2);
                double potential;

                if (r < r0)
                {
                    double r_inv = 1.0 / r;
                    double r6_inv = pow(sigma * r_inv, 6); // (σ/r)^6
                    potential = 4.0 * epsilon * (r6_inv * r6_inv - r6_inv);
                }
                else
                {
                    // 使用三次多项式过渡区的势能 V(r) = a*r^3 + b*r^2 + c*r + d
                    potential = a * pow(r, 3) + b * pow(r, 2) + c * r + d;
                }

                energy += potential;
            }
        }

        return energy;
    }
}

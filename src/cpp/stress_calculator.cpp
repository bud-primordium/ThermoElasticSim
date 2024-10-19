/**
 * @file stress_calculator.cpp
 * @brief 应力张量计算的实现
 *
 * 该文件实现了基于动能和势能贡献的应力张量计算，用于分子动力学模拟。
 * 包括原子速度和位置的贡献，以及通过力和位置的计算。
 *
 * @author Gilbert Young
 * @date 2024-10-20
 */

#include <cmath>
#include <vector>

extern "C"
{
    /**
     * @brief 计算应力张量
     *
     * @param num_atoms 原子数量
     * @param positions 原子位置数组（长度为 3*num_atoms）
     * @param velocities 原子速度数组（长度为 3*num_atoms）
     * @param forces 原子力数组（长度为 3*num_atoms）
     * @param masses 原子质量数组（长度为 num_atoms）
     * @param volume 系统体积
     * @param box_lengths 模拟盒子在每个维度的长度（长度为 3）
     * @param stress_tensor 输出的应力张量（长度为 9，按行主序存储 3x3 矩阵）
     */
    void compute_stress(
        int num_atoms,
        const double *positions,
        const double *velocities,
        const double *forces,
        const double *masses,
        double volume,
        const double *box_lengths,
        double *stress_tensor // 输出
    )
    {
        // 初始化应力张量
        for (int i = 0; i < 9; ++i)
        {
            stress_tensor[i] = 0.0;
        }

        // 动能贡献
        for (int i = 0; i < num_atoms; ++i)
        {
            double m = masses[i];
            const double *v = &velocities[3 * i];
            for (int alpha = 0; alpha < 3; ++alpha)
            {
                for (int beta = 0; beta < 3; ++beta)
                {
                    stress_tensor[3 * alpha + beta] += m * v[alpha] * v[beta];
                }
            }
        }

        // 势能贡献
        for (int i = 0; i < num_atoms; ++i)
        {
            const double *ri = &positions[3 * i];
            const double *fi = &forces[3 * i];
            for (int alpha = 0; alpha < 3; ++alpha)
            {
                for (int beta = 0; beta < 3; ++beta)
                {
                    stress_tensor[3 * alpha + beta] += ri[beta] * fi[alpha];
                }
            }
        }

        // 归一化
        for (int i = 0; i < 9; ++i)
        {
            stress_tensor[i] = (stress_tensor[i]) / volume;
        }
    }
}

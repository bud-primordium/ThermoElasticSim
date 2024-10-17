// src/cpp/stress_calculator.cpp

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
     * @param epsilon Lennard-Jones 势参数 ε，单位为 eV
     * @param sigma Lennard-Jones 势参数 σ，单位为 Å
     * @param cutoff 截断半径，单位为 Å
     * @param lattice_vectors 晶格矢量（长度为 9，按列存储 3x3 矩阵）
     * @param stress_tensor 输出的应力张量（长度为 9，按行主序存储 3x3 矩阵）
     */
    void compute_stress(
        int num_atoms,
        const double *positions,
        const double *velocities,
        const double *forces,
        const double *masses,
        double volume,
        double epsilon,
        double sigma,
        double cutoff,
        const double *lattice_vectors,
        double *stress_tensor // 输出
    )
    {
        // 初始化应力张量
        for (int i = 0; i < 9; ++i)
        {
            stress_tensor[i] = 0.0;
        }

        // 构建晶格矩阵 H 和逆矩阵 H_inv
        double H[3][3];
        for (int i = 0; i < 3; ++i)
        {
            H[0][i] = lattice_vectors[3 * i];
            H[1][i] = lattice_vectors[3 * i + 1];
            H[2][i] = lattice_vectors[3 * i + 2];
        }
        // 计算 H_inv
        double det = H[0][0] * (H[1][1] * H[2][2] - H[1][2] * H[2][1]) -
                     H[0][1] * (H[1][0] * H[2][2] - H[1][2] * H[2][0]) +
                     H[0][2] * (H[1][0] * H[2][1] - H[1][1] * H[2][0]);
        double H_inv[3][3];
        H_inv[0][0] = (H[1][1] * H[2][2] - H[1][2] * H[2][1]) / det;
        H_inv[0][1] = (H[0][2] * H[2][1] - H[0][1] * H[2][2]) / det;
        H_inv[0][2] = (H[0][1] * H[1][2] - H[0][2] * H[1][1]) / det;
        H_inv[1][0] = (H[1][2] * H[2][0] - H[1][0] * H[2][2]) / det;
        H_inv[1][1] = (H[0][0] * H[2][2] - H[0][2] * H[2][0]) / det;
        H_inv[1][2] = (H[0][2] * H[1][0] - H[0][0] * H[1][2]) / det;
        H_inv[2][0] = (H[1][0] * H[2][1] - H[1][1] * H[2][0]) / det;
        H_inv[2][1] = (H[0][1] * H[2][0] - H[0][0] * H[2][1]) / det;
        H_inv[2][2] = (H[0][0] * H[1][1] - H[0][1] * H[1][0]) / det;

        // 动能项
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

        // 势能项
        for (int i = 0; i < num_atoms; ++i)
        {
            const double *ri = &positions[3 * i];
            for (int j = i + 1; j < num_atoms; ++j)
            {
                const double *rj = &positions[3 * j];
                double rij[3];
                // 计算 rij = rj - ri
                for (int k = 0; k < 3; ++k)
                {
                    rij[k] = rj[k] - ri[k];
                }

                // 转换到分数坐标 s = H_inv * rij
                double s[3];
                for (int k = 0; k < 3; ++k)
                {
                    s[k] = H_inv[k][0] * rij[0] + H_inv[k][1] * rij[1] + H_inv[k][2] * rij[2];
                    // 映射到 [-0.5, 0.5]
                    s[k] -= round(s[k]);
                }

                // 转换回笛卡尔坐标 rij = H * s
                for (int k = 0; k < 3; ++k)
                {
                    rij[k] = H[0][k] * s[0] + H[1][k] * s[1] + H[2][k] * s[2];
                }

                double r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
                double r = sqrt(r2);

                if (r < cutoff)
                {
                    double sr = sigma / r;
                    double sr6 = pow(sr, 6);
                    double sr12 = sr6 * sr6;
                    double force_scalar = 24.0 * epsilon * (2.0 * sr12 - sr6) / r;
                    double fij[3];
                    for (int k = 0; k < 3; ++k)
                    {
                        fij[k] = force_scalar * rij[k] / r;
                    }
                    for (int alpha = 0; alpha < 3; ++alpha)
                    {
                        for (int beta = 0; beta < 3; ++beta)
                        {
                            stress_tensor[3 * alpha + beta] += rij[alpha] * fij[beta];
                        }
                    }
                }
            }
        }

        // 归一化并取负号
        for (int i = 0; i < 9; ++i)
        {
            stress_tensor[i] = -stress_tensor[i] / volume;
        }
    }
}

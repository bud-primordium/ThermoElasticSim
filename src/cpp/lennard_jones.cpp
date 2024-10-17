// src/cpp/lennard_jones.cpp

#include <cmath>
#include <vector>
#include <iostream>

extern "C"
{
    /**
     * @brief 计算 Lennard-Jones 力
     *
     * @param num_atoms 原子数量
     * @param positions 原子位置数组（长度为 3*num_atoms）
     * @param forces 输出的力数组（长度为 3*num_atoms）
     * @param epsilon Lennard-Jones 势参数 ε，单位为 eV
     * @param sigma Lennard-Jones 势参数 σ，单位为 Å
     * @param cutoff 截断半径，单位为 Å
     * @param lattice_vectors 晶格矢量（长度为 9，按列存储 3x3 矩阵）
     */
    void calculate_forces(
        int num_atoms,
        const double *positions,
        double *forces,
        double epsilon,
        double sigma,
        double cutoff,
        const double *lattice_vectors // 9 个元素：H = [a1, a2, a3] 作为列
    )
    {
        // 清零力数组
        for (int i = 0; i < 3 * num_atoms; ++i)
        {
            forces[i] = 0.0;
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
        if (det == 0)
        {
            std::cerr << "错误：晶格矩阵的行列式为零，无法求逆。" << std::endl;
            return;
        }
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

        // 预计算在截断距离处的力，以用于力的偏移
        double sr_cutoff = sigma / cutoff;
        double sr6_cutoff = pow(sr_cutoff, 6);
        double sr12_cutoff = sr6_cutoff * sr6_cutoff;
        double F_LJ_cutoff = 24.0 * epsilon * (2.0 * sr12_cutoff - sr6_cutoff) / cutoff;

        // 力的计算
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

                // 添加极小距离保护
                if (r < 1e-8)
                {
                    std::cerr << "警告：原子 " << i << " 和原子 " << j << " 太近 (r=" << r << ")，跳过力的计算。" << std::endl;
                    continue;
                }

                if (r < cutoff)
                {
                    double sr = sigma / r;
                    double sr6 = pow(sr, 6);
                    double sr12 = sr6 * sr6;
                    double F_LJ = 24.0 * epsilon * (2.0 * sr12 - sr6) / r;

                    // 力的偏移，使得在截断距离处力为零
                    double force_scalar = F_LJ - F_LJ_cutoff;

                    // 应用力
                    double fij[3];
                    for (int k = 0; k < 3; ++k)
                    {
                        fij[k] = force_scalar * rij[k] / r;
                        forces[3 * i + k] += fij[k];
                        forces[3 * j + k] -= fij[k];
                    }

                    // 调试输出
                    // std::cout << "原子 " << i << " - 原子 " << j << " 力：[" << fij[0] << ", " << fij[1] << ", " << fij[2] << "]" << std::endl;
                }
            }
        }
    }
}

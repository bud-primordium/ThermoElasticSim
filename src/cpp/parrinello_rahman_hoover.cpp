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
    /**
     * @brief 应用 Parrinello-Rahman-Hoover 恒压器
     *
     * @param dt 时间步长
     * @param num_atoms 原子数量
     * @param masses 原子质量数组（长度为 num_atoms）
     * @param velocities 原子速度数组（长度为 3*num_atoms）
     * @param forces 原子力数组（长度为 3*num_atoms）
     * @param lattice_vectors 当前晶格矢量（长度为 9, row-major order）
     * @param xi 热浴变量数组（长度为 6}
     * @param Q 热浴质量参数数组（长度为 6}
     * @param target_pressure 目标压力
     */
    void parrinello_rahman_hoover(
        double dt,
        int num_atoms,
        const double *masses,
        double *velocities,
        const double *forces,
        double *lattice_vectors, // 3x3 matrix, row-major
        double *xi,
        const double *Q,
        double target_pressure)
    {
        // 这个是一个简单的实现示例，实际的 PRH 恒温器更复杂
        // 这里只提供一个简单的缩放实现

        // 计算当前压力（简化示例）
        double current_pressure = 0.0;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                current_pressure += lattice_vectors[i * 3 + j] * forces[j * 3 + i];
            }
        }
        current_pressure /= 3.0;

        // 计算压力差
        double delta_P = current_pressure - target_pressure;

        // 更新热浴变量
        for (int i = 0; i < 6; ++i)
        {
            xi[i] += (delta_P)*dt / Q[i];
        }

        // 调整晶胞的晶格向量
        for (int i = 0; i < 9; ++i)
        {
            lattice_vectors[i] *= exp(-xi[i % 6] * dt); // 简单缩放
        }

        // 更新原子速度（简化示例）
        for (int i = 0; i < num_atoms; ++i)
        {
            velocities[3 * i] *= exp(-xi[0] * dt);
            velocities[3 * i + 1] *= exp(-xi[1] * dt);
            velocities[3 * i + 2] *= exp(-xi[2] * dt);
        }
    }
}

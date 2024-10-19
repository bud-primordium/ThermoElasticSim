/**
 * @file nose_hoover_chain.cpp
 * @brief Nose-Hoover 链恒温器的实现
 *
 * 该文件包含 Nose-Hoover 链恒温器的实现，主要用于分子动力学模拟中的温度控制。
 * 它通过引入多个热浴变量，递推更新系统的速度和热浴变量，从而实现温度调节。
 *
 * @author Gilbert Young
 * @date 2024-10-19
 */

#include <cmath>
#include <vector>

extern "C"
{
    /**
     * @brief 应用 Nose-Hoover 链恒温器
     *
     * @param dt 时间步长
     * @param num_atoms 原子数量
     * @param masses 原子质量数组（长度为 num_atoms）
     * @param velocities 原子速度数组（长度为 3*num_atoms）
     * @param forces 原子力数组（长度为 3*num_atoms）
     * @param xi 热浴变量数组（长度为 chain_length）
     * @param Q 热浴质量参数数组（长度为 chain_length）
     * @param chain_length 热浴链的长度
     * @param target_temperature 目标温度
     */
    void nose_hoover_chain(
        double dt,
        int num_atoms,
        const double *masses,
        double *velocities,
        const double *forces,
        double *xi,
        const double *Q,
        int chain_length,
        double target_temperature)
    {
        // 使用 std::vector 代替动态数组
        std::vector<double> G_chain(chain_length, 0.0);

        double dt_half = dt * 0.5;
        double kB = 8.617333262e-5; // 玻尔兹曼常数，单位 eV/K

        // 计算动能
        double kinetic_energy = 0.0;
        for (int i = 0; i < num_atoms; ++i)
        {
            double vx = velocities[3 * i];
            double vy = velocities[3 * i + 1];
            double vz = velocities[3 * i + 2];
            kinetic_energy += 0.5 * masses[i] * (vx * vx + vy * vy + vz * vz);
        }

        // 递推更新 Nose-Hoover 链
        G_chain[0] = (2.0 * kinetic_energy - 3.0 * num_atoms * kB * target_temperature) / Q[0];
        for (int i = 1; i < chain_length; ++i)
        {
            G_chain[i] = (Q[i - 1] * xi[i - 1] * xi[i - 1] - kB * target_temperature) / Q[i];
        }

        // 更新 xi
        for (int i = chain_length - 1; i >= 0; --i)
        {
            xi[i] += G_chain[i] * dt_half;
        }

        // 缩放速度
        double scale = exp(-xi[0] * dt);
        for (int i = 0; i < num_atoms; ++i)
        {
            velocities[3 * i] *= scale;
            velocities[3 * i + 1] *= scale;
            velocities[3 * i + 2] *= scale;
        }

        // 更新 xi
        for (int i = 0; i < chain_length; ++i)
        {
            xi[i] += G_chain[i] * dt_half;
        }

        // 更新速度，考虑力
        for (int i = 0; i < num_atoms; ++i)
        {
            velocities[3 * i] += dt * forces[3 * i] / masses[i];
            velocities[3 * i + 1] += dt * forces[3 * i + 1] / masses[i];
            velocities[3 * i + 2] += dt * forces[3 * i + 2] / masses[i];
        }
    }
}

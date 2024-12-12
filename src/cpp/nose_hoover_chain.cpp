/**
 * @file nose_hoover_chain.cpp
 * @brief Nose-Hoover 链恒温器的实现
 *
 * 该文件包含 Nose-Hoover 链恒温器的实现，主要用于分子动力学模拟中的温度控制。
 * 它通过引入多个热浴变量，递推更新系统的速度和热浴变量，从而实现温度调节。
 *
 * @autor Gilbert Young
 * @date 2024-10-20
 */

#include <cmath>
#include <cstddef>

// 内联函数用于计算动能
inline double compute_kinetic_energy(int num_atoms, const double *masses, const double *velocities)
{
    double kinetic_energy = 0.0;
    for (int i = 0; i < num_atoms; ++i)
    {
        double vx = velocities[3 * i];
        double vy = velocities[3 * i + 1];
        double vz = velocities[3 * i + 2];
        kinetic_energy += 0.5 * masses[i] * (vx * vx + vy * vy + vz * vz);
    }
    return kinetic_energy;
}

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
        if (chain_length <= 0)
            return; // 简单的参数检查

        const double dt_half = dt * 0.5;
        const double kB = 8.617332385e-5; // 玻尔兹曼常数，单位 eV/K

        // 计算动能
        double kinetic_energy = compute_kinetic_energy(num_atoms, masses, velocities);

        // 计算 G_chain
        // 预分配数组以避免 std::vector 的构造开销
        double *G_chain = new double[chain_length];

        G_chain[0] = (2.0 * kinetic_energy - 3.0 * num_atoms * kB * target_temperature) / Q[0];
        for (int i = 1; i < chain_length; ++i)
        {
            G_chain[i] = (Q[i - 1] * xi[i - 1] * xi[i - 1] - kB * target_temperature) / Q[i];
        }

        // 更新 xi - 第一半步（从链末端到开始）
        for (int i = chain_length - 1; i >= 0; --i)
        {
            xi[i] += G_chain[i] * dt_half;
        }

        // 缩放速度
        double scale = std::exp(-xi[0] * dt_half);
        for (int i = 0; i < num_atoms; ++i)
        {
            int idx = 3 * i;
            velocities[idx] *= scale;
            velocities[idx + 1] *= scale;
            velocities[idx + 2] *= scale;
        }

        // 更新 xi - 第二半步（从开始到链末端）
        for (int i = 0; i < chain_length; ++i)
        {
            xi[i] += G_chain[i] * dt_half;
        }

        // 更新速度，考虑力
        for (int i = 0; i < num_atoms; ++i)
        {
            int idx = 3 * i;
            double inv_mass = 1.0 / masses[i];
            velocities[idx] += dt * forces[idx] * inv_mass;
            velocities[idx + 1] += dt * forces[idx + 1] * inv_mass;
            velocities[idx + 2] += dt * forces[idx + 2] * inv_mass;
        }

        // 清理动态分配的内存
        delete[] G_chain;
    }
}

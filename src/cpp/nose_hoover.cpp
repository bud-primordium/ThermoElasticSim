/**
 * @file nose_hoover.cpp
 * @brief Nose-Hoover 恒温器的实现
 *
 * 该文件包含 Nose-Hoover 恒温器的实现，主要用于分子动力学模拟中的温度控制。
 * 它通过引入单个热浴变量，遵循拓展哈密顿量，更新系统的速度和热浴变量，从而实现温度调节。
 *
 * @author Gilbert
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
     * @brief 应用 Nose-Hoover 恒温器
     *
     * @param dt 时间步长
     * @param num_atoms 原子数量
     * @param masses 原子质量数组（长度为 num_atoms）
     * @param velocities 原子速度数组（长度为 3*num_atoms）
     * @param forces 原子力数组（长度为 3*num_atoms）
     * @param xi 热浴变量
     * @param Q 热浴质量参数
     * @param target_temperature 目标温度
     */
    void nose_hoover(
        double dt,
        int num_atoms,
        const double *masses,
        double *velocities,
        const double *forces,
        double *xi,
        double Q,
        double target_temperature)
    {
        const double dt_half = dt * 0.5;
        const double kB = 8.617333262e-5; // 玻尔兹曼常数，单位 eV/K

        // 第一半步：更新速度，考虑力
        for (int i = 0; i < num_atoms; ++i)
        {
            int idx = 3 * i;
            double inv_mass = 1.0 / masses[i];
            velocities[idx] += dt_half * forces[idx] * inv_mass;
            velocities[idx + 1] += dt_half * forces[idx + 1] * inv_mass;
            velocities[idx + 2] += dt_half * forces[idx + 2] * inv_mass;
        }

        // 计算动能
        double kinetic_energy = compute_kinetic_energy(num_atoms, masses, velocities);

        // 更新 xi（热浴变量） - 第一半步
        double G_xi = (2.0 * kinetic_energy - 3.0 * num_atoms * kB * target_temperature) / Q;
        *xi += dt_half * G_xi;

        // 缩放速度
        double scale = std::exp(-(*xi) * dt_half);
        for (int i = 0; i < num_atoms; ++i)
        {
            int idx = 3 * i;
            velocities[idx] *= scale;
            velocities[idx + 1] *= scale;
            velocities[idx + 2] *= scale;
        }

        // 重新计算动能
        kinetic_energy = compute_kinetic_energy(num_atoms, masses, velocities);

        // 更新 xi（热浴变量） - 第二半步
        G_xi = (2.0 * kinetic_energy - 3.0 * num_atoms * kB * target_temperature) / Q;
        *xi += dt_half * G_xi;

        // 第二半步：更新速度，考虑力
        for (int i = 0; i < num_atoms; ++i)
        {
            int idx = 3 * i;
            double inv_mass = 1.0 / masses[i];
            velocities[idx] += dt_half * forces[idx] * inv_mass;
            velocities[idx + 1] += dt_half * forces[idx + 1] * inv_mass;
            velocities[idx + 2] += dt_half * forces[idx + 2] * inv_mass;
        }
    }
}

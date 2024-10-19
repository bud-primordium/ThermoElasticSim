/**
 * @file nose_hoover.cpp
 * @brief Nose-Hoover 恒温器的实现
 *
 * 该文件包含 Nose-Hoover 恒温器的实现，主要用于分子动力学模拟中的温度控制。
 * 它通过引入单个热浴变量，遵循拓展哈密顿量，更新系统的速度和热浴变量，从而实现温度调节。
 *
 * @author Gilbert Young
 * @date 2024-10-19
 */

#include <cmath>
#include <vector>

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
        double dt_half = dt * 0.5;
        double kB = 8.617333262e-5; // 玻尔兹曼常数，单位 eV/K

        // 第一半步：更新速度，考虑力
        for (int i = 0; i < num_atoms; ++i)
        {
            velocities[3 * i] += dt_half * forces[3 * i] / masses[i];
            velocities[3 * i + 1] += dt_half * forces[3 * i + 1] / masses[i];
            velocities[3 * i + 2] += dt_half * forces[3 * i + 2] / masses[i];
        }

        // 更新 xi（热浴变量） - 第一半步
        double kinetic_energy = 0.0;
        for (int i = 0; i < num_atoms; ++i)
        {
            double vx = velocities[3 * i];
            double vy = velocities[3 * i + 1];
            double vz = velocities[3 * i + 2];
            kinetic_energy += 0.5 * masses[i] * (vx * vx + vy * vy + vz * vz);
        }
        double G_xi = (2.0 * kinetic_energy - 3.0 * num_atoms * kB * target_temperature) / Q;
        *xi += dt_half * G_xi;

        // 缩放速度
        double scale = exp(-(*xi) * dt);
        for (int i = 0; i < num_atoms; ++i)
        {
            velocities[3 * i] *= scale;
            velocities[3 * i + 1] *= scale;
            velocities[3 * i + 2] *= scale;
        }

        // 更新 xi（热浴变量） - 第二半步
        kinetic_energy = 0.0;
        for (int i = 0; i < num_atoms; ++i)
        {
            double vx = velocities[3 * i];
            double vy = velocities[3 * i + 1];
            double vz = velocities[3 * i + 2];
            kinetic_energy += 0.5 * masses[i] * (vx * vx + vy * vy + vz * vz);
        }
        G_xi = (2.0 * kinetic_energy - 3.0 * num_atoms * kB * target_temperature) / Q;
        *xi += dt_half * G_xi;

        // 第二半步：更新速度，考虑力
        for (int i = 0; i < num_atoms; ++i)
        {
            velocities[3 * i] += dt_half * forces[3 * i] / masses[i];
            velocities[3 * i + 1] += dt_half * forces[3 * i + 1] / masses[i];
            velocities[3 * i + 2] += dt_half * forces[3 * i + 2] / masses[i];
        }
    }
}

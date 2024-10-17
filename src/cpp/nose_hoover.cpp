// src/cpp/nose_hoover.cpp

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
        double dt2 = dt / 2.0;
        double kB = 8.617333262e-5; // 玻尔兹曼常数，单位 eV/K
        double kT = kB * target_temperature;

        // 第一半步：更新速度
        for (int i = 0; i < num_atoms; ++i)
        {
            velocities[3 * i] += dt2 * (forces[3 * i] / masses[i]);
            velocities[3 * i + 1] += dt2 * (forces[3 * i + 1] / masses[i]);
            velocities[3 * i + 2] += dt2 * (forces[3 * i + 2] / masses[i]);
        }

        // 计算动能
        double kinetic_energy = 0.0;
        for (int i = 0; i < num_atoms; ++i)
        {
            double vx = velocities[3 * i];
            double vy = velocities[3 * i + 1];
            double vz = velocities[3 * i + 2];
            kinetic_energy += 0.5 * masses[i] * (vx * vx + vy * vy + vz * vz);
        }

        // 更新热浴变量 xi
        double Gxi = (2.0 * kinetic_energy - 3.0 * num_atoms * kT) / Q;
        *xi += dt * Gxi;

        // 第二半步：更新速度，考虑热浴变量的影响
        double exp_factor = exp(-dt * (*xi));
        for (int i = 0; i < num_atoms; ++i)
        {
            velocities[3 * i] = velocities[3 * i] * exp_factor + dt2 * (forces[3 * i] / masses[i]);
            velocities[3 * i + 1] = velocities[3 * i + 1] * exp_factor + dt2 * (forces[3 * i + 1] / masses[i]);
            velocities[3 * i + 2] = velocities[3 * i + 2] * exp_factor + dt2 * (forces[3 * i + 2] / masses[i]);
        }
    }
}

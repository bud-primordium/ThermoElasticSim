/**
 * @file eam_al1.cpp
 * @brief 基于 Mendelev 等人（2008）的 EAM Al1 势函数实现
 *
 * 实现了 EAM Al1 势函数，主要包括三个函数：
 * 1. 对势函数 φ(r_ij)
 * 2. 电子密度贡献 ψ(r_ij)
 * 3. 嵌入能 F(ρ)
 *
 * 总能量：E = Σ_i Σ_{j>i} φ(r_ij) + Σ_i F(ρ_i)
 * 其中 ρ_i = Σ_{j≠i} ψ(r_ij)
 *
 * 力：F_i = -∇_i E
 *
 * 参考文献：
 * Mendelev 等人，“Analysis of semi-empirical interatomic potentials
 * appropriate for simulation of crystalline and liquid Al and Cu.”
 * Philosophical Magazine 88.12 (2008): 1723-1750.
 */

#include <cmath>
#include <vector>
#include <cstring>

extern "C"
{

    /**
     * @brief 对势函数 φ(r) 对于 Al1
     *
     * φ(r) 分段定义：
     * 1. r ∈ [1.5, 2.3]: exp(a₀ + a₁r + a₂r² + a₃r³)
     * 2. r ∈ (2.3, 3.2]: Σ bₙ(3.2 - r)ⁿ, n = 4..8
     * 3. r ∈ (2.3, 4.8]: Σ cₙ(4.8 - r)ⁿ, n = 4..8
     * 4. r ∈ (2.3, 6.5]: Σ dₙ(6.5 - r)ⁿ, n = 4..8
     *
     * @param r 原子间距
     * @return 对势能量
     */
    inline double phi(double r)
    {
        double phi = 0.0;

        // 定义常数
        const double a0 = 0.65196946237834;
        const double a1 = 7.6046051582736;
        const double a2 = -5.8187505542843;
        const double a3 = 1.0326940511805;

        const double b[5] = {13.695567100510, -44.514029786506, 95.853674731436, -83.744769235189, 29.906639687889};
        const double c[5] = {-2.3612121457801, 2.5279092055084, -3.3656803584012, 0.94831589893263, -0.20965407907747};
        const double d[5] = {0.24809459274509, -0.54072248340384, 0.46579408228733, -0.18481649031556, 0.028257788274378};

        // 区域1: [1.5, 2.3]
        if (r >= 1.5 && r <= 2.3)
        {
            phi += exp(a0 + a1 * r + a2 * r * r + a3 * r * r * r);
        }

        // 区域2: (2.3, 3.2]
        if (r > 2.3 && r <= 3.2)
        {
            const double dr = 3.2 - r;
            for (int n = 0; n < 5; ++n)
            {
                phi += b[n] * pow(dr, n + 4);
            }
        }

        // 区域3: (2.3, 4.8]
        if (r > 2.3 && r <= 4.8)
        {
            const double dr = 4.8 - r;
            for (int n = 0; n < 5; ++n)
            {
                phi += c[n] * pow(dr, n + 4);
            }
        }

        // 区域4: (2.3, 6.5]
        if (r > 2.3 && r <= 6.5)
        {
            const double dr = 6.5 - r;
            for (int n = 0; n < 5; ++n)
            {
                phi += d[n] * pow(dr, n + 4);
            }
        }

        return phi;
    }

    /**
     * @brief 电子密度贡献函数 ψ(r)
     *
     * ψ(r) = Σₖ Cₖ(rₖ - r)⁴ Θ(rₖ - r)
     * 其中 Θ 是 Heaviside 阶跃函数
     *
     * @param r 原子间距
     * @return 电子密度贡献
     */
    inline double psi(double r)
    {
        double psi = 0.0;

        // 定义常数
        const int num_terms = 10;
        const double c_k[num_terms] = {
            0.00019850823042883, 0.10046665347629, 0.10054338881951, 0.099104582963213,
            0.090086286376778, 0.0073022698419468, 0.014583614223199, -0.0010327381407070,
            0.0073219994475288, 0.0095726042919017};

        const double r_k[num_terms] = {2.5, 2.6, 2.7, 2.8, 3.0, 3.4, 4.2, 4.8, 5.6, 6.5};

        for (int i = 0; i < num_terms; ++i)
        {
            if (r <= r_k[i])
            {
                psi += c_k[i] * pow(r_k[i] - r, 4);
            }
        }

        return psi;
    }

    /**
     * @brief 嵌入能函数 F(ρ)
     *
     * F(ρ) = -√ρ + Σₙ Dₙ(ρ - 16)ⁿ，对于 ρ ≥ 16
     * F(ρ) = -√ρ，对于 ρ < 16
     *
     * @param rho 局域电子密度
     * @return 嵌入能
     */
    inline double Phi(double rho)
    {
        double F = -sqrt(rho); // 对所有 ρ 的基本项

        if (rho >= 16.0)
        {
            const double dr = rho - 16.0;
            const int num_terms = 6;
            const double D_n[num_terms] = {
                -6.1596236428225e-5, 1.4856817073764e-5, -1.4585661621587e-6,
                7.2242013524147e-8, -1.7925388537626e-9, 1.7720686711226e-11};

            for (int n = 0; n < num_terms; ++n)
            {
                F += D_n[n] * pow(dr, n + 4);
            }
        }

        return F;
    }

    /**
     * @brief 对势函数的导数 φ'(r)
     */
    inline double phi_grad(double r)
    {
        double dphi = 0.0;

        // 定义常数
        const double a0 = 0.65196946237834;
        const double a1 = 7.6046051582736;
        const double a2 = -5.8187505542843;
        const double a3 = 1.0326940511805;

        const double b[5] = {13.695567100510, -44.514029786506, 95.853674731436, -83.744769235189, 29.906639687889};
        const double c[5] = {-2.3612121457801, 2.5279092055084, -3.3656803584012, 0.94831589893263, -0.20965407907747};
        const double d[5] = {0.24809459274509, -0.54072248340384, 0.46579408228733, -0.18481649031556, 0.028257788274378};

        if (r < 1.5)
        {
            return -1e10; // 在 phi_grad 中
        }
        if (r >= 1.5 && r <= 2.3)
        {
            const double exp_term = exp(a0 + a1 * r + a2 * r * r + a3 * r * r * r);
            dphi += (a1 + 2.0 * a2 * r + 3.0 * a3 * r * r) * exp_term;
        }

        if (r > 2.3 && r <= 3.2)
        {
            const double dr = 3.2 - r;
            for (int n = 0; n < 5; ++n)
            {
                dphi += -(n + 4) * b[n] * pow(dr, n + 3);
            }
        }

        if (r > 2.3 && r <= 4.8)
        {
            const double dr = 4.8 - r;
            for (int n = 0; n < 5; ++n)
            {
                dphi += -(n + 4) * c[n] * pow(dr, n + 3);
            }
        }

        if (r > 2.3 && r <= 6.5)
        {
            const double dr = 6.5 - r;
            for (int n = 0; n < 5; ++n)
            {
                dphi += -(n + 4) * d[n] * pow(dr, n + 3);
            }
        }

        return -dphi; // 注意：用于力计算的负号
    }

    /**
     * @brief 电子密度 ψ(r) 的导数 ψ'(r)
     */
    inline double psi_grad(double r)
    {
        double dpsi = 0.0;

        // 定义常数
        const int num_terms = 10;
        const double c_k[num_terms] = {
            0.00019850823042883, 0.10046665347629, 0.10054338881951, 0.099104582963213,
            0.090086286376778, 0.0073022698419468, 0.014583614223199, -0.0010327381407070,
            0.0073219994475288, 0.0095726042919017};

        const double r_k[num_terms] = {2.5, 2.6, 2.7, 2.8, 3.0, 3.4, 4.2, 4.8, 5.6, 6.5};

        for (int i = 0; i < num_terms; ++i)
        {
            if (r <= r_k[i])
            {
                dpsi += -4.0 * c_k[i] * pow(r_k[i] - r, 3);
            }
        }

        return -dpsi; // 注意：用于力计算的负号
    }

    /**
     * @brief 嵌入能函数的导数 F'(ρ)
     */
    inline double Phi_grad(double rho)
    {
        double dF = -0.5 / sqrt(rho); // 对所有 ρ 的基本项

        if (rho >= 16.0)
        {
            const double dr = rho - 16.0;
            const int num_terms = 6;
            const double D_n[num_terms] = {
                -6.1596236428225e-5, 1.4856817073764e-5, -1.4585661621587e-6,
                7.2242013524147e-8, -1.7925388537626e-9, 1.7720686711226e-11};

            for (int n = 0; n < num_terms; ++n)
            {
                dF += (n + 4) * D_n[n] * pow(dr, n + 3);
            }
        }

        return dF;
    }

    /**
     * @brief 计算系统的总能量
     */
    void calculate_eam_al1_energy(
        int num_atoms,
        const double *positions,
        const double *box_lengths,
        double *energy // 输出参数
    )
    {
        double pair_energy = 0.0;
        std::vector<double> electron_density(num_atoms, 0.0);

        // 计算电子密度和对势能
        for (int i = 0; i < num_atoms; ++i)
        {
            for (int j = i + 1; j < num_atoms; ++j)
            {
                // 计算位移向量
                double rij[3];
                double r2 = 0.0;
                for (int k = 0; k < 3; ++k)
                {
                    rij[k] = positions[3 * j + k] - positions[3 * i + k];
                    // 使用最小镜像原则
                    rij[k] -= box_lengths[k] * round(rij[k] / box_lengths[k]);
                    r2 += rij[k] * rij[k];
                }

                double r = sqrt(r2);
                if (r <= 6.5)
                { // Al1 的截断半径
                    pair_energy += phi(r);
                    double psi_val = psi(r);
                    electron_density[i] += psi_val;
                    electron_density[j] += psi_val;
                }
            }
        }

        // 计算嵌入能
        double embed_energy = 0.0;
        for (int i = 0; i < num_atoms; ++i)
        {
            embed_energy += Phi(electron_density[i]);
        }

        *energy = pair_energy + embed_energy;
    }

    /**
     * @brief 计算所有原子的力
     */
    void calculate_eam_al1_forces(
        int num_atoms,
        const double *positions,
        const double *box_lengths,
        double *forces // 输出参数
    )
    {
        // 1. 计算电子密度
        std::vector<double> electron_density(num_atoms, 0.0);
        for (int i = 0; i < num_atoms; ++i)
        {
            for (int j = 0; j < num_atoms; ++j)
            {
                if (i == j)
                    continue;

                double rij[3];
                double r2 = 0.0;
                for (int k = 0; k < 3; ++k)
                {
                    rij[k] = positions[3 * j + k] - positions[3 * i + k];
                    rij[k] -= box_lengths[k] * round(rij[k] / box_lengths[k]);
                    r2 += rij[k] * rij[k];
                }
                double r = sqrt(r2);

                if (r <= 6.5)
                {
                    electron_density[i] += psi(r);
                }
            }
        }

        // 2. 将力初始化为零
        std::memset(forces, 0, 3 * num_atoms * sizeof(double));

        // 3. 计算原子对之间的力
        for (int i = 0; i < num_atoms; ++i)
        {
            double dFi = Phi_grad(electron_density[i]);
            for (int j = i + 1; j < num_atoms; ++j)
            {
                double rij[3];
                double r2 = 0.0;
                for (int k = 0; k < 3; ++k)
                {
                    rij[k] = positions[3 * j + k] - positions[3 * i + k];
                    rij[k] -= box_lengths[k] * round(rij[k] / box_lengths[k]);
                    r2 += rij[k] * rij[k];
                }
                double r = sqrt(r2);
                if (r <= 6.5)
                {
                    // 对势力
                    double phi_prime = phi_grad(r);
                    double pair_force = phi_prime;

                    // 嵌入能力
                    double psi_prime = psi_grad(r);
                    double dFj = Phi_grad(electron_density[j]);
                    double embed_force = (dFi + dFj) * psi_prime;

                    // 总力
                    double total_force = (pair_force + embed_force) / r;

                    // 应用力
                    for (int k = 0; k < 3; ++k)
                    {
                        double force_k = total_force * rij[k];
                        forces[3 * i + k] += force_k;
                        forces[3 * j + k] -= force_k;
                    }
                }
            }
        }
    }

} // extern "C"

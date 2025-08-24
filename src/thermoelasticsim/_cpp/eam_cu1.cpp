/**
 * @file eam_cu1.cpp
 * @brief 基于 Mendelev 等人（2008）的 EAM Cu1 势函数实现
 *
 * 实现了 EAM Cu1 势函数，主要包括三个函数：
 * 1. 对势函数 φ(r_ij): 4段分段函数，区间[1.0, 1.8], (1.8, 2.8], (1.8, 4.8], (1.8, 6.0]
 * 2. 电子密度贡献 ψ(r_ij): 4段简化形式，截断半径分别为2.4, 3.2, 4.5, 6.0 Å
 * 3. 嵌入能 F(ρ): 9段分段函数，基础项 -√ρ 加上8个密度依赖修正项
 *
 * 总能量：E = Σ_i Σ_{j>i} φ(r_ij) + Σ_i F(ρ_i)
 * 其中 ρ_i = Σ_{j≠i} ψ(r_ij)
 *
 * 力：F_i = -∇_i E
 *
 * 截断半径：6.0 Å
 * 适用元素：铜 (Cu)
 * 晶体结构：面心立方 (FCC)
 *
 * 参考文献：
 * Mendelev 等人，"Analysis of semi-empirical interatomic potentials
 * appropriate for simulation of crystalline and liquid Al and Cu."
 * Philosophical Magazine 88.12 (2008): 1723-1750.
 */

#include <cmath>
#include <vector>
#include <cstring>

extern "C"
{

    /**
     * @brief 对势函数 φ(r) 对于 Cu1
     *
     * φ(r) 分段定义：
     * 1. r ∈ [1.0, 1.8]: exp(11.026... - 10.167...r + 6.001...r² - 1.959...r³)
     * 2. r ∈ (1.8, 2.8]: Σ bₙ(2.8 - r)ⁿ, n = 4..8
     * 3. r ∈ (1.8, 4.8]: Σ cₙ(4.8 - r)ⁿ, n = 4..8
     * 4. r ∈ (1.8, 6.0]: Σ dₙ(6.0 - r)ⁿ, n = 4..8
     *
     * @param r 原子间距
     * @return 对势能量
     */
    static inline double phi(double r)
    {
        double phi_val = 0.0;

        // 定义常数
        const double a0 = 11.026565103477;
        const double a1 = -10.167211017722;
        const double a2 = 6.0017702915006;
        const double a3 = -1.9598299733506;

        const double b[5] = {3.3519281301971, -47.447602323833, 111.06454537813, -122.56379390195, 49.145722026502};
        const double c[5] = {4.0605833179061, 2.5958091214976, 5.5656640545299, 1.5184323060743, 0.39696001635415};
        const double d[5] = {-0.21402913758299, 1.1714811538458, -1.9913969426765, 1.3862043035438, -0.34520315264743};

        // 区域1: [1.0, 1.8]
        if (r >= 1.0 && r <= 1.8)
        {
            phi_val += exp(a0 + a1 * r + a2 * r * r + a3 * r * r * r);
        }

        // 根据文献附录格式，以下区间是累加的
        // 区域2: (1.8, 2.8]
        if (r > 1.8 && r <= 2.8)
        {
            const double dr = 2.8 - r;
            for (int n = 0; n < 5; ++n)
            {
                phi_val += b[n] * pow(dr, n + 4);
            }
        }

        // 区域3: (1.8, 4.8]
        if (r > 1.8 && r <= 4.8)
        {
            const double dr = 4.8 - r;
            for (int n = 0; n < 5; ++n)
            {
                phi_val += c[n] * pow(dr, n + 4);
            }
        }

        // 区域4: (1.8, 6.0]
        if (r > 1.8 && r <= 6.0)
        {
            const double dr = 6.0 - r;
            for (int n = 0; n < 5; ++n)
            {
                phi_val += d[n] * pow(dr, n + 4);
            }
        }

        return phi_val;
    }

    /**
     * @brief 电子密度贡献函数 ψ(r)
     *
     * ψ(r) = Σₖ Cₖ(rₖ - r)⁴ Θ(rₖ - r)
     * 其中 Θ 是 Heaviside 阶跃函数
     *
     * Cu1 势有4个分段
     *
     * @param r 原子间距
     * @return 电子密度贡献
     */
    static inline double psi(double r)
    {
        double psi_val = 0.0;

        // 定义常数和截断半径
        const int num_terms = 4;
        const double c_k[num_terms] = {
            0.019999999875362,
            0.019987533420669,
            0.018861676713565,
            0.0066082982694659};

        const double r_k[num_terms] = {2.4, 3.2, 4.5, 6.0};

        for (int i = 0; i < num_terms; ++i)
        {
            if (r <= r_k[i])
            {
                psi_val += c_k[i] * pow(r_k[i] - r, 4);
            }
        }

        return psi_val;
    }

    /**
     * @brief 嵌入能函数 F(ρ)
     *
     * F(ρ) = -√ρ + Σₙ Dₙ(ρ - ρₙ)⁴，对于 ρ ≥ ρₙ
     *
     * Cu1 势有9个分段
     *
     * @param rho 局域电子密度
     * @return 嵌入能
     */
    static inline double Phi(double rho)
    {
        double F = -sqrt(rho); // 对所有 ρ 的基本项

        // 定义分段点和系数
        const int num_terms = 8;
        const double rho_cutoffs[num_terms] = {9.0, 11.0, 13.0, 15.0, 16.0, 16.5, 17.0, 18.0};
        const double D_n[num_terms] = {
            -5.7112865649408e-5, // ρ >= 9
            3.0303487333648e-4,  // ρ >= 11
            -5.4720795296134e-4, // ρ >= 13
            4.6278681464721e-3,  // ρ >= 15
            1.0310712451906e-3,  // ρ >= 16
            3.0634000239833e-3,  // ρ >= 16.5
            -2.8308102136994e-3, // ρ >= 17
            6.4044567482688e-4   // ρ >= 18
        };

        // 累加各个分段贡献
        for (int n = 0; n < num_terms; ++n)
        {
            if (rho >= rho_cutoffs[n])
            {
                const double dr = rho - rho_cutoffs[n];
                F += D_n[n] * pow(dr, 4);
            }
        }

        return F;
    }

    /**
     * @brief 对势函数的导数 dφ/dr
     */
    static inline double phi_grad(double r)
    {
        double dphi = 0.0;

        // 定义常数（与phi函数一致）
        const double a0 = 11.026565103477;
        const double a1 = -10.167211017722;
        const double a2 = 6.0017702915006;
        const double a3 = -1.9598299733506;

        const double b[5] = {3.3519281301971, -47.447602323833, 111.06454537813, -122.56379390195, 49.145722026502};
        const double c[5] = {4.0605833179061, 2.5958091214976, 5.5656640545299, 1.5184323060743, 0.39696001635415};
        const double d[5] = {-0.21402913758299, 1.1714811538458, -1.9913969426765, 1.3862043035438, -0.34520315264743};

        if (r < 1.0)
        {
            return -1e10; // 在有效范围外，返回大的斥力导数
        }

        // 区域1: [1.0, 1.8] - 指数函数导数
        if (r >= 1.0 && r <= 1.8)
        {
            const double exp_term = exp(a0 + a1 * r + a2 * r * r + a3 * r * r * r);
            dphi += (a1 + 2.0 * a2 * r + 3.0 * a3 * r * r) * exp_term;
        }

        // 区域2: (1.8, 2.8] - 累加分段导数
        if (r > 1.8 && r <= 2.8)
        {
            const double dr = 2.8 - r;
            for (int n = 0; n < 5; ++n)
            {
                dphi += -(n + 4) * b[n] * pow(dr, n + 3);
            }
        }

        // 区域3: (1.8, 4.8]
        if (r > 1.8 && r <= 4.8)
        {
            const double dr = 4.8 - r;
            for (int n = 0; n < 5; ++n)
            {
                dphi += -(n + 4) * c[n] * pow(dr, n + 3);
            }
        }

        // 区域4: (1.8, 6.0]
        if (r > 1.8 && r <= 6.0)
        {
            const double dr = 6.0 - r;
            for (int n = 0; n < 5; ++n)
            {
                dphi += -(n + 4) * d[n] * pow(dr, n + 3);
            }
        }

        return dphi;
    }

    /**
     * @brief 电子密度 ψ(r) 的导数 dψ/dr
     */
    static inline double psi_grad(double r)
    {
        double dpsi = 0.0;

        // 定义常数（与psi函数一致）
        const int num_terms = 4;
        const double c_k[num_terms] = {
            0.019999999875362,
            0.019987533420669,
            0.018861676713565,
            0.0066082982694659};

        const double r_k[num_terms] = {2.4, 3.2, 4.5, 6.0};

        for (int i = 0; i < num_terms; ++i)
        {
            if (r <= r_k[i])
            {
                dpsi += -4.0 * c_k[i] * pow(r_k[i] - r, 3);
            }
        }

        return dpsi;
    }

    /**
     * @brief 嵌入能函数的导数 dF/dρ
     */
    static inline double Phi_grad(double rho)
    {
        double dF = -0.5 / sqrt(rho); // 对所有 ρ 的基本项导数

        // 定义分段点和系数（与Phi函数一致）
        const int num_terms = 8;
        const double rho_cutoffs[num_terms] = {9.0, 11.0, 13.0, 15.0, 16.0, 16.5, 17.0, 18.0};
        const double D_n[num_terms] = {
            -5.7112865649408e-5, // ρ >= 9
            3.0303487333648e-4,  // ρ >= 11
            -5.4720795296134e-4, // ρ >= 13
            4.6278681464721e-3,  // ρ >= 15
            1.0310712451906e-3,  // ρ >= 16
            3.0634000239833e-3,  // ρ >= 16.5
            -2.8308102136994e-3, // ρ >= 17
            6.4044567482688e-4   // ρ >= 18
        };

        // 累加各个分段贡献的导数
        for (int n = 0; n < num_terms; ++n)
        {
            if (rho >= rho_cutoffs[n])
            {
                const double dr = rho - rho_cutoffs[n];
                dF += 4.0 * D_n[n] * pow(dr, 3);
            }
        }

        return dF;
    }

    /**
     * @brief 计算系统的总能量
     */
    void calculate_eam_cu1_energy(
        int num_atoms,
        const double *positions,
        const double *lattice_vectors, // row-major 3x3
        double *energy                 // 输出参数
    )
    {
        // 读取晶格矩阵及其转置与逆
        // L is row-major: rows are a1, a2, a3
        double L[3][3];
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                L[r][c] = lattice_vectors[3 * r + c];
            }
        }
        // compute LT and inverse of LT
        double LT[3][3];
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                LT[r][c] = L[c][r];
            }
        }
        // invert LT (3x3 inverse)
        auto det3 = [&](double A[3][3])
        {
            return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
        };
        double detLT = det3(LT);
        // assume valid lattice (non-singular)
        double invLT[3][3];
        invLT[0][0] = (LT[1][1] * LT[2][2] - LT[1][2] * LT[2][1]) / detLT;
        invLT[0][1] = -(LT[0][1] * LT[2][2] - LT[0][2] * LT[2][1]) / detLT;
        invLT[0][2] = (LT[0][1] * LT[1][2] - LT[0][2] * LT[1][1]) / detLT;
        invLT[1][0] = -(LT[1][0] * LT[2][2] - LT[1][2] * LT[2][0]) / detLT;
        invLT[1][1] = (LT[0][0] * LT[2][2] - LT[0][2] * LT[2][0]) / detLT;
        invLT[1][2] = -(LT[0][0] * LT[1][2] - LT[0][2] * LT[1][0]) / detLT;
        invLT[2][0] = (LT[1][0] * LT[2][1] - LT[1][1] * LT[2][0]) / detLT;
        invLT[2][1] = -(LT[0][0] * LT[2][1] - LT[0][1] * LT[2][0]) / detLT;
        invLT[2][2] = (LT[0][0] * LT[1][1] - LT[0][1] * LT[1][0]) / detLT;

        double pair_energy = 0.0;
        std::vector<double> electron_density(num_atoms, 0.0);

        // 计算电子密度和对势能
        for (int i = 0; i < num_atoms; ++i)
        {
            for (int j = i + 1; j < num_atoms; ++j)
            {
                // 计算位移向量
                // rij = r_j - r_i
                double rij[3];
                rij[0] = positions[3 * j + 0] - positions[3 * i + 0];
                rij[1] = positions[3 * j + 1] - positions[3 * i + 1];
                rij[2] = positions[3 * j + 2] - positions[3 * i + 2];
                // s = inv(L^T) * rij  (fractional)
                double s[3];
                for (int a = 0; a < 3; ++a)
                {
                    s[a] = invLT[a][0] * rij[0] + invLT[a][1] * rij[1] + invLT[a][2] * rij[2];
                    // wrap to [-0.5,0.5)
                    s[a] -= std::floor(s[a] + 0.5);
                }
                // rij_mic = L^T * s
                double rij_mic[3];
                for (int a = 0; a < 3; ++a)
                {
                    rij_mic[a] = LT[a][0] * s[0] + LT[a][1] * s[1] + LT[a][2] * s[2];
                }
                double r2 = rij_mic[0] * rij_mic[0] + rij_mic[1] * rij_mic[1] + rij_mic[2] * rij_mic[2];

                double r = sqrt(r2);
                if (r <= 6.0)
                { // Cu1 的截断半径
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
    void calculate_eam_cu1_forces(
        int num_atoms,
        const double *positions,
        const double *lattice_vectors,
        double *forces // 输出参数
    )
    {
        // lattice matrices and inv(L^T)
        double L[3][3];
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                L[r][c] = lattice_vectors[3 * r + c];
            }
        }
        double LT[3][3];
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                LT[r][c] = L[c][r];
            }
        }
        auto det3 = [&](double A[3][3])
        {
            return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
        };
        double detLT = det3(LT);
        double invLT[3][3];
        invLT[0][0] = (LT[1][1] * LT[2][2] - LT[1][2] * LT[2][1]) / detLT;
        invLT[0][1] = -(LT[0][1] * LT[2][2] - LT[0][2] * LT[2][1]) / detLT;
        invLT[0][2] = (LT[0][1] * LT[1][2] - LT[0][2] * LT[1][1]) / detLT;
        invLT[1][0] = -(LT[1][0] * LT[2][2] - LT[1][2] * LT[2][0]) / detLT;
        invLT[1][1] = (LT[0][0] * LT[2][2] - LT[0][2] * LT[2][0]) / detLT;
        invLT[1][2] = -(LT[0][0] * LT[1][2] - LT[0][2] * LT[1][0]) / detLT;
        invLT[2][0] = (LT[1][0] * LT[2][1] - LT[1][1] * LT[2][0]) / detLT;
        invLT[2][1] = -(LT[0][0] * LT[2][1] - LT[0][1] * LT[2][0]) / detLT;
        invLT[2][2] = (LT[0][0] * LT[1][1] - LT[0][1] * LT[1][0]) / detLT;

        // 1. 计算电子密度
        std::vector<double> electron_density(num_atoms, 0.0);
        for (int i = 0; i < num_atoms; ++i)
        {
            for (int j = 0; j < num_atoms; ++j)
            {
                if (i == j)
                    continue;

                double rij[3];
                rij[0] = positions[3 * j + 0] - positions[3 * i + 0];
                rij[1] = positions[3 * j + 1] - positions[3 * i + 1];
                rij[2] = positions[3 * j + 2] - positions[3 * i + 2];
                double s[3];
                for (int a = 0; a < 3; ++a)
                {
                    s[a] = invLT[a][0] * rij[0] + invLT[a][1] * rij[1] + invLT[a][2] * rij[2];
                    s[a] -= std::floor(s[a] + 0.5);
                }
                double rij_mic[3];
                for (int a = 0; a < 3; ++a)
                {
                    rij_mic[a] = LT[a][0] * s[0] + LT[a][1] * s[1] + LT[a][2] * s[2];
                }
                double r2 = rij_mic[0] * rij_mic[0] + rij_mic[1] * rij_mic[1] + rij_mic[2] * rij_mic[2];
                double r = sqrt(r2);

                if (r <= 6.0)
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
            for (int j = i + 1; j < num_atoms; ++j)
            {
                double rij[3];
                rij[0] = positions[3 * j + 0] - positions[3 * i + 0];
                rij[1] = positions[3 * j + 1] - positions[3 * i + 1];
                rij[2] = positions[3 * j + 2] - positions[3 * i + 2];
                double s[3];
                for (int a = 0; a < 3; ++a)
                {
                    s[a] = invLT[a][0] * rij[0] + invLT[a][1] * rij[1] + invLT[a][2] * rij[2];
                    s[a] -= std::floor(s[a] + 0.5);
                }
                double rij_mic[3];
                for (int a = 0; a < 3; ++a)
                {
                    rij_mic[a] = LT[a][0] * s[0] + LT[a][1] * s[1] + LT[a][2] * s[2];
                }
                double r = std::sqrt(rij_mic[0] * rij_mic[0] + rij_mic[1] * rij_mic[1] + rij_mic[2] * rij_mic[2]);
                if (r > 1e-6 && r <= 6.0)
                {
                    double d_phi = phi_grad(r);
                    double d_psi = psi_grad(r);
                    double d_F_i = Phi_grad(electron_density[i]);
                    double d_F_j = Phi_grad(electron_density[j]);

                    // F = -dE/dr
                    double force_magnitude = -(d_phi + (d_F_i + d_F_j) * d_psi);

                    // direction along rij_mic
                    for (int k = 0; k < 3; ++k)
                    {
                        double force_component = force_magnitude * (rij_mic[k] / r);
                        forces[3 * i + k] += force_component;
                        forces[3 * j + k] -= force_component;
                    }
                }
            }
        }
    }

    /**
     * @brief 计算 EAM Cu1 的维里张量 (不除以体积)
     *
     * 维里定义（与Python实现一致）：
     * virial = - Σ_{i<j} r_ij^(mic) ⊗ F_ij
     * 其中 F_ij 是 j 对 i 的力（对称相反）
     *
     * @param num_atoms 原子数
     * @param positions 位置(3N)
     * @param lattice_vectors 晶格(3x3 row-major)
     * @param virial_tensor 输出(9, row-major)
     */
    void calculate_eam_cu1_virial(
        int num_atoms,
        const double *positions,
        const double *lattice_vectors,
        double *virial_tensor)
    {
        // 初始化输出
        for (int i = 0; i < 9; ++i)
            virial_tensor[i] = 0.0;

        // 构造 L, LT, inv(LT)
        double L[3][3];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                L[r][c] = lattice_vectors[3 * r + c];

        double LT[3][3];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                LT[r][c] = L[c][r];

        auto det3 = [&](double A[3][3])
        {
            return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
        };
        double detLT = det3(LT);
        double invLT[3][3];
        invLT[0][0] = (LT[1][1] * LT[2][2] - LT[1][2] * LT[2][1]) / detLT;
        invLT[0][1] = -(LT[0][1] * LT[2][2] - LT[0][2] * LT[2][1]) / detLT;
        invLT[0][2] = (LT[0][1] * LT[1][2] - LT[0][2] * LT[1][1]) / detLT;
        invLT[1][0] = -(LT[1][0] * LT[2][2] - LT[1][2] * LT[2][0]) / detLT;
        invLT[1][1] = (LT[0][0] * LT[2][2] - LT[0][2] * LT[2][0]) / detLT;
        invLT[1][2] = -(LT[0][0] * LT[1][2] - LT[0][2] * LT[1][0]) / detLT;
        invLT[2][0] = (LT[1][0] * LT[2][1] - LT[1][1] * LT[2][0]) / detLT;
        invLT[2][1] = -(LT[0][0] * LT[2][1] - LT[0][1] * LT[2][0]) / detLT;
        invLT[2][2] = (LT[0][0] * LT[1][1] - LT[0][1] * LT[1][0]) / detLT;

        // 1) 电子密度
        std::vector<double> electron_density(num_atoms, 0.0);
        for (int i = 0; i < num_atoms; ++i)
        {
            for (int j = 0; j < num_atoms; ++j)
            {
                if (i == j)
                    continue;
                double rij[3] = {positions[3 * j + 0] - positions[3 * i + 0],
                                 positions[3 * j + 1] - positions[3 * i + 1],
                                 positions[3 * j + 2] - positions[3 * i + 2]};
                double s[3];
                for (int a = 0; a < 3; ++a)
                {
                    s[a] = invLT[a][0] * rij[0] + invLT[a][1] * rij[1] + invLT[a][2] * rij[2];
                    s[a] -= std::floor(s[a] + 0.5);
                }
                double rij_mic[3] = {LT[0][0] * s[0] + LT[0][1] * s[1] + LT[0][2] * s[2],
                                     LT[1][0] * s[0] + LT[1][1] * s[1] + LT[1][2] * s[2],
                                     LT[2][0] * s[0] + LT[2][1] * s[1] + LT[2][2] * s[2]};
                double r = std::sqrt(rij_mic[0] * rij_mic[0] + rij_mic[1] * rij_mic[1] + rij_mic[2] * rij_mic[2]);
                if (r <= 6.0)
                {
                    electron_density[i] += psi(r);
                }
            }
        }

        // 2) 配对循环累加维里
        for (int i = 0; i < num_atoms; ++i)
        {
            for (int j = i + 1; j < num_atoms; ++j)
            {
                double rij[3] = {positions[3 * j + 0] - positions[3 * i + 0],
                                 positions[3 * j + 1] - positions[3 * i + 1],
                                 positions[3 * j + 2] - positions[3 * i + 2]};
                double s[3];
                for (int a = 0; a < 3; ++a)
                {
                    s[a] = invLT[a][0] * rij[0] + invLT[a][1] * rij[1] + invLT[a][2] * rij[2];
                    s[a] -= std::floor(s[a] + 0.5);
                }
                double rij_mic[3] = {LT[0][0] * s[0] + LT[0][1] * s[1] + LT[0][2] * s[2],
                                     LT[1][0] * s[0] + LT[1][1] * s[1] + LT[1][2] * s[2],
                                     LT[2][0] * s[0] + LT[2][1] * s[1] + LT[2][2] * s[2]};
                double r = std::sqrt(rij_mic[0] * rij_mic[0] + rij_mic[1] * rij_mic[1] + rij_mic[2] * rij_mic[2]);
                if (r > 1e-6 && r <= 6.0)
                {
                    double d_phi = phi_grad(r);
                    double d_psi = psi_grad(r);
                    double d_F_i = Phi_grad(electron_density[i]);
                    double d_F_j = Phi_grad(electron_density[j]);
                    double fmag = -(d_phi + (d_F_i + d_F_j) * d_psi);
                    double fij[3] = {fmag * (rij_mic[0] / r), fmag * (rij_mic[1] / r), fmag * (rij_mic[2] / r)};
                    // virial -= rij ⊗ F_ij
                    for (int a = 0; a < 3; ++a)
                    {
                        for (int b = 0; b < 3; ++b)
                        {
                            virial_tensor[3 * a + b] -= rij_mic[a] * fij[b];
                        }
                    }
                }
            }
        }
    }

} // extern "C"

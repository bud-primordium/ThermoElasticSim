/**
 * @file tersoff.cpp
 * @brief 单组分（C）Tersoff 势的能量/力/维里实现（含可选 shift=delta）
 *
 * 参考：Tersoff (1988, 1989)。实现遵循常见的多体团簇分解（成对+三元组）思路。
 */

#include <cmath>
#include <vector>
#include <cstring>

namespace tersoff_min
{
    struct Param
    {
        // 原始参数
        double A;       // biga
        double B;       // bigb
        double lambda1; // lam1
        double lambda2; // lam2
        double lambda3; // lam3
        double beta;    // beta
        double n;       // power n (>0)
        double c;       // c
        double d;       // d
        double h;       // h
        double R;       // bigr
        double D;       // bigd
        int m;          // powermint (typically 3)

        // 派生参数
        double cut;   // R + D
        double cutsq; // (R + D)^2

        // shift
        int shift_flag; // 0/1
        double shift;   // delta (r -> r + delta)
    };

    inline double fc(double r, const Param &p)
    {
        if (r < p.R - p.D)
            return 1.0;
        if (r > p.R + p.D)
            return 0.0;
        const double half_pi = 1.5707963267948966; // pi/2
        // 0.5 - 0.5 * sin(pi/2 * (r-R)/D)
        return 0.5 * (1.0 - std::sin(half_pi * (r - p.R) / p.D));
    }

    inline double fc_d(double r, const Param &p)
    {
        if (r <= p.R - p.D)
            return 0.0;
        if (r >= p.R + p.D)
            return 0.0;
        const double pi = 3.14159265358979323846;
        const double half_pi = 1.5707963267948966; // pi/2
        // derivative: -(pi/(4D)) * cos(pi/2 * (r-R)/D)
        return -(pi / (4.0 * p.D)) * std::cos(half_pi * (r - p.R) / p.D);
    }

    inline double fR(double r, const Param &p)
    {
        // A * exp(-lambda1 r)
        return p.A * std::exp(-p.lambda1 * r);
    }

    inline double fR_d(double r, const Param &p)
    {
        // d/dr [A exp(-lam1 r)] = -lam1 * A exp(-lam1 r) = -lam1 * fR
        return -p.lambda1 * fR(r, p);
    }

    inline double fA(double r, const Param &p)
    {
        // -B * exp(-lambda2 r)
        return -p.B * std::exp(-p.lambda2 * r);
    }

    inline double fA_d(double r, const Param &p)
    {
        // d/dr [-B exp(-lam2 r)] = + lam2 * B exp(-lam2 r) = -lam2 * fA (注意符号)
        return -p.lambda2 * fA(r, p);
    }

    inline double g_theta(double costh, const Param &p)
    {
        const double cc = p.c * p.c;
        const double dd = p.d * p.d;
        const double hcth = p.h - costh;
        return 1.0 + cc / dd - cc / (dd + hcth * hcth);
    }

    inline double g_theta_dcosth(double costh, const Param &p)
    {
        const double cc = p.c * p.c;
        const double dd = p.d * p.d;
        const double hcth = p.h - costh;
        const double denom = dd + hcth * hcth;
        return (-2.0 * cc * hcth) / (denom * denom);
    }

    inline double bij(double zeta, const Param &p)
    {
        // (1 + (beta^n) * zeta^n)^(-1/(2n))
        if (zeta <= 0.0)
            return 1.0;
        const double t1 = std::pow(p.beta, p.n) * std::pow(zeta, p.n);
        return std::pow(1.0 + t1, -1.0 / (2.0 * p.n));
    }

    inline double bij_dzeta(double zeta, const Param &p)
    {
        if (zeta <= 0.0)
            return 0.0;
        const double bn = std::pow(p.beta, p.n);
        const double zn_1 = std::pow(zeta, p.n - 1.0);
        const double t1 = 1.0 + bn * std::pow(zeta, p.n);
        // -1/2 * (beta^n) * zeta^{n-1} * (1 + beta^n zeta^n)^{-(2n+1)/(2n)}
        return -0.5 * bn * zn_1 * std::pow(t1, -(2.0 * p.n + 1.0) / (2.0 * p.n));
    }

    inline void mat3_from_flat(const double *flat9, double A[3][3])
    {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                A[r][c] = flat9[3 * r + c];
    }

    inline void mat3_transpose(const double A[3][3], double AT[3][3])
    {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                AT[r][c] = A[c][r];
    }

    inline double det3(const double A[3][3])
    {
        return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
               A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
               A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    }

    inline void inv3(const double A[3][3], double Ainv[3][3])
    {
        double det = det3(A);
        // 未做奇异性检查（调用方保证晶格有效）
        Ainv[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) / det;
        Ainv[0][1] = -(A[0][1] * A[2][2] - A[0][2] * A[2][1]) / det;
        Ainv[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) / det;
        Ainv[1][0] = -(A[1][0] * A[2][2] - A[1][2] * A[2][0]) / det;
        Ainv[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) / det;
        Ainv[1][2] = -(A[0][0] * A[1][2] - A[0][2] * A[1][0]) / det;
        Ainv[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) / det;
        Ainv[2][1] = -(A[0][0] * A[2][1] - A[0][1] * A[2][0]) / det;
        Ainv[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) / det;
    }

    inline void mic_displacement(const double LT[3][3], const double invLT[3][3],
                                 const double *ri, const double *rj, double rij[3])
    {
        double d[3] = {rj[0] - ri[0], rj[1] - ri[1], rj[2] - ri[2]};
        // fractional s = inv(L^T) * d
        double s[3];
        for (int a = 0; a < 3; ++a)
            s[a] = invLT[a][0] * d[0] + invLT[a][1] * d[1] + invLT[a][2] * d[2];
        // wrap to [-0.5, 0.5)
        for (int a = 0; a < 3; ++a)
            s[a] -= std::floor(s[a] + 0.5);
        // back to cartesian rij = LT * s
        for (int a = 0; a < 3; ++a)
            rij[a] = LT[a][0] * s[0] + LT[a][1] * s[1] + LT[a][2] * s[2];
    }

    inline double dot3(const double *a, const double *b)
    {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    inline double norm(const double *a)
    {
        return std::sqrt(dot3(a, a));
    }

    // 计算 cos(theta) 及其对 Ri, Rj, Rk 的导数
    inline void costheta_derivatives(const double *rij_hat, double rijinv,
                                     const double *rik_hat, double rikinv,
                                     double dri[3], double drj[3], double drk[3])
    {
        const double costh = dot3(rij_hat, rik_hat);
        for (int k = 0; k < 3; ++k)
            drj[k] = (-costh * rij_hat[k] + rik_hat[k]) * rijinv;
        for (int k = 0; k < 3; ++k)
            drk[k] = (-costh * rik_hat[k] + rij_hat[k]) * rikinv;
        for (int k = 0; k < 3; ++k)
            dri[k] = -(drj[k] + drk[k]);
    }

    inline double exp_m_term(double lambda3, int m, double x)
    {
        // exp( lambda3^m * x^m ); 针对 m==3 的常见情形
        if (lambda3 == 0.0)
            return 1.0;
        double t;
        if (m == 3)
            t = std::pow(lambda3, 3) * (x * x * x);
        else
            t = lambda3 * x; // 回退（非常规m）
        // 防止溢出
        if (t > 69.0776)
            return 1e30;
        if (t < -69.0776)
            return 0.0;
        return std::exp(t);
    }

    // 计算 zeta(i,j) 累积项
    inline double compute_zeta_ij(const Param &p, double rij_shift, double rik_shift,
                                  const double *rij_hat, const double *rik_hat)
    {
        const double costh = dot3(rij_hat, rik_hat);
        const double g = g_theta(costh, p);
        const double ex = exp_m_term(p.lambda3, p.m, rij_shift - rik_shift);
        return fc(rik_shift, p) * g * ex;
    }

    // 计算 d zeta * prefactor 对 Ri,Rj,Rk 的向量贡献（prefactor = -0.5 * fA * db/dzeta）
    inline void zeta_term_derivatives(const Param &p,
                                      double prefactor,
                                      const double *rij_hat, double rij, double rijinv,
                                      const double *rik_hat, double rik, double rikinv,
                                      double dri[3], double drj[3], double drk[3])
    {
        // 结构与主流实现一致
        double fc_rik = fc(rik, p);
        double dfc_rik = fc_d(rik, p);
        double ex = exp_m_term(p.lambda3, p.m, rij - rik);
        double dex_drij, dex_drik;
        if (p.lambda3 == 0.0)
        {
            dex_drij = 0.0;
            dex_drik = 0.0;
        }
        else if (p.m == 3)
        {
            const double coeff = 3.0 * std::pow(p.lambda3, 3) * (rij - rik) * (rij - rik);
            dex_drij = coeff * ex;
            dex_drik = -coeff * ex;
        }
        else
        {
            const double coeff = p.lambda3 * ex;
            dex_drij = coeff;
            dex_drik = -coeff;
        }

        const double costh = dot3(rij_hat, rik_hat);
        const double g = g_theta(costh, p);
        const double dg_dcth = g_theta_dcosth(costh, p);

        double dcos_dri[3], dcos_drj[3], dcos_drk[3];
        costheta_derivatives(rij_hat, rijinv, rik_hat, rikinv, dcos_dri, dcos_drj, dcos_drk);

        // 对 Ri
        for (int k = 0; k < 3; ++k)
        {
            double term = -dfc_rik * g * ex * (rik_hat[k])
                          + fc_rik * dg_dcth * ex * dcos_dri[k]
                          + fc_rik * g * (dex_drik * (rik_hat[k]) + dex_drij * (-rij_hat[k]));
            dri[k] = prefactor * term;
        }

        // 对 Rj:
        for (int k = 0; k < 3; ++k)
        {
            // fc*dg*ex*dcos_drj + fc*g*dex*(∂(rij)/∂Rj = +r̂_ij)
            double term = fc_rik * dg_dcth * ex * dcos_drj[k] + fc_rik * g * dex_drij * (rij_hat[k]);
            drj[k] = prefactor * term;
        }

        // 对 Rk:
        for (int k = 0; k < 3; ++k)
        {
            // dfc*g*ex*(∂rik/∂Rk = +r̂_ik) + fc*dg*ex*dcos_drk + fc*g*dex*(∂(-rik)/∂Rk = -r̂_ik)
            double term = dfc_rik * g * ex * (rik_hat[k]) + fc_rik * dg_dcth * ex * dcos_drk[k] + fc_rik * g * dex_drik * (-rik_hat[k]);
            drk[k] = prefactor * term;
        }
    }

    // 单对 repulsive 力与能量（基于 r_shift）
    inline void repulsive_pair(const Param &p, double r, double &fpair, double &eng)
    {
        const double rsh = p.shift_flag ? (r + p.shift) : r;
        const double fc_r = fc(rsh, p);
        const double dfc_r = fc_d(rsh, p);
        const double exp_r = std::exp(-p.lambda1 * rsh);
        // fpair_base = -A*exp * (fc'(rsh) - fc(rsh)*lam1) / rsh
        double fpair_base = -p.A * exp_r * (dfc_r - fc_r * p.lambda1) / (rsh > 0.0 ? rsh : 1.0);
        // SHIFT 修正：fpair *= (rsh/r)
        if (p.shift_flag && r > 0.0)
            fpair_base *= (rsh / r);
        fpair = fpair_base;
        eng = fc_r * p.A * exp_r;
    }

    // 计算总能量、力
    static void tersoff_compute_all(int num_atoms,
                                    const double *positions,       // 3N
                                    const double *lattice_vectors, // 9, row-major
                                    const Param &p,
                                    double *forces,       // 3N (in/out, will overwrite)
                                    double *energy_out,   // optional
                                    double *virial_tensor // optional 9 comp, accumulate -r_i ⊗ F_i
    )
    {
        // 初始化输出
        if (forces)
            for (int i = 0; i < 3 * num_atoms; ++i)
                forces[i] = 0.0;
        if (energy_out)
            *energy_out = 0.0;
        // 本地累加对 virial（vt += dr ⊗ F），最后取反输出
        double vt[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        // 构造 L^T 与 inv(L^T)
        double L[3][3];
        mat3_from_flat(lattice_vectors, L);
        double LT[3][3];
        mat3_transpose(L, LT);
        double invLT[3][3];
        inv3(LT, invLT);

        const double cutshort = p.cut; // R + D
        const double cutshort_sq = cutshort * cutshort;

        // 预构建邻居（NSQ 简易版）
        std::vector<std::vector<int>> neigh(num_atoms);
        std::vector<double> pair_r(num_atoms * num_atoms, 0.0);          // store r (unshifted) for reuse
        std::vector<double> pair_rsh(num_atoms * num_atoms, 0.0);        // shifted r
        std::vector<double> pair_rijhat(3 * num_atoms * num_atoms, 0.0); // unit vectors

        for (int i = 0; i < num_atoms; ++i)
        {
            const double *ri = &positions[3 * i];
            for (int j = 0; j < num_atoms; ++j)
            {
                if (i == j)
                    continue;
                const double *rj = &positions[3 * j];
                double rij[3];
                mic_displacement(LT, invLT, ri, rj, rij);
                double r = norm(rij);
                double rsh = p.shift_flag ? (r + p.shift) : r;
                if (rsh * rsh < cutshort_sq)
                {
                    neigh[i].push_back(j);
                    pair_r[i * num_atoms + j] = r;
                    pair_rsh[i * num_atoms + j] = rsh;
                    double *hat = &pair_rijhat[3 * (i * num_atoms + j)];
                    if (r > 0.0)
                    {
                        hat[0] = rij[0] / r;
                        hat[1] = rij[1] / r;
                        hat[2] = rij[2] / r;
                    }
                    else
                    {
                        hat[0] = hat[1] = hat[2] = 0.0;
                    }
                }
            }
        }

        double total_energy = 0.0;

        // 一、两体 repulsive 项：仅 i<j 计一次
        for (int i = 0; i < num_atoms; ++i)
        {
            for (int jj = 0; jj < (int)neigh[i].size(); ++jj)
            {
                int j = neigh[i][jj];
                if (j <= i)
                    continue;

                const double r = pair_r[i * num_atoms + j];
                const double rsh = pair_rsh[i * num_atoms + j];
                if (rsh * rsh >= cutshort_sq)
                    continue;

                // repulsive 力与能量
                double fpair, e_pair;
                repulsive_pair(p, r, fpair, e_pair);

                // 作用力（沿 rij，正负分别加到 i/j）
                const double *hat_ij = &pair_rijhat[3 * (i * num_atoms + j)];
                double fij[3] = {fpair * (r * hat_ij[0]), fpair * (r * hat_ij[1]), fpair * (r * hat_ij[2])};
                // 因为 fpair 已含 1/r，乘 r*hat_ij 相当于 delr 分量
                if (forces)
                {
                    forces[3 * i + 0] += fij[0];
                    forces[3 * i + 1] += fij[1];
                    forces[3 * i + 2] += fij[2];
                    forces[3 * j + 0] -= fij[0];
                    forces[3 * j + 1] -= fij[1];
                    forces[3 * j + 2] -= fij[2];
                }
                // vt += delr ⊗ Fij
                {
                    double delr_vec[3] = {r * hat_ij[0], r * hat_ij[1], r * hat_ij[2]};
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b)
                            vt[3 * a + b] += delr_vec[a] * fij[b];
                }
                total_energy += e_pair;
            }
        }

        // 二、三体项（吸引部分 + b_ij 导致的配对力）：
        for (int i = 0; i < num_atoms; ++i)
        {
            // 遍历 I 的短邻居 j
            for (int jj = 0; jj < (int)neigh[i].size(); ++jj)
            {
                int j = neigh[i][jj];
                const double r1 = pair_r[i * num_atoms + j];
                const double r1sh = pair_rsh[i * num_atoms + j];
                if (r1sh * r1sh >= cutshort_sq || r1 <= 0.0)
                    continue;

                const double *r1hat = &pair_rijhat[3 * (i * num_atoms + j)];

                // 先累计 zeta_ij
                double zeta_ij = 0.0;
                for (int kk = 0; kk < (int)neigh[i].size(); ++kk)
                {
                    if (kk == jj)
                        continue;
                    int k = neigh[i][kk];
                    const double r2 = pair_r[i * num_atoms + k];
                    const double r2sh = pair_rsh[i * num_atoms + k];
                    if (r2sh * r2sh >= cutshort_sq || r2 <= 0.0)
                        continue;
                    const double *r2hat = &pair_rijhat[3 * (i * num_atoms + k)];
                    zeta_ij += compute_zeta_ij(p, r1sh, r2sh, r1hat, r2hat);
                }

                // 由 zeta 导致的配对力 + 吸引能量的一半
                const double fa = fA(r1sh, p) * fc(r1sh, p);
                const double fa_d = fA_d(r1sh, p) * fc(r1sh, p) + fA(r1sh, p) * fc_d(r1sh, p);
                const double bij_val = bij(zeta_ij, p);
                const double dbij_dz = bij_dzeta(zeta_ij, p);

                // fforce = 0.5 * bij * fa_d; prefactor = -0.5 * fa * bij'
                double fpair_coeff = 0.5 * bij_val * fa_d; // 沿 r1 方向
                double prefactor = -0.5 * fa * dbij_dz;

                double fpair_vec[3] = {0, 0, 0};
                // fpair = fforce * r1inv; 分量 = delr1[k] * fpair = fforce * hat_k
                for (int k = 0; k < 3; ++k)
                    fpair_vec[k] = fpair_coeff * r1hat[k];

                if (forces)
                {
                    // pair作用：i 加，j 减
                    forces[3 * i + 0] += fpair_vec[0];
                    forces[3 * i + 1] += fpair_vec[1];
                    forces[3 * i + 2] += fpair_vec[2];
                    forces[3 * j + 0] -= fpair_vec[0];
                    forces[3 * j + 1] -= fpair_vec[1];
                    forces[3 * j + 2] -= fpair_vec[2];
                }

                // zeta 配对 virial 采用负号：vt += -(delr1 ⊗ fpair_vec)
                {
                    double delr1_vec[3] = {r1 * r1hat[0], r1 * r1hat[1], r1 * r1hat[2]};
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b)
                            vt[3 * a + b] -= delr1_vec[a] * fpair_vec[b];
                }

                if (energy_out)
                {
                    // 能量：0.5 * bij * fa
                    total_energy += 0.5 * bij_val * fa;
                }

                // 吸引项对 i,j,k 的三体力（由 zeta 的导数给出）
                for (int kk = 0; kk < (int)neigh[i].size(); ++kk)
                {
                    if (kk == jj)
                        continue;
                    int k = neigh[i][kk];
                    const double r2 = pair_r[i * num_atoms + k];
                    const double r2sh = pair_rsh[i * num_atoms + k];
                    if (r2sh * r2sh >= cutshort_sq || r2 <= 0.0)
                        continue;
                    const double *r2hat = &pair_rijhat[3 * (i * num_atoms + k)];

                    double fi[3], fj[3], fk[3];
                    // 与常见做法一致：在导数项中使用移位后的距离 rsh 用于 fc/exp 计算；
                    // 而 1/r 使用 (1/(r - shift)) 的修正。
                    const double r1sh_eff = p.shift_flag ? (r1 + p.shift) : r1;
                    const double r2sh_eff = p.shift_flag ? (r2 + p.shift) : r2;
                    const double rijinv = 1.0 / (p.shift_flag ? (r1 - p.shift) : r1);
                    const double rikinv = 1.0 / (p.shift_flag ? (r2 - p.shift) : r2);
                    zeta_term_derivatives(p, prefactor, r1hat, r1sh_eff, rijinv, r2hat, r2sh_eff, rikinv, fi, fj, fk);

                    if (forces)
                    {
                        forces[3 * i + 0] += fi[0];
                        forces[3 * i + 1] += fi[1];
                        forces[3 * i + 2] += fi[2];
                        forces[3 * j + 0] += fj[0];
                        forces[3 * j + 1] += fj[1];
                        forces[3 * j + 2] += fj[2];
                        forces[3 * k + 0] += fk[0];
                        forces[3 * k + 1] += fk[1];
                        forces[3 * k + 2] += fk[2];
                    }
                    // vt += delr_ij ⊗ fj + delr_ik ⊗ fk
                    {
                        const double *ri = &positions[3 * i];
                        const double *rjpos = &positions[3 * j];
                        const double *rkpos = &positions[3 * k];
                        double delr_ij[3], delr_ik[3];
                        mic_displacement(LT, invLT, ri, rjpos, delr_ij);
                        mic_displacement(LT, invLT, ri, rkpos, delr_ik);
                        for (int a = 0; a < 3; ++a)
                        {
                            for (int b = 0; b < 3; ++b)
                            {
                                vt[3 * a + b] += delr_ij[a] * fj[b];
                                vt[3 * a + b] += delr_ik[a] * fk[b];
                            }
                        }
                    }
                }
            }
        }

        // 输出时取反：项目定义 σ = -virial/V
        if (virial_tensor)
        {
            for (int i = 0; i < 9; ++i)
                virial_tensor[i] = -vt[i];
        }
        if (energy_out)
            *energy_out = total_energy;
    }

} // namespace tersoff_min

extern "C"
{
    // 计算能量（eV）——通用参数版
    double calculate_tersoff_energy(
        int num_atoms,
        const double *positions,
        const double *lattice_vectors,
        double A, double B,
        double lambda1, double lambda2, double lambda3,
        double beta, double n, double c, double d, double h,
        double R, double D,
        int m,
        int shift_flag,
        double delta)
    {
        tersoff_min::Param p{A, B, lambda1, lambda2, lambda3, beta, n, c, d, h, R, D, m};
        p.cut = R + D;
        p.cutsq = p.cut * p.cut;
        p.shift_flag = shift_flag;
        p.shift = delta;

        double energy = 0.0;
        tersoff_min::tersoff_compute_all(
            num_atoms, positions, lattice_vectors, p,
            /*forces*/ nullptr, &energy, /*virial*/ nullptr);
        return energy;
    }

    // 计算力（写入 forces，单位 eV/Å）——通用参数版
    void calculate_tersoff_forces(
        int num_atoms,
        const double *positions,
        const double *lattice_vectors,
        double *forces,
        double A, double B,
        double lambda1, double lambda2, double lambda3,
        double beta, double n, double c, double d, double h,
        double R, double D,
        int m,
        int shift_flag,
        double delta)
    {
        tersoff_min::Param p{A, B, lambda1, lambda2, lambda3, beta, n, c, d, h, R, D, m};
        p.cut = R + D;
        p.cutsq = p.cut * p.cut;
        p.shift_flag = shift_flag;
        p.shift = delta;

        tersoff_min::tersoff_compute_all(
            num_atoms, positions, lattice_vectors, p,
            forces, /*energy*/ nullptr, /*virial*/ nullptr);
    }

    // 计算维里张量（不除体积，9个分量，行主序）——通用参数版
    void calculate_tersoff_virial(
        int num_atoms,
        const double *positions,
        const double *lattice_vectors,
        double *virial_tensor,
        double A, double B,
        double lambda1, double lambda2, double lambda3,
        double beta, double n, double c, double d, double h,
        double R, double D,
        int m,
        int shift_flag,
        double delta)
    {
        tersoff_min::Param p{A, B, lambda1, lambda2, lambda3, beta, n, c, d, h, R, D, m};
        p.cut = R + D;
        p.cutsq = p.cut * p.cut;
        p.shift_flag = shift_flag;
        p.shift = delta;

        std::vector<double> forces(3 * num_atoms, 0.0);
        tersoff_min::tersoff_compute_all(
            num_atoms, positions, lattice_vectors, p,
            forces.data(), /*energy*/ nullptr, virial_tensor);
    }

    // ================== C(1988) 默认参数封装 ==================
    static inline tersoff_min::Param default_c1988_param(int shift_flag, double delta)
    {
        tersoff_min::Param p{
            /*A*/ 1393.6,
            /*B*/ 346.74,
            /*lambda1*/ 3.4879,
            /*lambda2*/ 2.2119,
            /*lambda3*/ 0.0,
            /*beta*/ 1.5724e-7,
            /*n*/ 0.72751,
            /*c*/ 38049.0,
            /*d*/ 4.3484,
            /*h*/ -0.57058,
            /*R*/ 1.95,
            /*D*/ 0.15,
            /*m*/ 3};
        p.cut = p.R + p.D;
        p.cutsq = p.cut * p.cut;
        p.shift_flag = shift_flag;
        p.shift = delta;
        return p;
    }

    double calculate_tersoff_c1988_energy(
        int num_atoms,
        const double *positions,
        const double *lattice_vectors,
        int shift_flag,
        double delta)
    {
        auto p = default_c1988_param(shift_flag, delta);
        double energy = 0.0;
        tersoff_min::tersoff_compute_all(
            num_atoms, positions, lattice_vectors, p,
            /*forces*/ nullptr, &energy, /*virial*/ nullptr);
        return energy;
    }

    void calculate_tersoff_c1988_forces(
        int num_atoms,
        const double *positions,
        const double *lattice_vectors,
        double *forces,
        int shift_flag,
        double delta)
    {
        auto p = default_c1988_param(shift_flag, delta);
        tersoff_min::tersoff_compute_all(
            num_atoms, positions, lattice_vectors, p,
            forces, /*energy*/ nullptr, /*virial*/ nullptr);
    }

    void calculate_tersoff_c1988_virial(
        int num_atoms,
        const double *positions,
        const double *lattice_vectors,
        double *virial_tensor,
        int shift_flag,
        double delta)
    {
        auto p = default_c1988_param(shift_flag, delta);
        std::vector<double> forces(3 * num_atoms, 0.0);
        tersoff_min::tersoff_compute_all(
            num_atoms, positions, lattice_vectors, p,
            forces.data(), /*energy*/ nullptr, virial_tensor);
    }
}

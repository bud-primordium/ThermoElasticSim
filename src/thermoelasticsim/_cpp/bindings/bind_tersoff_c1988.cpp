/**
 * @file bind_tersoff.cpp
 * @brief Tersoff (C) 势的 pybind11 绑定
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern "C"
{
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
        double delta);

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
        double delta);

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
        double delta);
    // default C(1988)
    double calculate_tersoff_c1988_energy(
        int num_atoms,
        const double *positions,
        const double *lattice_vectors,
        int shift_flag,
        double delta);

    void calculate_tersoff_c1988_forces(
        int num_atoms,
        const double *positions,
        const double *lattice_vectors,
        double *forces,
        int shift_flag,
        double delta);

    void calculate_tersoff_c1988_virial(
        int num_atoms,
        const double *positions,
        const double *lattice_vectors,
        double *virial_tensor,
        int shift_flag,
        double delta);
}

void bind_tersoff_c1988(py::module_ &m)
{
    m.def(
        "calculate_tersoff_energy",
        [](int num_atoms,
           py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           py::array_t<double, py::array::c_style | py::array::forcecast> lattice_vectors,
           double A, double B,
           double lambda1, double lambda2, double lambda3,
           double beta, double n, double c, double d, double h,
           double R, double D,
           int m,
           bool shift_flag,
           double delta)
        {
            if (positions.ndim() != 1 || positions.size() != 3 * num_atoms)
                throw std::runtime_error("positions must be 1D with length 3*num_atoms");
            if (lattice_vectors.ndim() != 1 || lattice_vectors.size() != 9)
                throw std::runtime_error("lattice_vectors must be 1D with length 9");
            return calculate_tersoff_energy(
                num_atoms,
                positions.data(),
                lattice_vectors.data(),
                A, B,
                lambda1, lambda2, lambda3,
                beta, n, c, d, h,
                R, D,
                m,
                shift_flag ? 1 : 0,
                delta);
        },
        py::arg("num_atoms"),
        py::arg("positions"),
        py::arg("lattice_vectors"),
        py::arg("A"), py::arg("B"),
        py::arg("lambda1"), py::arg("lambda2"), py::arg("lambda3"),
        py::arg("beta"), py::arg("n"), py::arg("c"), py::arg("d"), py::arg("h"),
        py::arg("R"), py::arg("D"),
        py::arg("m") = 3,
        py::arg("shift_flag") = false,
        py::arg("delta") = 0.0,
        "计算 Tersoff 势能 (eV) - 通用参数版");

    m.def(
        "calculate_tersoff_forces",
        [](int num_atoms,
           py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           py::array_t<double, py::array::c_style | py::array::forcecast> lattice_vectors,
           py::array_t<double, py::array::c_style | py::array::forcecast> forces,
           double A, double B,
           double lambda1, double lambda2, double lambda3,
           double beta, double n, double c, double d, double h,
           double R, double D,
           int m,
           bool shift_flag,
           double delta)
        {
            if (positions.ndim() != 1 || positions.size() != 3 * num_atoms)
                throw std::runtime_error("positions must be 1D with length 3*num_atoms");
            if (lattice_vectors.ndim() != 1 || lattice_vectors.size() != 9)
                throw std::runtime_error("lattice_vectors must be 1D with length 9");
            if (forces.ndim() != 1 || forces.size() != 3 * num_atoms)
                throw std::runtime_error("forces must be 1D with length 3*num_atoms");
            calculate_tersoff_forces(
                num_atoms,
                positions.data(),
                lattice_vectors.data(),
                forces.mutable_data(),
                A, B,
                lambda1, lambda2, lambda3,
                beta, n, c, d, h,
                R, D,
                m,
                shift_flag ? 1 : 0,
                delta);
        },
        py::arg("num_atoms"),
        py::arg("positions"),
        py::arg("lattice_vectors"),
        py::arg("forces"),
        py::arg("A"), py::arg("B"),
        py::arg("lambda1"), py::arg("lambda2"), py::arg("lambda3"),
        py::arg("beta"), py::arg("n"), py::arg("c"), py::arg("d"), py::arg("h"),
        py::arg("R"), py::arg("D"),
        py::arg("m") = 3,
        py::arg("shift_flag") = false,
        py::arg("delta") = 0.0,
        "计算 Tersoff 作用力 (eV/Å) - 通用参数版");

    m.def(
        "calculate_tersoff_virial",
        [](int num_atoms,
           py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           py::array_t<double, py::array::c_style | py::array::forcecast> lattice_vectors,
           double A, double B,
           double lambda1, double lambda2, double lambda3,
           double beta, double n, double c, double d, double h,
           double R, double D,
           int m,
           bool shift_flag,
           double delta)
        {
            if (positions.ndim() != 1 || positions.size() != 3 * num_atoms)
                throw std::runtime_error("positions must be 1D with length 3*num_atoms");
            if (lattice_vectors.ndim() != 1 || lattice_vectors.size() != 9)
                throw std::runtime_error("lattice_vectors must be 1D with length 9");
            py::array_t<double> vir(9);
            calculate_tersoff_virial(
                num_atoms,
                positions.data(),
                lattice_vectors.data(),
                vir.mutable_data(),
                A, B,
                lambda1, lambda2, lambda3,
                beta, n, c, d, h,
                R, D,
                m,
                shift_flag ? 1 : 0,
                delta);
            return vir;
        },
        py::arg("num_atoms"),
        py::arg("positions"),
        py::arg("lattice_vectors"),
        py::arg("A"), py::arg("B"),
        py::arg("lambda1"), py::arg("lambda2"), py::arg("lambda3"),
        py::arg("beta"), py::arg("n"), py::arg("c"), py::arg("d"), py::arg("h"),
        py::arg("R"), py::arg("D"),
        py::arg("m") = 3,
        py::arg("shift_flag") = false,
        py::arg("delta") = 0.0,
        "计算 Tersoff 的维里张量（未除体积，返回形状(9,)） - 通用参数版");

    // ============ 默认 C(1988) 参数的绑定（不需显式传参） ============
    m.def(
        "calculate_tersoff_c1988_energy",
        [](int num_atoms,
           py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           py::array_t<double, py::array::c_style | py::array::forcecast> lattice_vectors,
           bool shift_flag,
           double delta)
        {
            if (positions.ndim() != 1 || positions.size() != 3 * num_atoms)
                throw std::runtime_error("positions must be 1D with length 3*num_atoms");
            if (lattice_vectors.ndim() != 1 || lattice_vectors.size() != 9)
                throw std::runtime_error("lattice_vectors must be 1D with length 9");
            return calculate_tersoff_c1988_energy(
                num_atoms,
                positions.data(),
                lattice_vectors.data(),
                shift_flag ? 1 : 0,
                delta);
        },
        py::arg("num_atoms"),
        py::arg("positions"),
        py::arg("lattice_vectors"),
        py::arg("shift_flag") = false,
        py::arg("delta") = 0.0,
        "计算 Tersoff C(1988) 势能 (eV)，默认参数内置");

    m.def(
        "calculate_tersoff_c1988_forces",
        [](int num_atoms,
           py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           py::array_t<double, py::array::c_style | py::array::forcecast> lattice_vectors,
           py::array_t<double, py::array::c_style | py::array::forcecast> forces,
           bool shift_flag,
           double delta)
        {
            if (positions.ndim() != 1 || positions.size() != 3 * num_atoms)
                throw std::runtime_error("positions must be 1D with length 3*num_atoms");
            if (lattice_vectors.ndim() != 1 || lattice_vectors.size() != 9)
                throw std::runtime_error("lattice_vectors must be 1D with length 9");
            if (forces.ndim() != 1 || forces.size() != 3 * num_atoms)
                throw std::runtime_error("forces must be 1D with length 3*num_atoms");
            calculate_tersoff_c1988_forces(
                num_atoms,
                positions.data(),
                lattice_vectors.data(),
                forces.mutable_data(),
                shift_flag ? 1 : 0,
                delta);
        },
        py::arg("num_atoms"),
        py::arg("positions"),
        py::arg("lattice_vectors"),
        py::arg("forces"),
        py::arg("shift_flag") = false,
        py::arg("delta") = 0.0,
        "计算 Tersoff C(1988) 作用力 (eV/Å)，默认参数内置");

    m.def(
        "calculate_tersoff_c1988_virial",
        [](int num_atoms,
           py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           py::array_t<double, py::array::c_style | py::array::forcecast> lattice_vectors,
           bool shift_flag,
           double delta)
        {
            if (positions.ndim() != 1 || positions.size() != 3 * num_atoms)
                throw std::runtime_error("positions must be 1D with length 3*num_atoms");
            if (lattice_vectors.ndim() != 1 || lattice_vectors.size() != 9)
                throw std::runtime_error("lattice_vectors must be 1D with length 9");
            py::array_t<double> vir(9);
            calculate_tersoff_c1988_virial(
                num_atoms,
                positions.data(),
                lattice_vectors.data(),
                vir.mutable_data(),
                shift_flag ? 1 : 0,
                delta);
            return vir;
        },
        py::arg("num_atoms"),
        py::arg("positions"),
        py::arg("lattice_vectors"),
        py::arg("shift_flag") = false,
        py::arg("delta") = 0.0,
        "计算 Tersoff C(1988) 的维里张量（未除体积），默认参数内置");
}

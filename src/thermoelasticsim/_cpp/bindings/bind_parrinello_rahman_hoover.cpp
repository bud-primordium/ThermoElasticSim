/**
 * @file bind_parrinello_rahman_hoover.cpp
 * @brief Parrinello-Rahman-Hoover 晶胞动力学的 pybind11 绑定
 * @author Gilbert Young
 * @date 2025-08-12
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern "C" {
    // Parrinello-Rahman-Hoover 晶胞动力学函数 C 接口声明
    void parrinello_rahman_hoover(
        double dt,
        int num_atoms,
        const double *masses,
        double *velocities,
        const double *forces,
        double *lattice_vectors,
        double *xi,
        const double *Q,
        const double *total_stress,
        const double *target_pressure,
        double W);
}

void bind_parrinello_rahman_hoover(py::module_ &m) {
    // ============ Parrinello-Rahman-Hoover 晶胞动力学绑定 ============

    // 绑定 Parrinello-Rahman-Hoover 晶胞动力学函数
    m.def(
        "parrinello_rahman_hoover",
        [](double dt,
           int num_atoms,
           py::array_t<double, py::array::c_style | py::array::forcecast> masses,
           py::array_t<double, py::array::c_style | py::array::forcecast> velocities,
           py::array_t<double, py::array::c_style | py::array::forcecast> forces,
           py::array_t<double, py::array::c_style | py::array::forcecast> lattice_vectors,
           py::array_t<double, py::array::c_style | py::array::forcecast> xi,
           py::array_t<double, py::array::c_style | py::array::forcecast> Q,
           py::array_t<double, py::array::c_style | py::array::forcecast> total_stress,
           py::array_t<double, py::array::c_style | py::array::forcecast> target_pressure,
           double W) {
            // 输入参数验证
            if (masses.ndim() != 1 || masses.size() != num_atoms) {
                throw std::runtime_error("masses must be 1D with length num_atoms");
            }
            if (velocities.ndim() != 1 || velocities.size() != 3 * num_atoms) {
                throw std::runtime_error("velocities must be 1D with length 3*num_atoms");
            }
            if (forces.ndim() != 1 || forces.size() != 3 * num_atoms) {
                throw std::runtime_error("forces must be 1D with length 3*num_atoms");
            }
            if (lattice_vectors.ndim() != 1 || lattice_vectors.size() != 9) {
                throw std::runtime_error("lattice_vectors must be 1D with length 9");
            }
            if (xi.ndim() != 1 || xi.size() != 9) {
                throw std::runtime_error("xi must be 1D with length 9");
            }
            if (Q.ndim() != 1 || Q.size() != 9) {
                throw std::runtime_error("Q must be 1D with length 9");
            }
            if (total_stress.ndim() != 1 || total_stress.size() != 9) {
                throw std::runtime_error("total_stress must be 1D with length 9");
            }
            if (target_pressure.ndim() != 1 || target_pressure.size() != 9) {
                throw std::runtime_error("target_pressure must be 1D with length 9");
            }

            // 获取数据指针
            const double *masses_ptr = masses.data();
            double *velocities_ptr = velocities.mutable_data();
            const double *forces_ptr = forces.data();
            double *lattice_vectors_ptr = lattice_vectors.mutable_data();
            double *xi_ptr = xi.mutable_data();
            const double *Q_ptr = Q.data();
            const double *total_stress_ptr = total_stress.data();
            const double *target_pressure_ptr = target_pressure.data();

            // 调用 C 函数执行 Parrinello-Rahman-Hoover 晶胞动力学
            parrinello_rahman_hoover(
                dt,
                num_atoms,
                masses_ptr,
                velocities_ptr,
                forces_ptr,
                lattice_vectors_ptr,
                xi_ptr,
                Q_ptr,
                total_stress_ptr,
                target_pressure_ptr,
                W);

            return py::none();
        },
        py::arg("dt"),
        py::arg("num_atoms"),
        py::arg("masses"),
        py::arg("velocities"),
        py::arg("forces"),
        py::arg("lattice_vectors"),
        py::arg("xi"),
        py::arg("Q"),
        py::arg("total_stress"),
        py::arg("target_pressure"),
        py::arg("W"),
        "应用 Parrinello-Rahman-Hoover 算法进行晶胞动力学演化");
}
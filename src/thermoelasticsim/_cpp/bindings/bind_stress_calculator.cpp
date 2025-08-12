/**
 * @file bind_stress_calculator.cpp
 * @brief 应力张量计算函数的 pybind11 绑定
 * @author Gilbert Young
 * @date 2025-08-12
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern "C" {
    // 应力张量计算函数 C 接口声明
    void compute_stress(
        int num_atoms,
        const double *positions,
        const double *velocities,
        const double *forces,
        const double *masses,
        double volume,
        const double *box_lengths,
        double *stress_tensor);
}

void bind_stress_calculator(py::module_ &m) {
    // ============ 应力张量计算绑定 ============

    // 绑定应力张量计算函数
    m.def(
        "compute_stress",
        [](int num_atoms,
           py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           py::array_t<double, py::array::c_style | py::array::forcecast> velocities,
           py::array_t<double, py::array::c_style | py::array::forcecast> forces,
           py::array_t<double, py::array::c_style | py::array::forcecast> masses,
           double volume,
           py::array_t<double, py::array::c_style | py::array::forcecast> box_lengths,
           py::array_t<double, py::array::c_style | py::array::forcecast> stress_tensor) {
            // 输入参数验证
            if (positions.ndim() != 1 || positions.size() != 3 * num_atoms) {
                throw std::runtime_error("positions must be 1D with length 3*num_atoms");
            }
            if (velocities.ndim() != 1 || velocities.size() != 3 * num_atoms) {
                throw std::runtime_error("velocities must be 1D with length 3*num_atoms");
            }
            if (forces.ndim() != 1 || forces.size() != 3 * num_atoms) {
                throw std::runtime_error("forces must be 1D with length 3*num_atoms");
            }
            if (masses.ndim() != 1 || masses.size() != num_atoms) {
                throw std::runtime_error("masses must be 1D with length num_atoms");
            }
            if (box_lengths.ndim() != 1 || box_lengths.size() != 3) {
                throw std::runtime_error("box_lengths must be 1D with length 3");
            }
            if (stress_tensor.ndim() != 1 || stress_tensor.size() != 9) {
                throw std::runtime_error("stress_tensor must be 1D with length 9");
            }

            // 获取数据指针
            const double *pos_ptr = positions.data();
            const double *vel_ptr = velocities.data();
            const double *forces_ptr = forces.data();
            const double *masses_ptr = masses.data();
            const double *box_ptr = box_lengths.data();
            double *stress_ptr = stress_tensor.mutable_data();

            // 调用 C 函数计算应力张量
            compute_stress(
                num_atoms,
                pos_ptr,
                vel_ptr,
                forces_ptr,
                masses_ptr,
                volume,
                box_ptr,
                stress_ptr);

            return py::none();
        },
        py::arg("num_atoms"),
        py::arg("positions"),
        py::arg("velocities"),
        py::arg("forces"),
        py::arg("masses"),
        py::arg("volume"),
        py::arg("box_lengths"),
        py::arg("stress_tensor"),
        "计算系统应力张量");
}
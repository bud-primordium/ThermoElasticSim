/**
 * @file bind_nose_hoover.cpp
 * @brief Nose-Hoover 恒温器的 pybind11 绑定
 * @author Gilbert Young
 * @date 2025-08-12
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern "C" {
    // Nose-Hoover 恒温器函数 C 接口声明
    void nose_hoover(
        double dt,
        int num_atoms,
        const double *masses,
        double *velocities,
        const double *forces,
        double *xi,
        double Q,
        double target_temperature);
}

void bind_nose_hoover(py::module_ &m) {
    // ============ Nose-Hoover 恒温器绑定 ============

    // 绑定 Nose-Hoover 恒温器函数
    m.def(
        "nose_hoover",
        [](double dt,
           int num_atoms,
           py::array_t<double, py::array::c_style | py::array::forcecast> masses,
           py::array_t<double, py::array::c_style | py::array::forcecast> velocities,
           py::array_t<double, py::array::c_style | py::array::forcecast> forces,
           py::array_t<double, py::array::c_style | py::array::forcecast> xi_array,
           double Q,
           double target_temperature) {
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
            if (xi_array.ndim() != 1 || xi_array.size() != 1) {
                throw std::runtime_error("xi_array must be 1D with length 1");
            }

            // 获取数据指针
            const double *masses_ptr = masses.data();
            double *velocities_ptr = velocities.mutable_data();
            const double *forces_ptr = forces.data();
            double *xi_ptr = xi_array.mutable_data();

            // 调用 C 函数执行 Nose-Hoover 恒温器
            nose_hoover(
                dt,
                num_atoms,
                masses_ptr,
                velocities_ptr,
                forces_ptr,
                xi_ptr,
                Q,
                target_temperature);

            return py::none();
        },
        py::arg("dt"),
        py::arg("num_atoms"),
        py::arg("masses"),
        py::arg("velocities"),
        py::arg("forces"),
        py::arg("xi_array"),
        py::arg("Q"),
        py::arg("target_temperature"),
        "应用 Nose-Hoover 恒温器进行温度控制");
}
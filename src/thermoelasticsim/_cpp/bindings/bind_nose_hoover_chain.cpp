/**
 * @file bind_nose_hoover_chain.cpp
 * @brief Nose-Hoover 链恒温器的 pybind11 绑定
 * @author Gilbert Young
 * @date 2025-08-12
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern "C" {
    // Nose-Hoover 链恒温器函数 C 接口声明
    void nose_hoover_chain(
        double dt,
        int num_atoms,
        const double *masses,
        double *velocities,
        const double *forces,
        double *xi_chain,
        const double *Q,
        int chain_length,
        double target_temperature);
}

void bind_nose_hoover_chain(py::module_ &m) {
    // ============ Nose-Hoover 链恒温器绑定 ============

    // 绑定 Nose-Hoover 链恒温器函数
    m.def(
        "nose_hoover_chain",
        [](double dt,
           int num_atoms,
           py::array_t<double, py::array::c_style | py::array::forcecast> masses,
           py::array_t<double, py::array::c_style | py::array::forcecast> velocities,
           py::array_t<double, py::array::c_style | py::array::forcecast> forces,
           py::array_t<double, py::array::c_style | py::array::forcecast> xi_chain,
           py::array_t<double, py::array::c_style | py::array::forcecast> Q,
           int chain_length,
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
            if (xi_chain.ndim() != 1 || xi_chain.size() != chain_length) {
                throw std::runtime_error("xi_chain size must equal chain_length");
            }
            if (Q.ndim() != 1 || Q.size() != chain_length) {
                throw std::runtime_error("Q size must equal chain_length");
            }

            // 获取数据指针
            const double *masses_ptr = masses.data();
            double *velocities_ptr = velocities.mutable_data();
            const double *forces_ptr = forces.data();
            double *xi_chain_ptr = xi_chain.mutable_data();
            const double *Q_ptr = Q.data();

            // 调用 C 函数执行 Nose-Hoover 链恒温器
            nose_hoover_chain(
                dt,
                num_atoms,
                masses_ptr,
                velocities_ptr,
                forces_ptr,
                xi_chain_ptr,
                Q_ptr,
                chain_length,
                target_temperature);

            return py::none();
        },
        py::arg("dt"),
        py::arg("num_atoms"),
        py::arg("masses"),
        py::arg("velocities"),
        py::arg("forces"),
        py::arg("xi_chain"),
        py::arg("Q"),
        py::arg("chain_length"),
        py::arg("target_temperature"),
        "应用 Nose-Hoover 链恒温器进行温度控制");
}
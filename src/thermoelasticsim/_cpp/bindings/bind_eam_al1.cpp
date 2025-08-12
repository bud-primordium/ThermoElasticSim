/**
 * @file bind_eam_al1.cpp
 * @brief EAM Al1 势能函数的 pybind11 绑定
 * @author Gilbert Young
 * @date 2025-08-12
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern "C" {
    // EAM Al1 势能函数 C 接口声明
    void calculate_eam_al1_energy(
        int num_atoms,
        const double *positions,
        const double *box_lengths,
        double *energy);

    void calculate_eam_al1_forces(
        int num_atoms,
        const double *positions,
        const double *box_lengths,
        double *forces);
}

void bind_eam_al1(py::module_ &m) {
    // ============ EAM Al1 势函数绑定 ============

    // 绑定能量计算函数
    m.def(
        "calculate_eam_al1_energy",
        [](int num_atoms,
           py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           py::array_t<double, py::array::c_style | py::array::forcecast> box_lengths) {
            // 输入参数验证
            if (positions.ndim() != 1 || positions.size() != 3 * num_atoms) {
                throw std::runtime_error("positions must be 1D with length 3*num_atoms");
            }
            if (box_lengths.ndim() != 1 || box_lengths.size() != 3) {
                throw std::runtime_error("box_lengths must be 1D with length 3");
            }

            // 获取数据指针
            const double *pos_ptr = positions.data();
            const double *box_ptr = box_lengths.data();
            double energy;

            // 调用 C 函数计算能量
            calculate_eam_al1_energy(
                num_atoms,
                pos_ptr,
                box_ptr,
                &energy);
            return energy;
        },
        py::arg("num_atoms"),
        py::arg("positions"),
        py::arg("box_lengths"),
        "计算 EAM Al1 势能总能量");

    // 绑定力计算函数
    m.def(
        "calculate_eam_al1_forces",
        [](int num_atoms,
           py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           py::array_t<double, py::array::c_style | py::array::forcecast> box_lengths,
           py::array_t<double, py::array::c_style | py::array::forcecast> forces) {
            // 输入参数验证
            if (positions.ndim() != 1 || positions.size() != 3 * num_atoms) {
                throw std::runtime_error("positions must be 1D with length 3*num_atoms");
            }
            if (box_lengths.ndim() != 1 || box_lengths.size() != 3) {
                throw std::runtime_error("box_lengths must be 1D with length 3");
            }
            if (forces.ndim() != 1 || forces.size() != 3 * num_atoms) {
                throw std::runtime_error("forces must be 1D with length 3*num_atoms");
            }

            // 获取数据指针
            const double *pos_ptr = positions.data();
            const double *box_ptr = box_lengths.data();
            double *forces_ptr = forces.mutable_data();

            // 调用 C 函数计算力
            calculate_eam_al1_forces(
                num_atoms,
                pos_ptr,
                box_ptr,
                forces_ptr);

            return py::none();
        },
        py::arg("num_atoms"),
        py::arg("positions"),
        py::arg("box_lengths"),
        py::arg("forces"),
        "计算 EAM Al1 势能作用力");
}
/**
 * @file bind_lennard_jones.cpp
 * @brief Lennard-Jones 势能函数的 pybind11 绑定
 * @author Gilbert Young
 * @date 2025-08-12
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern "C" {
    // Lennard-Jones 势能函数 C 接口声明
    void calculate_lj_forces(
        int num_atoms,
        const double *positions,
        double *forces,
        double epsilon,
        double sigma,
        double cutoff,
        const double *box_lengths,
        const int *neighbor_pairs,
        int num_pairs);

    double calculate_lj_energy(
        int num_atoms,
        const double *positions,
        double epsilon,
        double sigma,
        double cutoff,
        const double *box_lengths,
        const int *neighbor_pairs,
        int num_pairs);
}

void bind_lennard_jones(py::module_ &m) {
    // ============ Lennard-Jones 势函数绑定 ============
    
    // 绑定能量计算函数
    m.def(
        "calculate_lj_energy",
        [](int num_atoms,
           py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           double epsilon,
           double sigma,
           double cutoff,
           py::array_t<double, py::array::c_style | py::array::forcecast> box_lengths,
           py::array_t<int,    py::array::c_style | py::array::forcecast> neighbor_pairs,
           int num_pairs) {
            // 输入参数验证
            if (positions.ndim() != 1 || positions.size() != 3 * num_atoms) {
                throw std::runtime_error("positions must be 1D with length 3*num_atoms");
            }
            if (box_lengths.ndim() != 1 || box_lengths.size() != 3) {
                throw std::runtime_error("box_lengths must be 1D with length 3");
            }
            if (neighbor_pairs.ndim() != 1 || neighbor_pairs.size() != 2 * num_pairs) {
                throw std::runtime_error("neighbor_pairs must be 1D with length 2*num_pairs");
            }

            // 获取数据指针
            const double *pos_ptr = positions.data();
            const double *box_ptr = box_lengths.data();
            const int *pairs_ptr = neighbor_pairs.data();

            // 调用 C 函数计算能量
            double energy = calculate_lj_energy(
                num_atoms,
                pos_ptr,
                epsilon,
                sigma,
                cutoff,
                box_ptr,
                pairs_ptr,
                num_pairs);
            return energy;
        },
        py::arg("num_atoms"),
        py::arg("positions"),
        py::arg("epsilon"),
        py::arg("sigma"),
        py::arg("cutoff"),
        py::arg("box_lengths"),
        py::arg("neighbor_pairs"),
        py::arg("num_pairs"),
        "计算 Lennard-Jones 势能总能量");

    // 绑定力计算函数
    m.def(
        "calculate_lj_forces",
        [](int num_atoms,
           py::array_t<double, py::array::c_style | py::array::forcecast> positions,
           py::array_t<double, py::array::c_style | py::array::forcecast> forces,
           double epsilon,
           double sigma,
           double cutoff,
           py::array_t<double, py::array::c_style | py::array::forcecast> box_lengths,
           py::array_t<int,    py::array::c_style | py::array::forcecast> neighbor_pairs,
           int num_pairs) {
            // 输入参数验证
            if (positions.ndim() != 1 || positions.size() != 3 * num_atoms) {
                throw std::runtime_error("positions must be 1D with length 3*num_atoms");
            }
            if (forces.ndim() != 1 || forces.size() != 3 * num_atoms) {
                throw std::runtime_error("forces must be 1D with length 3*num_atoms");
            }
            if (box_lengths.ndim() != 1 || box_lengths.size() != 3) {
                throw std::runtime_error("box_lengths must be 1D with length 3");
            }
            if (neighbor_pairs.ndim() != 1 || neighbor_pairs.size() != 2 * num_pairs) {
                throw std::runtime_error("neighbor_pairs must be 1D with length 2*num_pairs");
            }

            // 获取数据指针
            const double *pos_ptr = positions.data();
            double *forces_ptr = forces.mutable_data();
            const double *box_ptr = box_lengths.data();
            const int *pairs_ptr = neighbor_pairs.data();

            // 调用 C 函数计算力
            calculate_lj_forces(
                num_atoms,
                pos_ptr,
                forces_ptr,
                epsilon,
                sigma,
                cutoff,
                box_ptr,
                pairs_ptr,
                num_pairs);

            return py::none();
        },
        py::arg("num_atoms"),
        py::arg("positions"),
        py::arg("forces"),
        py::arg("epsilon"),
        py::arg("sigma"),
        py::arg("cutoff"),
        py::arg("box_lengths"),
        py::arg("neighbor_pairs"),
        py::arg("num_pairs"),
        "计算 Lennard-Jones 势能作用力");
}
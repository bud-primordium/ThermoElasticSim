#include <pybind11/pybind11.h>

namespace py = pybind11;

// 声明各个模块的绑定函数
void bind_lennard_jones(py::module_ &m);
void bind_eam_al1(py::module_ &m);
void bind_eam_cu1(py::module_ &m);
void bind_stress_calculator(py::module_ &m);
void bind_nose_hoover(py::module_ &m);
void bind_nose_hoover_chain(py::module_ &m);
void bind_parrinello_rahman_hoover(py::module_ &m);

PYBIND11_MODULE(_cpp_core, m) {
    m.doc() = "ThermoElasticSim C++ 核心模块 - 包含势函数、恒温器和动力学算法";

    // 绑定势函数模块
    bind_lennard_jones(m);
    bind_eam_al1(m);
    bind_eam_cu1(m);

    // 绑定计算工具模块
    bind_stress_calculator(m);

    // 绑定恒温器模块
    bind_nose_hoover(m);
    bind_nose_hoover_chain(m);

    // 绑定动力学算法模块
    bind_parrinello_rahman_hoover(m);
}

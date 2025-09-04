cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_cpp_core MODULE
    src/thermoelasticsim/_cpp/bindings.cpp
    src/thermoelasticsim/_cpp/bindings/bind_lennard_jones.cpp
    src/thermoelasticsim/_cpp/bindings/bind_eam_al1.cpp
    src/thermoelasticsim/_cpp/bindings/bind_eam_cu1.cpp
    src/thermoelasticsim/_cpp/bindings/bind_tersoff_c1988.cpp
    src/thermoelasticsim/_cpp/bindings/bind_stress_calculator.cpp
    src/thermoelasticsim/_cpp/bindings/bind_nose_hoover.cpp
    src/thermoelasticsim/_cpp/bindings/bind_nose_hoover_chain.cpp
    src/thermoelasticsim/_cpp/bindings/bind_parrinello_rahman_hoover.cpp
    src/thermoelasticsim/_cpp/lennard_jones.cpp
    src/thermoelasticsim/_cpp/eam_al1.cpp
    src/thermoelasticsim/_cpp/eam_cu1.cpp
    src/thermoelasticsim/_cpp/tersoff_c1988.cpp
    src/thermoelasticsim/_cpp/nose_hoover.cpp
    src/thermoelasticsim/_cpp/nose_hoover_chain.cpp
    src/thermoelasticsim/_cpp/stress_calculator.cpp
    src/thermoelasticsim/_cpp/parrinello_rahman_hoover.cpp
)

target_compile_features(_cpp_core PRIVATE cxx_std_11)

# 让 scikit-build-core 负责安装/可编辑链接，不强制输出到源目录，
# 以避免在仓库根目录产生 in-source 构建的临时文件。

# 安装到包内
install(TARGETS _cpp_core DESTINATION ${SKBUILD_PROJECT_NAME} COMPONENT python)

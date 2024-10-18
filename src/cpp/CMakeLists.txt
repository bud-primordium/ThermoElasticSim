cmake_minimum_required(VERSION 3.0)
project(cpp_libs)

# 使用C++ 11标准
set(CMAKE_CXX_STANDARD 11)

# 设置生成位置无关代码（适用于共享库）
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 设置库文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../lib)

# 添加共享库
add_library(lennard_jones SHARED lennard_jones.cpp)
add_library(nose_hoover SHARED nose_hoover.cpp)
add_library(nose_hoover_chain SHARED nose_hoover_chain.cpp)
add_library(stress_calculator SHARED stress_calculator.cpp)
add_library(parrinello_rahman_hoover SHARED parrinello_rahman_hoover.cpp)

# 去掉库文件的 lib 前缀
set_target_properties(lennard_jones PROPERTIES PREFIX "")
set_target_properties(nose_hoover PROPERTIES PREFIX "")
set_target_properties(nose_hoover_chain PROPERTIES PREFIX "")
set_target_properties(stress_calculator PROPERTIES PREFIX "")
set_target_properties(parrinello_rahman_hoover PROPERTIES PREFIX "")

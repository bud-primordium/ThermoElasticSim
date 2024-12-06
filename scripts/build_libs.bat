@echo off
REM 设置编译器和编译选项
set COMPILER=gcc
set CFLAGS=-shared -O2 -fPIC -lstdc++

REM 设置源代码目录和目标库文件目录
set SRC_DIR=src\cpp
set LIB_DIR=src\lib

REM 确保库文件目录存在
if not exist %LIB_DIR% mkdir %LIB_DIR%

REM 编译每个共享库
echo Compiling lennard_jones.dll ...
%COMPILER% %SRC_DIR%\lennard_jones.cpp -o %LIB_DIR%\lennard_jones.dll %CFLAGS%

echo Compiling nose_hoover.dll ...
%COMPILER% %SRC_DIR%\nose_hoover.cpp -o %LIB_DIR%\nose_hoover.dll %CFLAGS%

echo Compiling nose_hoover_chain.dll ...
%COMPILER% %SRC_DIR%\nose_hoover_chain.cpp -o %LIB_DIR%\nose_hoover_chain.dll %CFLAGS%

echo Compiling stress_calculator.dll ...
%COMPILER% %SRC_DIR%\stress_calculator.cpp -o %LIB_DIR%\stress_calculator.dll %CFLAGS%

echo Compiling parrinello_rahman_hoover.dll ...
%COMPILER% %SRC_DIR%\parrinello_rahman_hoover.cpp -o %LIB_DIR%\parrinello_rahman_hoover.dll %CFLAGS%

echo All libraries compiled successfully.

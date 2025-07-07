# src/fortran/Makefile

# Fortran 编译器
FC = gfortran

# 编译选项
FFLAGS = -O3 -fPIC -shared

# 目标共享库名称（根据系统不同，可能需要调整后缀，如 .so 或 .dll）
LIBNAME = stress_evaluator.so

# 源文件
SRC = stress_evaluator.f90 structure_optimizer.f90

# 目标文件
OBJ = $(SRC:.f90=.o)

# 默认目标
all: $(LIBNAME)

# 生成共享库
$(LIBNAME): $(OBJ)
	$(FC) $(FFLAGS) -o $@ $^

# 生成目标文件
%.o: %.f90
	$(FC) -c $(FFLAGS) $< -o $@

# 清理
clean:
	rm -f $(OBJ) $(LIBNAME)

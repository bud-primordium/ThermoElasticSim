# Minimal makefile for Sphinx documentation and project build

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
DOXYGEN       ?= doxygen
SOURCEDIR     = source
BUILDDIR      = build
DOXYFILE      = docs/Doxyfile

# 默认目标，运行所有任务
all: update_dates build_libs doxygen html test

# 帮助目标，显示可用的 make 目标
help:
	@echo "可用的目标："
	@echo "  all           - 运行所有任务"
	@echo "  update_dates  - 更新源代码中的日期"
	@echo "  build_libs    - 编译 C++ 库文件"
	@echo "  doxygen       - 生成 Doxygen XML 文档"
	@echo "  html          - 生成 HTML 格式的 Sphinx 文档"
	@echo "  test          - 运行测试"
	@echo "  clean         - 清理生成的文件"

.PHONY: help all update_dates build_libs doxygen html test clean

# 更新日期
update_dates:
	@echo "运行日期更新脚本..."
	python scripts/datetime_updater.py
	@echo "日期更新完成。"

# 编译 C++ 库文件
build_libs:
	@echo "编译 C++ 库文件..."
	$(MAKE) -f Makefile.libs build_libs
	@echo "C++ 库编译完成。"

# 运行 Doxygen 生成 XML 文档
doxygen:
	@echo "运行 Doxygen 生成 XML 文档..."
	@$(DOXYGEN) $(DOXYFILE)
	@echo "Doxygen 文档生成完成。"

# 生成 Sphinx HTML 文档
html: doxygen
	@echo "生成 Sphinx HTML 文档..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo "HTML 文档生成完成。"

# 运行测试
test:
	@echo "运行测试..."
	pytest tests/
	@echo "测试完成。"

# 清理生成的文件
clean:
	@echo "清理生成的文件..."
	rm -rf $(BUILDDIR)/*.html $(BUILDDIR)/doctrees
	rm -f logs/*.log
	rm -rf src/lib/*.dll src/lib/*.so src/lib/*.dylib
	@echo "清理完成。"

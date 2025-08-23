"""
自动生成API文档的Make文件
"""

# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Generate API documentation
apidoc:
	sphinx-apidoc -f -o $(SOURCEDIR)/api ../../src/thermoelasticsim --separate --module-first

# Clean build directory
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(SOURCEDIR)/api/*.rst

# Build HTML documentation
html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build and open HTML documentation
livehtml: html
	open $(BUILDDIR)/html/index.html

# Full rebuild: clean, generate API docs, and build HTML
full: clean apidoc html

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

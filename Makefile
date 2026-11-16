MODULE = aideforge

RUN = uv run
PY := $(RUN) python
INSTALL = uv pip install

BUILD := $(PY) -m build
TWINE := $(PY) -m twine
SIGN := $(RUN) pypi-attestations sign

SPHINXBUILD := $(RUN) sphinx-build
SPHINXAPIDOC := $(RUN) sphinx-apidoc

VERSION := $(shell grep -oP '(?<=^version = ")[^"]+' pyproject.toml)
DISTBASE := dist/$(MODULE)-$(VERSION)
DOCDIR = doc
DOCAPIDIR := $(DOCDIR)/api
DOCBUILDDIR := $(DOCDIR)/_build
PACKAGEDIR = src/aideforge

# High level targets

.PHONY: help pypi doc tools clean

help:
	@echo AideForge $(VERSION) make utility
	@echo
	@echo High level targets:
	@echo "    make help: Help how to use make"
	@echo "    make doc: Create documentation"
	@echo "    make pypi: Build and Upload current version to PyPi"
	@echo "    make clean: Clean up work space"
	@echo "    make tools: Install necessary development tools"
	@echo
	@echo Low level targets:
	@echo "    make build: Build PyPi version"
	@echo "    make sign: Sign packages"
	@echo "    make upload: Upload packages to PyPi"
	@echo "    make apidoc: Create rst source from code"
	@echo "    make html: Create html from rst source"

pypi: upload build sign

doc: apidoc html

tools:
	$(INSTALL) build twine pypi-attestations

clean:
	rm -rf $(DOCBUILDDIR) $(DOCAPIDIR)

# Low level targets

.PHONY: build sign upload html apidoc

build:
	$(BUILD)

sign:
	$(SIGN) $(DISTBASE).whl
	$(SIGN) $(DISTBASE).tar.gz

upload:
	$(TWINE) upload --attestations $(DISTBASE).*

apidoc:
	$(SPHINXAPIDOC) --force --separate --remove-old -o $(DOCAPIDIR) $(PACKAGEDIR)

html:
	$(SPHINXBUILD) --write-all --builder html $(DOCDIR) $(DOCBUILDDIR)/html

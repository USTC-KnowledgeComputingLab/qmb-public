# This Makefile is used for temporary compilation during development.
# For release, the configuration in setup.py should be used for compilation.

CXX := g++
CXXFLAGS := -O3 -Wall -shared -std=c++17 -fPIC
PYTHON_INCLUDE := $(shell python3 -m pybind11 --includes)
PYTHON_SUFFIX := $(shell python3-config --extension-suffix)
SOURCES := $(wildcard _*.cpp)
MODULES := $(basename $(SOURCES))
TARGETS := $(addsuffix $(PYTHON_SUFFIX), $(MODULES))

all: $(TARGETS)

$(TARGETS): %$(PYTHON_SUFFIX): %.cpp
	$(CXX) $(CXXFLAGS) $(PYTHON_INCLUDE) $< -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean

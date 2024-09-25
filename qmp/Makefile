CXX := g++
CXXFLAGS := -O3 -Wall -shared -std=c++17 -fPIC
PYTHON_INCLUDE := $(shell python -m pybind11 --includes)
PYTHON_SUFFIX := $(shell python-config --extension-suffix)
MODULES := _openfermion _ising
TARGETS := $(addsuffix $(PYTHON_SUFFIX), $(MODULES))

all: $(TARGETS)

$(TARGETS): %$(PYTHON_SUFFIX): %.cpp
	$(CXX) $(CXXFLAGS) $(PYTHON_INCLUDE) $< -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean

# Makefile — DBSCAN noise detector (src/dbscan.cpp), Proyecto Apertura 2026B
#
#   make                     build bin/dbscan (macOS: Apple clang + Homebrew libomp; elsewhere: g++ -fopenmp)
#   make GNU=1 CXX=g++-16    macOS build with Homebrew GCC instead of Apple clang
#   make info                bin/build_info.txt: compiler, flags, machine and date of the current build
#   make test                every required validation check   (scripts/validate.py all)
#   make viz-test            headless run of the working notebook (scripts/validate.py notebook)
#   make clean               remove bin/
#
# Windows (MSYS2 UCRT64): make test PYTHON=python

# Plain IEEE double arithmetic (docs/DECISIONS.md, D2 and D10): no fused multiply-add, never -ffast-math or
# -march=native, so every build makes exactly the same neighbor decisions.
# Banderas CONGELADAS para el benchmark oficial (D10): no cambiarlas después de compilar el binario oficial.
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -ffp-contract=off
PYTHON ?= python3

UNAME_S := $(shell uname -s)

# GNU make's built-in CXX is g++; on macOS use Apple clang explicitly (unless CXX is given on the command line).
ifeq ($(UNAME_S),Darwin)
  ifeq ($(origin CXX),default)
    CXX := clang++
  endif
endif

# Apple clang rejects -fopenmp: it needs the OpenMP preprocessor flag plus Homebrew's libomp.
# GCC (Linux, MSYS2 UCRT64, or GNU=1 on macOS) only needs -fopenmp to compile and to link.
USE_LIBOMP :=
ifeq ($(UNAME_S),Darwin)
  ifeq ($(GNU),)
    USE_LIBOMP := 1
  endif
endif

ifeq ($(USE_LIBOMP),1)
  LIBOMP ?= $(shell brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
  OMPFLAGS := -Xpreprocessor -fopenmp -I$(LIBOMP)/include
  LDLIBS := -L$(LIBOMP)/lib -lomp
else
  OMPFLAGS := -fopenmp
  LDLIBS := -fopenmp
endif

.PHONY: all info test viz-test clean

all: bin/dbscan

# The build also records its own configuration, so bin/build_info.txt always describes the binary next to it.
bin/dbscan: src/dbscan.cpp Makefile
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ src/dbscan.cpp $(LDLIBS)
	@{ echo "compiler: $$($(CXX) --version | head -n 1)"; \
	   echo "CXX: $(CXX)"; \
	   echo "CXXFLAGS: $(CXXFLAGS)"; \
	   echo "OMPFLAGS: $(OMPFLAGS)"; \
	   echo "LDLIBS: $(LDLIBS)"; \
	   echo "uname: $$(uname -a)"; \
	   echo "date_utc: $$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > bin/build_info.txt

info: bin/dbscan
	@echo "bin/build_info.txt describes the current bin/dbscan"

test: bin/dbscan
	$(PYTHON) scripts/validate.py all

viz-test: bin/dbscan
	$(PYTHON) scripts/validate.py notebook

clean:
	rm -rf bin

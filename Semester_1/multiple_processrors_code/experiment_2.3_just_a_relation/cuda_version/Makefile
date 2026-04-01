# Makefile for CUDA-based gen_relation
# Target: CUDA Toolkit 11.4, GCC 10 compatible

# Host compiler (can be overridden, e.g., make HOST_COMPILER=gcc-10)
HOST_COMPILER ?= g++

NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -ccbin $(HOST_COMPILER)

# Detect GPU architecture
# Common architectures:
#   sm_50 - Maxwell (GTX 750 Ti)
#   sm_60 - Pascal (GTX 1080, P100)
#   sm_70 - Volta (V100)
#   sm_75 - Turing (RTX 2080)
#   sm_80 - Ampere (A100, RTX 3090)
#   sm_86 - Ampere (RTX 3080)
GPU_ARCH ?= sm_50

NVCC_FLAGS += -arch=$(GPU_ARCH)

# Enable line info for profiling (optional)
# NVCC_FLAGS += -lineinfo

# Debug flags (uncomment for debugging)
# NVCC_FLAGS += -g -G

TARGET = gen_relation_cuda
SRC = gen_relation.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
	rm -rf p_*

# Run with test primes
test: $(TARGET)
	echo "5" > test_primes.txt
	./$(TARGET) test_primes.txt
	@echo "=== Test output ==="
	@cat p_5/matrices_n_2.txt 2>/dev/null || echo "No output generated"

# Run with custom primes file
run: $(TARGET)
	@if [ -z "$(PRIMES)" ]; then \
		echo "Usage: make run PRIMES=<primes_file.txt>"; \
		exit 1; \
	fi
	./$(TARGET) $(PRIMES)

# Build with different GPU architectures
maxwell: GPU_ARCH=sm_50
maxwell: all

pascal: GPU_ARCH=sm_60
pascal: all

volta: GPU_ARCH=sm_70
volta: all

turing: GPU_ARCH=sm_75
turing: all

ampere: GPU_ARCH=sm_80
ampere: all

help:
	@echo "Makefile for CUDA gen_relation"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build the CUDA executable (default)"
	@echo "  clean     - Remove executable and output directories"
	@echo "  test      - Build and run with prime p=5"
	@echo "  run       - Run with custom primes file (make run PRIMES=file.txt)"
	@echo ""
	@echo "GPU Architecture variants:"
	@echo "  maxwell   - Build for Maxwell GPUs (sm_50, e.g. GTX 750 Ti)"
	@echo "  pascal    - Build for Pascal GPUs (sm_60)"
	@echo "  volta     - Build for Volta GPUs (sm_70)"  
	@echo "  turing    - Build for Turing GPUs (sm_75)"
	@echo "  ampere    - Build for Ampere GPUs (sm_80)"
	@echo ""
	@echo "Custom architecture:"
	@echo "  make GPU_ARCH=sm_86 all"

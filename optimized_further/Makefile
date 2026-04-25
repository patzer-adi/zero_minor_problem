# ==============================================================================
# Makefile — APM Brahma (Optimized Further)
#
# Kernel-Range + Fixed Deviation + __int128 + Numerical Sort
#
# Auto-detects GPU compute capability — no hardcoded SM versions.
# One .cu file, one binary.
#
# Optimizations:
#   - __int128 mod_mul on SM>=70, __umul64hi fallback
#   - Numerical file sorting (1,2,...100 not 1,10,100,11...)
#   - Kernel range + deviation args for multi-GPU parallelism
#   - Templated kernel apm_kernel_k<K> (optimal stack per K)
#   - __constant__ pow2_64 (L1-cached, zero param overhead)
#   - Flat int[] index arrays (~5-8x VRAM savings)
#
# Targets:
#   make          — build the binary (apm_brahma)
#   make run      — build + run
#   make run_log  — build + run + save log
#   make clean    — remove build artifacts
#   make help     — show help
#
# Override GPU args:       make ARGS="38 38 1 35 2"
# Override compiler:       make CCBIN=g++-12
# ==============================================================================

NVCC        ?= nvcc
BIN         := apm_brahma
SRC         := apm_brahma.cu
ARGS        ?=

# ==============================================================================
# GPU compute capability auto-detection
# ==============================================================================

GPU_CC_RAW  := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1)
GPU_SM      := $(subst .,,$(GPU_CC_RAW))

ifeq ($(GPU_SM),)
  NVCC_HAS_NATIVE := $(shell $(NVCC) --help 2>/dev/null | grep -c "\-arch=native")
  ifneq ($(NVCC_HAS_NATIVE),0)
    ARCH_FLAGS := -arch=native
    GPU_DETECT_METHOD := nvcc-native
  else
    GPU_SM := 70
    ARCH_FLAGS := -gencode arch=compute_$(GPU_SM),code=sm_$(GPU_SM) \
                  -gencode arch=compute_$(GPU_SM),code=compute_$(GPU_SM)
    GPU_DETECT_METHOD := fallback-70
  endif
else
  ARCH_FLAGS := -gencode arch=compute_$(GPU_SM),code=sm_$(GPU_SM) \
                -gencode arch=compute_$(GPU_SM),code=compute_$(GPU_SM)
  GPU_DETECT_METHOD := nvidia-smi
endif

# ==============================================================================
# Host compiler auto-detection
# ==============================================================================

CCBIN ?= $(shell \
  if command -v g++ >/dev/null 2>&1; then echo g++; \
  elif command -v gcc-12 >/dev/null 2>&1; then echo gcc-12; \
  elif command -v gcc-10 >/dev/null 2>&1; then echo gcc-10; \
  else echo g++; fi)

CUDA_FIX_DIR ?= $(HOME)/cuda_fix
ifeq ($(wildcard $(CUDA_FIX_DIR)/bits/mathcalls.h),)
  CUDA_FIX_INC :=
else
  CUDA_FIX_INC := -I$(CUDA_FIX_DIR)
endif

# ==============================================================================
# Compiler flags
# ==============================================================================

NVCC_FLAGS  := -O3 -std=c++14 \
               -ccbin $(CCBIN) \
               -Wno-deprecated-gpu-targets \
               $(CUDA_FIX_INC) \
               $(ARCH_FLAGS)

# ==============================================================================
# Build rules
# ==============================================================================

.PHONY: all clean run run_log help

all: $(BIN)
	@echo ""
	@echo "============================================================"
	@echo " Build OK: $(BIN)  [OPTIMIZED FURTHER]"
	@echo "============================================================"
	@echo "  GPU_SM        : $(if $(GPU_SM),$(GPU_SM),auto)"
	@echo "  ARCH_FLAGS    : $(ARCH_FLAGS)"
	@echo "  DETECT METHOD : $(GPU_DETECT_METHOD)"
	@echo "  CCBIN         : $(CCBIN)"
	@echo "============================================================"
	@echo ""
	@echo "  Run with:  ./$(BIN)                          (default groups 25-60, all deviations)"
	@echo "  Or:        ./$(BIN) 38 38                    (group 38, all kernels, all deviations)"
	@echo "  Or:        ./$(BIN) 38 38 1 35               (group 38, kernels 1-35, all deviations)"
	@echo "  Or:        ./$(BIN) 38 38 1 35 2             (group 38, kernels 1-35, deviation 2 ONLY)"
	@echo "  Or:        make run ARGS=\"38 38 1 30 2\"     (via make)"
	@echo ""

# NOTE: Compilation takes longer (~45-90s) due to 57 template instantiations
# (apm_kernel_k<4> through apm_kernel_k<60>). Each produces a separate kernel
# binary with optimal stack frame size for that K value.
# The --ptxas-options=-v flag shows register/spill/stack counts per kernel.

$(BIN): $(SRC)
	$(NVCC) $(NVCC_FLAGS) --ptxas-options=-v $(SRC) -o $(BIN)

run: $(BIN)
	./$(BIN) $(ARGS)

run_log: $(BIN)
	./$(BIN) $(ARGS) 2>&1 | tee brahma_log.txt

clean:
	rm -f $(BIN)

clean_results:
	rm -rf Results_further/ brahma_log.txt

clean_all: clean clean_results

help:
	@echo ""
	@echo "============================================================"
	@echo " APM Brahma — Optimized Further"
	@echo " Kernel-Range + Fixed Deviation + __int128 + Numerical Sort"
	@echo "============================================================"
	@echo ""
	@echo "Build:"
	@echo "  make              — compile $(SRC) -> $(BIN)"
	@echo "  make clean        — remove binary"
	@echo "  make clean_all    — remove binary and results"
	@echo ""
	@echo "Sequential (single GPU, auto-iterate deviations):"
	@echo "  make run ARGS=\"38 38\"                — group 38, all kernels, all devs"
	@echo "  make run ARGS=\"38 38 1 35\"            — kernels 1-35, all devs"
	@echo ""
	@echo "Parallel (multi-GPU, FIXED deviation):"
	@echo "  ./submit_parallel.sh 38 100 4 2        — 4 GPUs, deviation 2"
	@echo "  Or manually:"
	@echo "    CUDA_VISIBLE_DEVICES=0 ./apm_brahma 38 38  1 25 2 &"
	@echo "    CUDA_VISIBLE_DEVICES=1 ./apm_brahma 38 38 26 50 2 &"
	@echo "    CUDA_VISIBLE_DEVICES=2 ./apm_brahma 38 38 51 75 2 &"
	@echo "    CUDA_VISIBLE_DEVICES=3 ./apm_brahma 38 38 76 100 2 &"
	@echo "    wait"
	@echo ""
	@echo "Workflow:"
	@echo "  1. python3 sort_kernel_files.py 38      (sort files, one-time)"
	@echo "  2. make"
	@echo "  3. ./submit_parallel.sh 38 100 4 2      (deviation 2)"
	@echo "  4. python3 check_results.py 38 -d 2     (verify)"
	@echo "  5. ./submit_parallel.sh 38 100 4 3      (deviation 3 if needed)"
	@echo "============================================================"
	@echo ""

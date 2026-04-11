# ==============================================================================
# Makefile — APM Brahma: Single-file version
#
# Auto-detects GPU compute capability — no hardcoded SM versions.
# One .cu file, one binary.
#
# Targets:
#   make          — build the binary (apm_brahma)
#   make run      — build + run (default groups 25-50)
#   make run_log  — build + run + save log
#   make clean    — remove build artifacts
#   make help     — show help
#
# Override GPU args:       make ARGS="32 35"
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
    GPU_SM := 50
    ARCH_FLAGS := -gencode arch=compute_$(GPU_SM),code=sm_$(GPU_SM) \
                  -gencode arch=compute_$(GPU_SM),code=compute_$(GPU_SM)
    GPU_DETECT_METHOD := fallback-50
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
	@echo " Build OK: $(BIN)"
	@echo "============================================================"
	@echo "  GPU_SM        : $(if $(GPU_SM),$(GPU_SM),auto)"
	@echo "  ARCH_FLAGS    : $(ARCH_FLAGS)"
	@echo "  DETECT METHOD : $(GPU_DETECT_METHOD)"
	@echo "  CCBIN         : $(CCBIN)"
	@echo "============================================================"
	@echo ""
	@echo "  Run with:  ./$(BIN)             (default groups 25-50)"
	@echo "  Or:        ./$(BIN) 32 35       (groups 32-35 only)"
	@echo "  Or:        make run              (live terminal)"
	@echo "  Or:        make run_log          (live + saved to log)"
	@echo ""

$(BIN): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(BIN)

run: $(BIN)
	./$(BIN) $(ARGS)

run_log: $(BIN)
	./$(BIN) $(ARGS) 2>&1 | tee brahma_log.txt

clean:
	rm -f $(BIN)

clean_results:
	rm -rf Results_brahma/ brahma_log.txt

clean_all: clean clean_results

help:
	@echo ""
	@echo "============================================================"
	@echo " APM Brahma — Makefile Help (Single-File Version)"
	@echo "============================================================"
	@echo ""
	@echo "Build:"
	@echo "  make              — compile $(SRC) -> $(BIN)"
	@echo "  make clean        — remove binary"
	@echo "  make clean_all    — remove binary and results"
	@echo ""
	@echo "Run:"
	@echo "  make run           — compile + run (default groups 25-50)"
	@echo "  make run_log       — compile + run + save log"
	@echo "  make run ARGS=\"32 35\"  — run only groups 32..35"
	@echo ""
	@echo "GPU auto-detection (same as server_brahma):"
	@echo "  1. nvidia-smi --query-gpu=compute_cap"
	@echo "  2. nvcc -arch=native (CUDA >= 11.1)"
	@echo "  3. Fallback to compute_50"
	@echo "============================================================"
	@echo ""

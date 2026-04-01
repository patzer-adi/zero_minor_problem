# ==============================================================================
# Makefile — APM Brahma: Modular First-Hit Search
#
# Auto-detects GPU compute capability — no hardcoded SM versions.
# Separate compilation of .cu and .cpp files.
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
ARGS        ?=

# ==============================================================================
# GPU compute capability auto-detection
#
# Strategy (in priority order):
#   1. Use nvidia-smi to query the GPU's compute capability directly
#   2. Fall back to -arch=native (CUDA >= 11.1) which lets nvcc detect it
#   3. Default to compute_50 (Maxwell) as a safe baseline
# ==============================================================================

# Try to detect compute_cap from nvidia-smi (format: "7.5" or "8.6")
GPU_CC_RAW  := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1)

# Strip the dot: "7.5" -> "75"
GPU_SM      := $(subst .,,$(GPU_CC_RAW))

# Validate: if GPU_SM is empty, try nvcc device-query
ifeq ($(GPU_SM),)
  # Check if nvcc supports --list-gpu-arch (CUDA 11.5+)
  NVCC_HAS_NATIVE := $(shell $(NVCC) --help 2>/dev/null | grep -c "\-arch=native")
  ifneq ($(NVCC_HAS_NATIVE),0)
    ARCH_FLAGS := -arch=native
    GPU_DETECT_METHOD := nvcc-native
  else
    # Safe fallback: compute_50 (Maxwell, widely supported)
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

# Optional glibc/CUDA fix include dir
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

CXX_FLAGS   := -O3 -std=c++14 -Wall -Wextra -Wno-unused-parameter

# ==============================================================================
# Source files
# ==============================================================================

CU_SRCS     := apm_search.cu main.cu
CPP_SRCS    := file_utils.cpp matrix_parser.cpp prime_loader.cpp result_writer.cpp

CU_OBJS     := $(CU_SRCS:.cu=.o)
CPP_OBJS    := $(CPP_SRCS:.cpp=.o)
ALL_OBJS    := $(CU_OBJS) $(CPP_OBJS)

# ==============================================================================
# Build rules
# ==============================================================================

.PHONY: all clean run run_log help config_log

all: config_log $(BIN)

config_log:
	@echo ""
	@echo "============================================================"
	@echo " APM Brahma — Build Configuration"
	@echo "============================================================"
	@echo "  NVCC          : $(NVCC)"
	@echo "  CCBIN         : $(CCBIN)"
	@echo "  GPU_SM        : $(if $(GPU_SM),$(GPU_SM),auto)"
	@echo "  ARCH_FLAGS    : $(ARCH_FLAGS)"
	@echo "  DETECT METHOD : $(GPU_DETECT_METHOD)"
	@echo "  CUDA_FIX_INC  : $(if $(CUDA_FIX_INC),$(CUDA_FIX_INC),<none>)"
	@echo "============================================================"
	@echo ""

# Link all object files into final binary
$(BIN): $(ALL_OBJS)
	@echo "Linking $(BIN) ..."
	$(NVCC) $(NVCC_FLAGS) $(ALL_OBJS) -o $(BIN) -Xlinker -lstdc++
	@echo ""
	@echo "  Build OK"
	@echo "  Run with:  ./$(BIN)             (default groups 25-50)"
	@echo "  Or:        ./$(BIN) 32 35       (groups 32-35 only)"
	@echo "  Or:        make run              (live terminal)"
	@echo "  Or:        make run_log          (live terminal + saved to log)"

# Compile .cu files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile .cpp files using nvcc (so it handles CUDA includes properly)
%.o: %.cpp
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(CXX_FLAGS)" -c $< -o $@

# ==============================================================================
# Run targets
# ==============================================================================

run: $(BIN)
	./$(BIN) $(ARGS)

run_log: $(BIN)
	./$(BIN) $(ARGS) 2>&1 | tee brahma_log.txt

# ==============================================================================
# Clean
# ==============================================================================

clean:
	rm -f $(BIN) $(ALL_OBJS)

clean_results:
	rm -rf Results_brahma/ brahma_log.txt

clean_all: clean clean_results

# ==============================================================================
# Help
# ==============================================================================

help:
	@echo ""
	@echo "============================================================"
	@echo " APM Brahma — Makefile Help"
	@echo "============================================================"
	@echo ""
	@echo "Build targets:"
	@echo "  make              — compile all sources -> $(BIN)"
	@echo "  make clean        — remove binary and object files"
	@echo "  make clean_all    — remove binary, objects, and results"
	@echo ""
	@echo "Run targets:"
	@echo "  make run           — compile + run (default groups 25-50)"
	@echo "  make run_log       — compile + run + save to brahma_log.txt"
	@echo "  make run ARGS=\"32 35\"  — run only groups 32..35"
	@echo ""
	@echo "Configuration overrides:"
	@echo "  make CCBIN=g++-12      — use specific host compiler"
	@echo ""
	@echo "GPU auto-detection:"
	@echo "  The Makefile automatically detects GPU compute capability."
	@echo "  Detection methods (in priority order):"
	@echo "    1. nvidia-smi --query-gpu=compute_cap"
	@echo "    2. nvcc -arch=native (CUDA >= 11.1)"
	@echo "    3. Fallback to compute_50 (Maxwell)"
	@echo ""
	@echo "Output structure:"
	@echo "  Results_brahma/"
	@echo "    <group>/"
	@echo "      result.txt                  — best deviation summary"
	@echo "      deviation_<d>/"
	@echo "        <matrix>_result.txt       — per-matrix result"
	@echo "        SUMMARY_detailed.txt      — detailed summary"
	@echo "        SUMMARY_brief.txt         — brief summary"
	@echo "============================================================"
	@echo ""

# =============================================================================
# Makefile — APM Exhaustive Search
# GPU  : GTX 750 Ti  (SM 5.0 / Maxwell)
# CUDA : 11.4
# GCC  : 10  (GCC 11 causes __malloc__ errors with CUDA 11.4 on Ubuntu 22.04)
#
# Two binaries:
#   apm3d  — original p_1048573 only   (apm_research_3.cu)
#   apm4d  — extended multi-prime      (apm_research_4.cu)
#
# Execution order for apm4d (three-phase):
#   Phase 1 — p_1048573 antioffset  (12×12, fast)
#   Phase 2 — 25_29/{25..29}        (75×75, per-folder prime)
#   Phase 3 — p_1048573 offset      (larger, slow)
# =============================================================================

NVCC    ?= nvcc

# =============================================================================
# Host/GPU autodetect (GTX 750 Ti vs RTX 2080/2080 Ti)
# - You can override any of these with `make VAR=value ...`
# =============================================================================

# Best-effort GPU name detection (works on most CUDA boxes with nvidia-smi)
GPU_NAME ?= $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)

# Compute capability selection based on GPU name
ifeq ($(findstring 750,$(GPU_NAME)),750)
  GPU_FAMILY := 750ti
  SM         := 50
  CCBIN      ?= gcc-10
else ifneq (,$(filter %2080%,$(GPU_NAME)))
  GPU_FAMILY := 2080
  SM         := 75
  CCBIN      ?= g++-12
else
  GPU_FAMILY := unknown
  SM         := 75
  CCBIN      ?= g++
endif

# Optional glibc-2.42 / CUDA-13.1 fix include dir (used on the SSH/CUDA 13.1 box)
CUDA_FIX_DIR ?= $(HOME)/cuda_fix
ifeq ($(wildcard $(CUDA_FIX_DIR)/bits/mathcalls.h),)
  CUDA_FIX_INC :=
else
  CUDA_FIX_INC := -I$(CUDA_FIX_DIR)
endif

# ── Original binary (apm_research_3.cu) ──────────────────────────────────────
BIN     := apm3d
SRC     := apm_research_3.cu
PRIME   := 1048573

# ── Extended binary (apm_research_4.cu) ───────────────────────────────────────
BIN4    := apm4d
SRC4    := apm_research_4.cu

# ── Specialized binary (apm_research_5.cu) ───────────────────────────────────
BIN5    := apm5d
SRC5    := apm_research_5.cu

# ── New folder binary (apm_research_5_new.cu) ────────────────────────────────
BIN5NEW := apm5newd
SRC5NEW := apm_research_5_new.cu

# ── V6 kernel_output binary (apm_research_6.cu) ──────────────────────────────
BIN6    := apm6
SRC6    := apm_research_6.cu

# ── First-hit binary (apm_research_hits_one.cu) ──────────────────────────────
BINH1   := apm_hits_one
SRCH1   := apm_research_hits_one.cu

# ── Check-till-100 binary (apm_research_hits_one_check_till_100.cu) ─────────
BINTILL := apm_hits_one_check_till_100
SRCTILL := apm_research_hits_one_check_till_100.cu

# Optional args passed to apm6 / apm_hits_one.
# Example: make run6 ARGS="32 35"
ARGS    ?=

ARCH    := -gencode arch=compute_$(SM),code=sm_$(SM) \
           -gencode arch=compute_$(SM),code=compute_$(SM)

FLAGS   := -O3 -std=c++14              \
           -ccbin $(CCBIN)             \
           -Wno-deprecated-gpu-targets \
           $(CUDA_FIX_INC)             \
           $(ARCH)

# ─── Build ───────────────────────────────────────────────────────────────────

.PHONY: all clean run run_log run_explicit clean_results clean_all help
.PHONY: apm4 run4 run4_log apm5 run5 run5_log clean5
.PHONY: apm5new run5new run5new_log clean5new
.PHONY: run6 run6_log clean6 clean6_results
.PHONY: run_hits_one run_hits_one_log clean_hits_one clean_hits_one_results
.PHONY: apm_till_100 run_till_100 run_till_100_log clean_till_100 clean_till_100_results
.PHONY: config_log

all: $(BIN)

apm4: $(BIN4)

apm5: $(BIN5)

apm5new: $(BIN5NEW)

both: $(BIN) $(BIN5)

config_log:
	@echo ""
	@echo "============================================================"
	@echo " APM Makefile build config"
	@echo "============================================================"
	@echo "  GPU_NAME      : $(if $(GPU_NAME),$(GPU_NAME),<not detected>)"
	@echo "  GPU_FAMILY    : $(GPU_FAMILY)"
	@echo "  SM            : $(SM)"
	@echo "  NVCC          : $(NVCC)"
	@echo "  CCBIN         : $(CCBIN)"
	@echo "  CUDA_FIX_INC  : $(if $(CUDA_FIX_INC),$(CUDA_FIX_INC),<none>)"
	@echo "  ARCH          : $(ARCH)"
	@echo "============================================================"
	@echo ""

$(BIN): $(SRC)
	@$(MAKE) --no-print-directory config_log
	@echo "Building $(BIN) ..."
	$(NVCC) $(FLAGS) $< -o $@ -Xlinker -lstdc++
	@echo ""
	@echo "  Build OK"
	@echo "  Run with:  ./$(BIN) $(PRIME)"
	@echo "  Or:        make run        (live terminal)"
	@echo "  Or:        make run_log    (live terminal + saved to run_log.txt)"

$(BIN4): $(SRC4)
	@$(MAKE) --no-print-directory config_log
	@echo "Building $(BIN4) ..."
	$(NVCC) $(FLAGS) $< -o $@ -Xlinker -lstdc++
	@echo ""
	@echo "  Build OK  (extended multi-prime version)"
	@echo "  Run with:  ./$(BIN4)       (no arguments needed)"
	@echo "  Or:        make run4       (live terminal)"
	@echo "  Or:        make run4_log   (live terminal + saved to run4_log.txt)"

$(BIN5): $(SRC5)
	@$(MAKE) --no-print-directory config_log
	@echo "Building $(BIN5) ..."
	$(NVCC) $(FLAGS) $< -o $@ -Xlinker -lstdc++
	@echo ""
	@echo "  Build OK  (specialized version)"
	@echo "  Run with:  ./$(BIN5)       (no arguments needed)"
	@echo "  Or:        make run5       (live terminal)"
	@echo "  Or:        make run5_log   (live terminal + saved to run5_log.txt)"

$(BIN5NEW): $(SRC5NEW)
	@$(MAKE) --no-print-directory config_log
	@echo "Building $(BIN5NEW) ..."
	$(NVCC) $(FLAGS) $< -o $@ -Xlinker -lstdc++
	@echo ""
	@echo "  Build OK  (new folder version — primes 25-35)"
	@echo "  Run with:  ./$(BIN5NEW)       (no arguments needed)"
	@echo "  Or:        make run5new       (live terminal)"
	@echo "  Or:        make run5new_log   (live terminal + saved to run5new_log.txt)"

$(BIN6): $(SRC6)
	@$(MAKE) --no-print-directory config_log
	@echo "Building $(BIN6) ..."
	$(NVCC) $(FLAGS) $< -o $@ -Xlinker -lstdc++
	@echo ""
	@echo "  Build OK  (V6 — kernel_output, primes 25-35, all long long)"
	@echo "  Run with:  ./$(BIN6)          (no arguments needed)"
	@echo "  Or:        make run6          (live terminal)"
	@echo "  Or:        make run6_log      (live terminal + saved to run6_log.txt)"

$(BINH1): $(SRCH1)
	@$(MAKE) --no-print-directory config_log
	@echo "Building $(BINH1) ..."
	$(NVCC) $(FLAGS) $< -o $@ -Xlinker -lstdc++
	@echo ""
	@echo "  Build OK  (first-hit — stops after 1st zero minor per matrix)"
	@echo "  Run with:  ./$(BINH1)              (no arguments needed)"
	@echo "  Or:        make run_hits_one        (live terminal)"
	@echo "  Or:        make run_hits_one_log    (live terminal + saved to hits_one_log.txt)"

$(BINTILL): $(SRCTILL)
	@$(MAKE) --no-print-directory config_log
	@echo "Building $(BINTILL) ..."
	$(NVCC) $(FLAGS) $< -o $@ -Xlinker -lstdc++
	@echo ""
	@echo "  Build OK  (check-till-100 — early stops when 100 hits reached)"
	@echo "  Run with:  ./$(BINTILL)              (no arguments needed)"
	@echo "  Or:        make run_till_100          (live terminal)"
	@echo "  Or:        make run_till_100_log      (live terminal + saved to till_100_log.txt)"

# ─── Run targets — original apm3d ───────────────────────────────────────────

run: $(BIN)
	./$(BIN) $(PRIME)

run_log: $(BIN)
	./$(BIN) $(PRIME) 2>&1 | tee run_log.txt

run_explicit: $(BIN)
	./$(BIN) $(PRIME) p_$(PRIME)_antioffset=0.2/ p_$(PRIME)_offset=1/

run_explicit_log: $(BIN)
	./$(BIN) $(PRIME) p_$(PRIME)_antioffset=0.2/ p_$(PRIME)_offset=1/ 2>&1 | tee run_log.txt

# ─── Run targets — specialized apm5d ──────────────────────────────────────────

run4: $(BIN4)
	./$(BIN4)

run4_log: $(BIN4)
	./$(BIN4) 2>&1 | tee run4_log.txt

run5: $(BIN5)
	./$(BIN5)

run5_log: $(BIN5)
	./$(BIN5) 2>&1 | tee run5_log.txt

# ─── Run targets — new folder apm5new ─────────────────────────────────────────

run5new: $(BIN5NEW)
	./$(BIN5NEW)

run5new_log: $(BIN5NEW)
	./$(BIN5NEW) 2>&1 | tee run5new_log.txt

# ─── Run targets — V6 kernel_output ───────────────────────────────────────────

run6: $(BIN6)
	./$(BIN6) $(ARGS)

run6_log: $(BIN6)
	./$(BIN6) $(ARGS) 2>&1 | tee run6_log.txt

# ─── Run targets — first-hit apm_hits_one ─────────────────────────────────────

run_hits_one: $(BINH1)
	./$(BINH1) $(ARGS)

run_hits_one_log: $(BINH1)
	./$(BINH1) $(ARGS) 2>&1 | tee hits_one_log.txt

# ─── Run targets — check-till-100 ─────────────────────────────────────────────

apm_till_100: $(BINTILL)

run_till_100: $(BINTILL)
	./$(BINTILL) $(ARGS)

run_till_100_log: $(BINTILL)
	./$(BINTILL) $(ARGS) 2>&1 | tee till_100_log.txt

# ─── Clean ───────────────────────────────────────────────────────────────────

clean:
	rm -f $(BIN) apm3

clean4:
	rm -f $(BIN4) apm4

clean5:
	rm -f $(BIN5) apm5

clean5new:
	rm -f $(BIN5NEW) apm5new

clean6:
	rm -f $(BIN6) apm6

clean_hits_one:
	rm -f $(BINH1) hits_one apm_hits_one

clean_till_100:
	rm -f $(BINTILL) apm_hits_one_check_till_100

clean6_results:
	rm -rf Results_6/ run6_log.txt

clean_hits_one_results:
	rm -rf Results_hits_one/ hits_one_log.txt

clean_till_100_results:
	rm -rf Results_hits_one_check_till_100/ till_100_log.txt

clean_results:
	rm -rf Results/ run_log.txt

clean4_results:
	rm -rf Results/ run4_log.txt

clean_all: clean clean4 clean5 clean5new clean6 clean_hits_one clean_till_100 clean_results clean6_results clean_hits_one_results clean_till_100_results
	rm -f run_log.txt run4_log.txt run5_log.txt run5new_log.txt run6_log.txt hits_one_log.txt till_100_log.txt
	rm -f apm3 apm4 apm5 apm6 hits_one apm5new apm_hits_one_check_till_100

# ─── Help ────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "============================================================"
	@echo " APM Exhaustive Search — Makefile Help"
	@echo "============================================================"
	@echo ""
	@echo "Build targets:"
	@echo "  make              — compile apm_research_3.cu -> $(BIN)"
	@echo "  make apm4         — compile apm_research_4.cu -> $(BIN4)"
	@echo "  make apm5new      — compile apm_research_5_new.cu -> $(BIN5NEW)"
	@echo "  make apm6         — compile apm_research_6.cu -> $(BIN6)"
	@echo "  make apm_hits_one — compile apm_research_hits_one.cu -> $(BINH1)"
	@echo "  make both         — compile both binaries"
	@echo ""
	@echo "Run targets (original — p_1048573 only):"
	@echo "  make run              — compile + run $(BIN)"
	@echo "  make run_log          — compile + run + save log"
	@echo "  make run_explicit     — run with hard-coded folder names"
	@echo "  make run_explicit_log — run_explicit + save log"
	@echo ""
	@echo "Run targets (extended — multi-prime, all folders):"
	@echo "  make run4             — compile + run $(BIN4) (no args needed)"
	@echo "  make run4_log         — compile + run + save to run4_log.txt"
	@echo ""
	@echo "Run targets (new folder — primes 25-35, long long kernel):"
	@echo "  make run5new          — compile + run $(BIN5NEW) (no args needed)"
	@echo "  make run5new_log      — compile + run + save to run5new_log.txt"
	@echo ""
	@echo "Run targets (V6 — kernel_output, group range via ARGS):"
	@echo "  make run6                 — compile + run $(BIN6) (default groups 25-35)"
	@echo "  make run6_log         — compile + run + save to run6_log.txt"
	@echo "  make run6 ARGS=\"GMIN GMAX\"   — run only groups GMIN..GMAX (any range)"
	@echo "    examples: ARGS=\"32 35\"  or  ARGS=\"25 27\""
	@echo ""
	@echo "Run targets (first-hit — stops after 1st zero minor per matrix):"
	@echo "  make run_hits_one         — compile + run $(BINH1) (default groups 25-35)"
	@echo "  make run_hits_one_log — compile + run + save to hits_one_log.txt"
	@echo "  make run_hits_one ARGS=\"GMIN GMAX\"   — run only groups GMIN..GMAX (any range)"
	@echo "    examples: ARGS=\"32 35\"  or  ARGS=\"25 27\""
	@echo ""
	@echo "Clean targets:"
	@echo "  make clean                  — remove $(BIN) only"
	@echo "  make clean4                 — remove $(BIN4) only"
	@echo "  make clean5new              — remove $(BIN5NEW) only"
	@echo "  make clean6                 — remove $(BIN6) only"
	@echo "  make clean_hits_one         — remove $(BINH1) only"
	@echo "  make clean6_results         — remove Results_6/ and run6_log.txt"
	@echo "  make clean_hits_one_results — remove Results_hits_one/ and hits_one_log.txt"
	@echo "  make clean_results          — remove Results/ and logs"
	@echo "  make clean_all              — remove everything"
	@echo ""
	@echo "$(BIN4) execution order (three-phase, no arguments):"
	@echo "  Phase 1: p_1048573 antioffset  (12×12 matrices, fast)"
	@echo "  Phase 2: 25_29/{25..29}        (75×75 matrices, per-folder prime)"
	@echo "           25→33554393  26→44923183  27→134217689"
	@echo "           28→268435399 29→536870909"
	@echo "  Phase 3: p_1048573 offset      (larger matrices, slow)"
	@echo ""
	@echo "Output structure after run4:"
	@echo "  Results/"
	@echo "    deviation_2/"
	@echo "      antioffset/    <- p_1048573 antioffset results"
	@echo "      25_29_25/      <- subfolder 25 (prime=33554393)"
	@echo "      25_29_26/      <- subfolder 26 (prime=44923183)"
	@echo "      25_29_27/      <- subfolder 27 (prime=134217689)"
	@echo "      25_29_28/      <- subfolder 28 (prime=268435399)"
	@echo "      25_29_29/      <- subfolder 29 (prime=536870909)"
	@echo "      offset/        <- p_1048573 offset results"
	@echo "    deviation_3/ ... (same structure)"
	@echo "    deviation_4/ ..."
	@echo "    deviation_5/ ..."
	@echo ""
	@echo "Tip: use 'make run4_log' so you can watch live output AND"
	@echo "     have a full record in run4_log.txt when it finishes."
	@echo "============================================================"
	@echo ""

// =============================================================================
// apm_research_hits_one.cu
//
// APM — Almost Principal Minors, First-Hit Search (based on V6)
//
// Searches for submatrices (minors) with zero determinant mod prime in ECDLP
// kernel matrices.  Covers 11 prime groups (25-bit through 35-bit primes).
//
// Difference from V6:
//   - EARLY EXIT: as soon as the FIRST zero minor is found for a matrix,
//     search stops immediately and moves to the next matrix.
//   - Does NOT check the minor matrix (no further chunks/s-iterations).
//   - Output files report at most 1 zero minor per matrix.
//
// Key properties (inherited from V6):
//   - NO MinorJob struct/array — threads derive (r,c) from tid
//   - NO ChunkResult / atomicCAS — uses d_zero_flags[] to capture first zero
//   - Runtime COL_CHUNK — computed per (n,dev) to stay within 1500 MB
//   - ALL GPU arithmetic uses long long — correct for 35-bit primes
//   - Runtime max_dev = n - (PM_SIZE + 1) — not hardcoded
//   - MAX_IDX_STATIC = 32 — supports matrices up to 34×34
//   - Primes read from files at startup (with hardcoded fallback)
//   - Two summary files: SUMMARY_detailed.txt and SUMMARY_brief.txt
//
// Execution order (group-outer, deviation-inner):
//   For group g = 25, 26, ..., 35:
//     prime = read_prime(g)
//     files = kernel_output/<g>/kernel_<g>_*.txt  (skip _RN.txt)
//     For deviation d = 2, 3, ..., n-3:
//       process all matrices, write results to Results_6/<g>/deviation_<d>/
//
// Hardware target:
//   GPU:  GTX 750 Ti (SM 5.0, Maxwell, 2 GB VRAM)
//   CUDA: 11.4
//   Host: gcc-10
//
// Compile:
//   nvcc -O3 -std=c++14 -ccbin gcc-10 \
//        -Wno-deprecated-gpu-targets \
//        -gencode arch=compute_50,code=sm_50 \
//        -gencode arch=compute_50,code=compute_50 \
//        apm_research_hits_one.cu -o apm_hits_one -Xlinker -lstdc++
//
// =============================================================================

#define _GNU_SOURCE
#include <cmath>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

// =============================================================================
// Compile-time constants
// =============================================================================

#define PM_SIZE 2          // principal minor block size — FIXED at 2
#define MIN_DEV 2          // minimum deviation to test
#define MAX_IDX_STATIC 32  // max indices in any IndexSet (covers up to 34×34)
#define VRAM_LIMIT_MB 1500 // safety limit for GPU allocations (MB)

// =============================================================================
// Prime table — filled from files at startup, hardcoded fallback
// =============================================================================

struct FolderPrime {
  int folder_id;
  long long prime;
  char label[64];
};

// Hardcoded defaults (authoritative values come from files)
static FolderPrime FOLDER_PRIMES[11] = {
    {25, 33554393LL, "25 (p=33554393)"},
    {26, 44923183LL, "26 (p=44923183)"},
    {27, 134217689LL, "27 (p=134217689)"},
    {28, 268435399LL, "28 (p=268435399)"},
    {29, 536870909LL, "29 (p=536870909)"},
    {30, 1073741789LL, "30 (p=1073741789)"},
    {31, 2147483647LL, "31 (p=2147483647)"},
    {32, 4294967291LL, "32 (p=4294967291)"},
    {33, 8589934583LL, "33 (p=8589934583)"},
    {34, 17179869143LL, "34 (p=17179869143)"},
    {35, 34359738337LL, "35 (p=34359738337)"},
};
static const int NUM_FOLDER_PRIMES = 11;

// =============================================================================
// Data structures
// =============================================================================

// One row-set or col-set: sorted indices into the matrix
struct IndexSet {
  int k;                   // number of indices (= PM_SIZE + dev)
  int idx[MAX_IDX_STATIC]; // sorted matrix indices
};

// Parsed matrix
struct MatrixData {
  std::string filename;
  int n;
  std::vector<long long> data; // flat n×n row-major
};

// One recorded zero minor
struct ZeroMinor {
  int k;   // minor size
  int dev; // deviation level
  int s;   // principal block anchor
  int row_idx[MAX_IDX_STATIC];
  int col_idx[MAX_IDX_STATIC];
  double time_ms; // time since matrix start (ms)
};

// =============================================================================
// Device: modular arithmetic — ALL long long, no exceptions
// =============================================================================

__device__ inline long long mod_sub(long long a, long long b, long long p) {
  long long r = a - b;
  return (r < 0) ? r + p : r;
}

// Split multiplication: a < p < 2^35, so a_hi < 2^19, a_lo < 2^16
// a_hi * b < 2^54, fits in signed long long (max ~2^63)
__device__ inline long long mod_mul(long long a, long long b, long long p) {
  a %= p;
  if (a < 0)
    a += p;
  b %= p;
  if (b < 0)
    b += p;
  long long a_hi = a >> 16;
  long long a_lo = a & 0xFFFFLL;
  long long t1 = (a_hi * b) % p; // < p < 2^35
  t1 = (t1 << 16) % p;           // t1 < p, shift OK (< 2^51)
  long long t2 = (a_lo * b) % p;
  return (t1 + t2) % p;
}

// Extended Euclidean — all long long
__device__ long long mod_inv(long long a, long long p) {
  long long t = 0, nt = 1, r = p, nr = a % p;
  if (nr < 0)
    nr += p;
  while (nr) {
    long long q = r / nr, tmp;
    tmp = t;
    t = nt;
    nt = tmp - q * nt;
    tmp = r;
    r = nr;
    nr = tmp - q * nr;
  }
  return (t < 0) ? t + p : t;
}

// =============================================================================
// Device: determinant mod p via Gaussian elimination — all long long
// sub is flat k×k row-major, values already in [0, p-1]
// =============================================================================

__device__ long long det_mod(const long long *sub, int k, long long p) {
  long long a[MAX_IDX_STATIC][MAX_IDX_STATIC];
  for (int i = 0; i < k; i++)
    for (int j = 0; j < k; j++)
      a[i][j] = sub[i * k + j];

  long long det = 1LL;
  for (int col = 0; col < k; col++) {
    // Find pivot
    int piv = -1;
    for (int row = col; row < k; row++)
      if (a[row][col]) {
        piv = row;
        break;
      }
    if (piv < 0)
      return 0LL;

    // Swap rows
    if (piv != col) {
      for (int j = col; j < k; j++) {
        long long tmp = a[col][j];
        a[col][j] = a[piv][j];
        a[piv][j] = tmp;
      }
      det = (p - det) % p; // row swap negates determinant
    }

    det = mod_mul(det, a[col][col], p);
    long long inv = mod_inv(a[col][col], p);

    for (int row = col + 1; row < k; row++) {
      if (!a[row][col])
        continue;
      long long f = mod_mul(a[row][col], inv, p);
      for (int j = col + 1; j < k; j++)
        a[row][j] = mod_sub(a[row][j], mod_mul(f, a[col][j], p), p);
      a[row][col] = 0LL;
    }
  }
  return det;
}

// =============================================================================
// CUDA kernel
//
// Each thread handles one (row_set, col_set) pair.
// Thread computes (r, c) from tid — NO MinorJob array.
// Writes d_zero_flags[tid] = 1 if det == 0, else 0.
// Host scans flags and stops at the FIRST zero found.
// =============================================================================

__global__ void
apm_kernel(const long long *d_matrix, // flat n×n raw integers
           int n,
           const IndexSet *d_row_sets, // all row index sets
           const IndexSet *d_col_sets, // current chunk of col index sets
           int num_row_sets,
           int num_col_sets, // actual count in this chunk
           int k,            // minor size = PM_SIZE + dev
           long long prime,
           int *d_zero_flags) // output: 1 if det==0, 0 otherwise
{
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long)num_row_sets * num_col_sets;
  if (tid >= total)
    return;

  int r = (int)(tid / num_col_sets);
  int c = (int)(tid % num_col_sets);

  // Extract and reduce k×k submatrix
  long long sub[MAX_IDX_STATIC * MAX_IDX_STATIC];
  for (int i = 0; i < k; i++)
    for (int j = 0; j < k; j++) {
      long long v = d_matrix[d_row_sets[r].idx[i] * n + d_col_sets[c].idx[j]];
      sub[i * k + j] = ((v % prime) + prime) % prime;
    }

  d_zero_flags[tid] = (det_mod(sub, k, prime) == 0LL) ? 1 : 0;
}

// =============================================================================
// CUDA error check macro
// =============================================================================

#define CK(call)                                                               \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d — %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(_e));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// =============================================================================
// Host utilities
// =============================================================================

static double now_ms() {
  using namespace std::chrono;
  return (double)duration_cast<microseconds>(
             high_resolution_clock::now().time_since_epoch())
             .count() /
         1000.0;
}

// nCr — compute binomial coefficient (returns 0 on overflow or invalid)
static long long nCr(int n, int r) {
  if (r < 0 || r > n)
    return 0;
  if (r == 0 || r == n)
    return 1;
  if (r > n - r)
    r = n - r; // symmetry
  long long result = 1;
  for (int i = 0; i < r; i++) {
    result = result * (n - i) / (i + 1);
  }
  return result;
}

// Create a single directory (ignore if exists)
static void mkdir_safe(const std::string &p) { mkdir(p.c_str(), 0755); }

// Create Results_6/<group>/deviation_<d>/
static std::string make_out_dir(int group, int dev) {
  mkdir_safe("Results_hits_one");
  std::string d1 = "Results_hits_one/" + std::to_string(group);
  mkdir_safe(d1);
  std::string d2 = d1 + "/deviation_" + std::to_string(dev);
  mkdir_safe(d2);
  return d2;
}

// Check if a filename ends with a given suffix
static bool ends_with(const std::string &str, const std::string &suffix) {
  if (suffix.size() > str.size())
    return false;
  return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Check if a filename starts with a given prefix
static bool starts_with(const std::string &str, const std::string &prefix) {
  if (prefix.size() > str.size())
    return false;
  return str.compare(0, prefix.size(), prefix) == 0;
}

// Collect all .txt files in a directory that match a given prefix, sorted.
// SKIPS files ending in _RN.txt (row-number metadata).
static std::vector<std::string> collect_files(const std::string &dir_path,
                                              const std::string &prefix) {
  std::vector<std::string> files;
  DIR *d = opendir(dir_path.c_str());
  if (!d)
    return files;
  struct dirent *e;
  while ((e = readdir(d)) != NULL) {
    std::string nm = e->d_name;
    if (!starts_with(nm, prefix))
      continue;
    if (nm.size() <= 4 || nm.substr(nm.size() - 4) != ".txt")
      continue;
    if (ends_with(nm, "_RN.txt"))
      continue;
    files.push_back(dir_path + "/" + nm);
  }
  closedir(d);
  std::sort(files.begin(), files.end());
  return files;
}

// =============================================================================
// Prime file reader
//
// Format: lines of tab-separated integers, then a line with "#".
// The prime is the first line that has exactly ONE integer token.
// =============================================================================

static long long read_prime_from_file(const std::string &path) {
  std::ifstream ifs(path.c_str());
  if (!ifs.is_open())
    return -1;

  std::string line;
  while (std::getline(ifs, line)) {
    // Trim
    size_t fs = line.find_first_not_of(" \t\r\n");
    if (fs == std::string::npos)
      continue;
    line = line.substr(fs);

    // Stop at "#"
    if (line[0] == '#')
      break;

    // Tokenize by whitespace/tabs
    std::istringstream ss(line);
    std::vector<std::string> tokens;
    std::string tok;
    while (ss >> tok)
      tokens.push_back(tok);

    // Single-token line = the prime
    if (tokens.size() == 1) {
      long long val = 0;
      try {
        val = std::stoll(tokens[0]);
      } catch (...) {
        continue;
      }
      return val;
    }
  }
  return -1; // not found
}

// Read all 11 primes from files, overriding hardcoded defaults
static void load_primes_from_files() {
  for (int i = 0; i < NUM_FOLDER_PRIMES; i++) {
    int fid = FOLDER_PRIMES[i].folder_id;
    std::string path;

    // Try 25_29/<N>_1.txt for groups 25-29
    if (fid >= 25 && fid <= 29) {
      // Try flat: 25_29/<N>_1.txt
      path = "25_29/" + std::to_string(fid) + "_1.txt";
      long long p = read_prime_from_file(path);
      if (p > 0) {
        FOLDER_PRIMES[i].prime = p;
        snprintf(FOLDER_PRIMES[i].label, sizeof(FOLDER_PRIMES[i].label),
                 "%d (p=%lld)", fid, p);
        printf("  [PRIME] %d -> %lld  (from %s)\n", fid, p, path.c_str());
        continue;
      }
      // Try nested: 25_29/<N>/<N>_1.txt
      path =
          "25_29/" + std::to_string(fid) + "/" + std::to_string(fid) + "_1.txt";
      p = read_prime_from_file(path);
      if (p > 0) {
        FOLDER_PRIMES[i].prime = p;
        snprintf(FOLDER_PRIMES[i].label, sizeof(FOLDER_PRIMES[i].label),
                 "%d (p=%lld)", fid, p);
        printf("  [PRIME] %d -> %lld  (from %s)\n", fid, p, path.c_str());
        continue;
      }
    }

    // Try exp/<N>_1.txt for groups 30-35
    if (fid >= 30 && fid <= 35) {
      path = "exp/" + std::to_string(fid) + "_1.txt";
      long long p = read_prime_from_file(path);
      if (p > 0) {
        FOLDER_PRIMES[i].prime = p;
        snprintf(FOLDER_PRIMES[i].label, sizeof(FOLDER_PRIMES[i].label),
                 "%d (p=%lld)", fid, p);
        printf("  [PRIME] %d -> %lld  (from %s)\n", fid, p, path.c_str());
        continue;
      }
    }

    // Fallback: use hardcoded default
    printf("  [PRIME] %d -> %lld  (HARDCODED FALLBACK — file not found)\n", fid,
           FOLDER_PRIMES[i].prime);
  }
}

// =============================================================================
// Parse matrix from Sage/Python [[...],[...],...] format
// =============================================================================

static MatrixData parse_matrix(const std::string &path) {
  std::ifstream ifs(path.c_str());
  if (!ifs.is_open())
    throw std::runtime_error("Cannot open: " + path);

  MatrixData md;
  md.filename = path.substr(path.find_last_of("/\\") + 1);

  std::vector<std::vector<long long>> rows;
  std::string line;
  bool started = false;

  while (std::getline(ifs, line)) {
    if (!started) {
      if (line.find("[[") != std::string::npos)
        started = true;
      else
        continue;
    }
    size_t lb = line.find('[');
    size_t rb = line.find(']');
    if (lb == std::string::npos || rb == std::string::npos || rb <= lb)
      continue;

    std::string tok = line.substr(lb + 1, rb - lb - 1);
    size_t fs = tok.find_first_not_of(" [");
    if (fs == std::string::npos)
      continue;
    tok = tok.substr(fs);
    std::replace(tok.begin(), tok.end(), ',', ' ');

    std::istringstream ss(tok);
    std::vector<long long> row;
    long long v;
    while (ss >> v)
      row.push_back(v);
    if (!row.empty())
      rows.push_back(row);
    if (line.find("]]") != std::string::npos)
      break;
  }

  if (rows.empty())
    throw std::runtime_error("No matrix data in: " + path);
  md.n = (int)rows.size();
  md.data.assign((size_t)md.n * md.n, 0LL);
  for (int i = 0; i < md.n; i++)
    for (int j = 0; j < (int)rows[i].size() && j < md.n; j++)
      md.data[(size_t)i * md.n + j] = rows[i][j];
  return md;
}

// =============================================================================
// Generate all C(pool_sz, r) index sets for one anchor position s
// =============================================================================

static void gen_combos(const std::vector<int> &pool, int r,
                       const int *principal, std::vector<IndexSet> &out) {
  int psz = (int)pool.size();
  if (r > psz)
    return;

  std::vector<int> cidx(r);
  for (int i = 0; i < r; i++)
    cidx[i] = i;

  while (true) {
    IndexSet is;
    is.k = PM_SIZE + r;
    for (int i = 0; i < PM_SIZE; i++)
      is.idx[i] = principal[i];
    for (int i = 0; i < r; i++)
      is.idx[PM_SIZE + i] = pool[cidx[i]];
    std::sort(is.idx, is.idx + is.k);
    out.push_back(is);

    // Advance combination
    int carry = r - 1;
    while (carry >= 0 && cidx[carry] == psz - r + carry)
      carry--;
    if (carry < 0)
      break;
    cidx[carry]++;
    for (int i = carry + 1; i < r; i++)
      cidx[i] = cidx[i - 1] + 1;
  }
}

// =============================================================================
// GPU buffer manager — 4 pointers only, runtime col_chunk
// =============================================================================

struct GPUBufs {
  IndexSet *d_row_sets;
  IndexSet *d_col_sets;
  long long *d_matrix;
  int *d_zero_flags;

  size_t cap_rows;  // allocated row-set capacity
  size_t col_chunk; // column chunk size (computed at runtime)
  size_t flag_cap;  // cap_rows * col_chunk

  GPUBufs()
      : d_row_sets(NULL), d_col_sets(NULL), d_matrix(NULL), d_zero_flags(NULL),
        cap_rows(0), col_chunk(0), flag_cap(0) {}

  void alloc(int n, int dev) {
    free_all();

    // Upper bound: C(n - PM_SIZE, dev) = C(n-2, dev)
    int pool = n - PM_SIZE;
    cap_rows = (size_t)nCr(pool, dev);

    // Compute col_chunk at runtime to stay within VRAM limit
    // Each thread needs one int flag; each col chunk needs col_chunk IndexSets
    // Main cost: flags = cap_rows * col_chunk * sizeof(int)
    size_t avail_bytes = (size_t)VRAM_LIMIT_MB * 1024ULL * 1024ULL;

    // Subtract fixed allocations
    size_t fixed = cap_rows * sizeof(IndexSet)          // d_row_sets
                   + (size_t)n * n * sizeof(long long); // d_matrix
    if (fixed >= avail_bytes) {
      fprintf(stderr, "[ERROR] Fixed allocations (%zu MB) exceed VRAM limit\n",
              fixed / (1024 * 1024));
      exit(1);
    }
    size_t remain = avail_bytes - fixed;

    // remain must hold: col_chunk * sizeof(IndexSet) + cap_rows * col_chunk *
    // sizeof(int) = col_chunk * (sizeof(IndexSet) + cap_rows * sizeof(int))
    size_t per_col = sizeof(IndexSet) + cap_rows * sizeof(int);
    size_t max_chunk = (per_col > 0) ? remain / per_col : 512;
    if (max_chunk < 1)
      max_chunk = 1;
    if (max_chunk > cap_rows)
      max_chunk = cap_rows;

    // Round down to power of 2 for alignment (capped at cap_rows)
    col_chunk = 1;
    while (col_chunk * 2 <= max_chunk)
      col_chunk *= 2;
    if (col_chunk > cap_rows)
      col_chunk = cap_rows;
    if (col_chunk < 1)
      col_chunk = 1;

    flag_cap = cap_rows * col_chunk;

    printf("    [GPU] Allocating: rows=%zu  col_chunk=%zu  flags=%zu (%.1f MB)"
           "  matrix=%d×%d\n",
           cap_rows, col_chunk, flag_cap, flag_cap * sizeof(int) / 1e6, n, n);
    fflush(stdout);

    CK(cudaMalloc(&d_row_sets, cap_rows * sizeof(IndexSet)));
    CK(cudaMalloc(&d_col_sets, col_chunk * sizeof(IndexSet)));
    CK(cudaMalloc(&d_matrix, (size_t)n * n * sizeof(long long)));
    CK(cudaMalloc(&d_zero_flags, flag_cap * sizeof(int)));
  }

  void free_all() {
    if (d_row_sets) {
      cudaFree(d_row_sets);
      d_row_sets = NULL;
    }
    if (d_col_sets) {
      cudaFree(d_col_sets);
      d_col_sets = NULL;
    }
    if (d_matrix) {
      cudaFree(d_matrix);
      d_matrix = NULL;
    }
    if (d_zero_flags) {
      cudaFree(d_zero_flags);
      d_zero_flags = NULL;
    }
  }

  ~GPUBufs() { free_all(); }
};

// =============================================================================
// Shared helper: write one zero minor's principal block + submatrix to FILE*
// =============================================================================

static void write_zero_minor_detail(FILE *f, const ZeroMinor &zm,
                                    const MatrixData &md) {
  // Column width from full matrix for consistent alignment
  int col_w = 1;
  for (int i = 0; i < md.n * md.n; i++) {
    int v = md.data[i], w = 1;
    if (v < 0) {
      v = -v;
      w = 2;
    }
    while (v >= 10) {
      v /= 10;
      w++;
    }
    if (w > col_w)
      col_w = w;
  }

  // Principal 2x2 block
  fprintf(f, "  Principal 2x2 block (s=%d, rows/cols {%d,%d}):\n", zm.s, zm.s,
          zm.s + 1);
  fprintf(f, "    [ %*d  %*d ]\n", col_w, md.data[zm.s * md.n + zm.s], col_w,
          md.data[zm.s * md.n + zm.s + 1]);
  fprintf(f, "    [ %*d  %*d ]\n", col_w, md.data[(zm.s + 1) * md.n + zm.s],
          col_w, md.data[(zm.s + 1) * md.n + zm.s + 1]);

  // Submatrix column width (tighter)
  int sw = 1;
  for (int r = 0; r < zm.k; r++)
    for (int c = 0; c < zm.k; c++) {
      int v = md.data[zm.row_idx[r] * md.n + zm.col_idx[c]], w = 1;
      if (v < 0) {
        v = -v;
        w = 2;
      }
      while (v >= 10) {
        v /= 10;
        w++;
      }
      if (w > sw)
        sw = w;
    }

  fprintf(f, "\n  Extracted %dx%d submatrix  (det mod p = 0):\n", zm.k, zm.k);

  // Column index header — mark principal block cols with '*'
  fprintf(f, "       ");
  for (int c = 0; c < zm.k; c++) {
    bool is_pm = (zm.col_idx[c] == zm.s || zm.col_idx[c] == zm.s + 1);
    fprintf(f, " %*d%c", sw, zm.col_idx[c], is_pm ? '*' : ' ');
  }
  fprintf(f, "\n       ");
  for (int c = 0; c < zm.k; c++) {
    fprintf(f, " ");
    for (int w = 0; w <= sw; w++)
      fprintf(f, "-");
  }
  fprintf(f, "\n");

  // Rows — mark principal block rows with '*', bracket the 2x2 cells
  for (int r = 0; r < zm.k; r++) {
    bool pm_r = (zm.row_idx[r] == zm.s || zm.row_idx[r] == zm.s + 1);
    fprintf(f, "  %3d%c |", zm.row_idx[r], pm_r ? '*' : ' ');
    for (int c = 0; c < zm.k; c++) {
      int val = md.data[zm.row_idx[r] * md.n + zm.col_idx[c]];
      bool pm_c = (zm.col_idx[c] == zm.s || zm.col_idx[c] == zm.s + 1);
      if (pm_r && pm_c)
        fprintf(f, "[%*d]", sw, val);
      else
        fprintf(f, " %*d ", sw, val);
    }
    fprintf(f, "\n");
  }
  fprintf(f, "\n");
}

// =============================================================================
// Write per-matrix result file
// =============================================================================

static void write_result_file(const std::string &out_dir, const MatrixData &md,
                              long long prime, int dev,
                              const std::vector<ZeroMinor> &minors,
                              double matrix_ms, double minors_tested) {
  std::string base = md.filename;
  size_t dot = base.rfind('.');
  if (dot != std::string::npos)
    base = base.substr(0, dot);
  std::string outpath = out_dir + "/" + base + "_result.txt";

  FILE *f = fopen(outpath.c_str(), "w");
  if (!f) {
    fprintf(stderr, "  [WARN] Cannot write: %s\n", outpath.c_str());
    return;
  }

  double avg_ms = (minors_tested > 0.0) ? matrix_ms / minors_tested : 0.0;

  fprintf(f, "============================================================\n");
  fprintf(f, "APM Result\n");
  fprintf(f, "============================================================\n");
  fprintf(f, "Matrix file       : %s\n", md.filename.c_str());
  fprintf(f, "Matrix size       : %d x %d\n", md.n, md.n);
  fprintf(f, "Prime (mod p)     : %lld\n", prime);
  fprintf(f, "PM block size     : %d (fixed)\n", PM_SIZE);
  fprintf(f, "Deviation level   : %d\n", dev);
  fprintf(f, "Minor size tested : %d x %d\n", PM_SIZE + dev, PM_SIZE + dev);
  fprintf(f, "------------------------------------------------------------\n");
  fprintf(f, "Timing\n");
  fprintf(f, "------------------------------------------------------------\n");
  fprintf(f, "Matrix total time : %.4f ms  (%.6f s)\n", matrix_ms,
          matrix_ms / 1000.0);
  fprintf(f, "Minors tested     : %.0f\n", minors_tested);
  fprintf(f, "Avg per minor     : %.8f ms\n", avg_ms);
  fprintf(f, "------------------------------------------------------------\n");
  fprintf(f, "Zero Minors Found : %zu\n", minors.size());
  fprintf(f,
          "============================================================\n\n");

  // Column width for full matrix
  int col_w = 1;
  for (int i = 0; i < md.n * md.n; i++) {
    int v = md.data[i], w = 1;
    if (v < 0) {
      v = -v;
      w = 2;
    }
    while (v >= 10) {
      v /= 10;
      w++;
    }
    if (w > col_w)
      col_w = w;
  }

  // Write the full source matrix
  fprintf(f, "Full matrix (%dx%d) mod %lld:\n", md.n, md.n, prime);
  fprintf(f, "     ");
  for (int c = 0; c < md.n; c++)
    fprintf(f, " %*d", col_w, c);
  fprintf(f, "\n     ");
  for (int c = 0; c < md.n; c++) {
    fprintf(f, " ");
    for (int w = 0; w < col_w; w++)
      fprintf(f, "-");
  }
  fprintf(f, "\n");
  for (int r = 0; r < md.n; r++) {
    fprintf(f, "%3d |", r);
    for (int c = 0; c < md.n; c++)
      fprintf(f, " %*d", col_w, md.data[r * md.n + c]);
    fprintf(f, "\n");
  }
  fprintf(f, "\n");

  if (minors.empty()) {
    fprintf(f, "No zero minor found at deviation level %d.\n", dev);
  } else {
    for (int mi = 0; mi < (int)minors.size(); mi++) {
      const ZeroMinor &zm = minors[mi];
      fprintf(f, "--- Zero Minor #%d ---\n", mi + 1);
      fprintf(f, "  Minor size (k)   : %d\n", zm.k);
      fprintf(f, "  Deviation        : %d\n", zm.dev);
      fprintf(f, "  Principal block s: %d  (indices {%d, %d})\n", zm.s, zm.s,
              zm.s + 1);
      fprintf(f, "  Row indices [%d]  : ", zm.k);
      for (int j = 0; j < zm.k; j++)
        fprintf(f, "%d%s", zm.row_idx[j], j < zm.k - 1 ? " " : "");
      fprintf(f, "\n");
      fprintf(f, "  Col indices [%d]  : ", zm.k);
      for (int j = 0; j < zm.k; j++)
        fprintf(f, "%d%s", zm.col_idx[j], j < zm.k - 1 ? " " : "");
      fprintf(f, "\n");
      fprintf(f, "  Time found       : %.4f ms  (since matrix start)\n",
              zm.time_ms);
      fprintf(f, "\n");
      write_zero_minor_detail(f, zm, md);
    }
  }

  fclose(f);
  printf("      -> %s\n", outpath.c_str());
  fflush(stdout);
}

// =============================================================================
// Search for the FIRST zero minor in one matrix at one deviation level.
// Returns immediately after finding one — does not scan further chunks or
// further s-positions.
// Returns: vector of at most 1 ZeroMinor
// =============================================================================

static std::vector<ZeroMinor>
search_matrix(GPUBufs &gpu,
              const long long *h_mat, // host: n*n flat matrix
              int n, long long prime, int dev, double &minors_tested,
              double matrix_start_ms) {
  std::vector<ZeroMinor> found;
  minors_tested = 0.0;

  const int BLOCK = 256;
  const int k = PM_SIZE + dev;

  // Upload matrix once
  CK(cudaMemcpy(gpu.d_matrix, h_mat, (size_t)n * n * sizeof(long long),
                cudaMemcpyHostToDevice));

  int pool_sz = n - PM_SIZE;

  // Loop over every principal block position s
  for (int s = 0; s <= n - PM_SIZE; s++) {
    int principal[PM_SIZE];
    for (int i = 0; i < PM_SIZE; i++)
      principal[i] = s + i;

    // Build deviation pool: all indices NOT in the principal block
    std::vector<int> pool;
    pool.reserve(pool_sz);
    for (int idx = 0; idx < n; idx++) {
      bool in_pm = false;
      for (int j = 0; j < PM_SIZE; j++)
        if (idx == principal[j]) {
          in_pm = true;
          break;
        }
      if (!in_pm)
        pool.push_back(idx);
    }

    // Generate all C(pool_sz, dev) index sets
    std::vector<IndexSet> sets;
    gen_combos(pool, dev, principal, sets);
    int num_sets = (int)sets.size();
    if (num_sets == 0)
      continue;

    if ((size_t)num_sets > gpu.cap_rows) {
      fprintf(stderr,
              "  [WARN] s=%d dev=%d: %d sets > GPU buffer %zu — skipping\n", s,
              dev, num_sets, gpu.cap_rows);
      continue;
    }

    // Upload all row index sets
    CK(cudaMemcpy(gpu.d_row_sets, sets.data(), num_sets * sizeof(IndexSet),
                  cudaMemcpyHostToDevice));

    // Chunk over col-sets
    int chunk_sz = (int)gpu.col_chunk;
    bool hit_found = false;

    for (int col_start = 0; col_start < num_sets; col_start += chunk_sz) {
      int col_end = std::min(col_start + chunk_sz, num_sets);
      int num_cols_this_chunk = col_end - col_start;

      long long total_jobs = (long long)num_sets * num_cols_this_chunk;
      minors_tested += (double)total_jobs;

      // Clear zero flags
      CK(cudaMemset(gpu.d_zero_flags, 0,
                    (size_t)num_sets * num_cols_this_chunk * sizeof(int)));

      // Upload col chunk
      CK(cudaMemcpy(gpu.d_col_sets, sets.data() + col_start,
                    num_cols_this_chunk * sizeof(IndexSet),
                    cudaMemcpyHostToDevice));

      // Launch kernel — grid computed with long long
      int grid = (int)((total_jobs + 255LL) / 256LL);
      apm_kernel<<<grid, BLOCK>>>(gpu.d_matrix, n, gpu.d_row_sets,
                                  gpu.d_col_sets, num_sets, num_cols_this_chunk,
                                  k, prime, gpu.d_zero_flags);

      CK(cudaDeviceSynchronize());
      CK(cudaGetLastError());

      // Copy flags back and scan for the FIRST zero only
      std::vector<int> h_flags((size_t)num_sets * num_cols_this_chunk);
      CK(cudaMemcpy(h_flags.data(), gpu.d_zero_flags,
                    h_flags.size() * sizeof(int), cudaMemcpyDeviceToHost));

      for (long long ji = 0; ji < total_jobs; ji++) {
        if (h_flags[ji]) {
          int ri = (int)(ji / num_cols_this_chunk);
          int ci = (int)(ji % num_cols_this_chunk) + col_start;

          ZeroMinor zm;
          zm.k = k;
          zm.dev = dev;
          zm.s = s;
          for (int j = 0; j < k; j++)
            zm.row_idx[j] = sets[ri].idx[j];
          for (int j = 0; j < k; j++)
            zm.col_idx[j] = sets[ci].idx[j];
          zm.time_ms = now_ms() - matrix_start_ms;
          found.push_back(zm);

          // ── FIRST HIT: stop all further searching for this matrix ──
          hit_found = true;
          break;
        }
      }

      if (hit_found)
        break; // stop chunk loop
    }   // col chunk loop

    if (hit_found)
      break; // stop s loop — matrix is done
  } // s loop

  return found;
}

// =============================================================================
// Process one prime group at one deviation level
// =============================================================================

static void run_group_deviation(int group, long long prime, int dev,
                                const std::vector<std::string> &files, int n,
                                GPUBufs &gpu) {
  std::string out_dir = make_out_dir(group, dev);
  std::string input_dir = "kernel_output/" + std::to_string(group) + "/";
  std::string prefix = "kernel_" + std::to_string(group) + "_";
  std::string group_str = std::to_string(group);

  // Header
  printf("\n");
  printf("  +----------------------------------------------------------+\n");
  printf("  |  Group %d  |  Deviation %d  |  %zu matrices\n", group, dev,
         files.size());
  printf("  |  Input : %s  (prefix: %s)\n", input_dir.c_str(), prefix.c_str());
  printf("  |  Output: %s\n", out_dir.c_str());
  printf("  |  Prime : %lld\n", prime);
  printf("  |  Minor : %dx%d\n", PM_SIZE + dev, PM_SIZE + dev);
  printf("  +----------------------------------------------------------+\n");
  fflush(stdout);

  // Open SUMMARY_detailed.txt
  std::string det_path = out_dir + "/SUMMARY_detailed.txt";
  FILE *det_f = fopen(det_path.c_str(), "w");
  if (det_f) {
    fprintf(det_f, "APM Summary\n");
    fprintf(det_f, "Deviation level : %d\n", dev);
    fprintf(det_f, "Prime group     : %d\n", group);
    fprintf(det_f, "Input folder    : %s\n", input_dir.c_str());
    fprintf(det_f, "File prefix     : %s\n", prefix.c_str());
    fprintf(det_f, "Prime           : %lld\n", prime);
    fprintf(det_f, "Matrix size n   : %d\n", n);
    fprintf(det_f, "PM block size   : %d\n", PM_SIZE);
    fprintf(det_f, "Matrices        : %zu\n", files.size());
    fprintf(det_f,
            "======================================================\n\n");
  }

  // Accumulators
  double folder_start_ms = now_ms();
  long long total_zero_minors = 0;
  double total_minors_tested = 0.0;
  int matrices_hit = 0;
  double total_time_ms = 0.0;

  // Per-matrix loop
  for (int fi = 0; fi < (int)files.size(); fi++) {
    printf("\n  [%d/%zu] %s\n", fi + 1, files.size(), files[fi].c_str());
    fflush(stdout);

    MatrixData md;
    try {
      md = parse_matrix(files[fi]);
    } catch (std::exception &ex) {
      printf("    [ERROR] %s — skipping\n", ex.what());
      continue;
    }
    if (md.n != n) {
      printf("    [WARN] n=%d != expected %d — skipping\n", md.n, n);
      continue;
    }

    double matrix_start_ms = now_ms();
    double minors_tested = 0.0;

    printf("    Searching for first zero minor at dev=%d ...\n", dev);
    fflush(stdout);

    std::vector<ZeroMinor> found = search_matrix(
        gpu, md.data.data(), n, prime, dev, minors_tested, matrix_start_ms);

    double matrix_ms = now_ms() - matrix_start_ms;

    total_zero_minors += (long long)found.size();
    total_minors_tested += minors_tested;
    total_time_ms += matrix_ms;
    if (!found.empty())
      matrices_hit++;

    // Terminal summary per matrix
    printf("    Time: %.3f s | Tested: %.0f minors | Zero minors: %zu\n",
           matrix_ms / 1000.0, minors_tested, found.size());

    if (found.empty()) {
      printf("    -> No zero minor at deviation %d\n", dev);
    } else {
      for (int mi = 0; mi < (int)found.size(); mi++) {
        const ZeroMinor &zm = found[mi];
        printf("    [Zero #%d] size=%d  dev=%d  s=%d  t=%.2f ms\n", mi + 1,
               zm.k, zm.dev, zm.s, zm.time_ms);
        printf("      Rows [%d]: ", zm.k);
        for (int j = 0; j < zm.k; j++)
          printf("%d%s", zm.row_idx[j], j < zm.k - 1 ? " " : "");
        printf("\n      Cols [%d]: ", zm.k);
        for (int j = 0; j < zm.k; j++)
          printf("%d%s", zm.col_idx[j], j < zm.k - 1 ? " " : "");
        printf("\n");
      }
    }
    fflush(stdout);

    // Write per-matrix result file
    write_result_file(out_dir, md, prime, dev, found, matrix_ms, minors_tested);

    // Append to SUMMARY_detailed.txt — only matrices with zeros
    if (det_f && !found.empty()) {
      fprintf(det_f,
              "------------------------------------------------------------\n");
      fprintf(det_f, "[%d/%d] %s\n", fi + 1, (int)files.size(),
              md.filename.c_str());
      fprintf(det_f, "  time=%.4f s | tested=%.0f | zeros=%zu\n",
              matrix_ms / 1000.0, minors_tested, found.size());
      for (int mi = 0; mi < (int)found.size(); mi++) {
        const ZeroMinor &zm = found[mi];
        fprintf(det_f, "\n  --- Zero Minor #%d ---\n", mi + 1);
        fprintf(det_f, "  Minor size (k)   : %d\n", zm.k);
        fprintf(det_f, "  Deviation        : %d\n", zm.dev);
        fprintf(det_f, "  Principal block s: %d  (indices {%d, %d})\n", zm.s,
                zm.s, zm.s + 1);
        fprintf(det_f, "  Row indices [%d]  : ", zm.k);
        for (int j = 0; j < zm.k; j++)
          fprintf(det_f, "%d%s", zm.row_idx[j], j < zm.k - 1 ? " " : "");
        fprintf(det_f, "\n");
        fprintf(det_f, "  Col indices [%d]  : ", zm.k);
        for (int j = 0; j < zm.k; j++)
          fprintf(det_f, "%d%s", zm.col_idx[j], j < zm.k - 1 ? " " : "");
        fprintf(det_f, "\n");
        fprintf(det_f, "  Time found       : %.4f ms\n\n", zm.time_ms);
        write_zero_minor_detail(det_f, zm, md);
      }
    }
  } // end matrix loop

  double folder_ms = now_ms() - folder_start_ms;

  // Close SUMMARY_detailed.txt with footer
  if (det_f) {
    fprintf(det_f, "======================================================\n");
    fprintf(det_f, "FOLDER TOTALS\n");
    fprintf(det_f, "Total time     : %.4f s\n", folder_ms / 1000.0);
    fprintf(det_f, "Matrices       : %zu\n", files.size());
    fprintf(det_f, "Minors tested  : %.0f\n", total_minors_tested);
    fprintf(det_f, "Zero minors    : %lld\n", total_zero_minors);
    fclose(det_f);
    printf("  Summary (detailed) -> %s\n", det_path.c_str());
  }

  // Write SUMMARY_brief.txt
  std::string brief_path = out_dir + "/SUMMARY_brief.txt";
  FILE *brief_f = fopen(brief_path.c_str(), "w");
  if (brief_f) {
    double hit_pct =
        (files.size() > 0) ? 100.0 * matrices_hit / (double)files.size() : 0.0;
    fprintf(brief_f,
            "============================================================\n");
    fprintf(brief_f, "APM Brief Summary\n");
    fprintf(brief_f,
            "============================================================\n");
    fprintf(brief_f, "Prime group      : %d\n", group);
    fprintf(brief_f, "Prime (p)        : %lld\n", prime);
    fprintf(brief_f, "Deviation level  : %d\n", dev);
    fprintf(brief_f, "Minor size       : %d x %d\n", PM_SIZE + dev,
            PM_SIZE + dev);
    fprintf(brief_f, "PM block size    : %d\n", PM_SIZE);
    fprintf(brief_f, "Matrix size (n)  : %d\n", n);
    fprintf(brief_f, "Input folder     : %s\n",
            ("kernel_output/" + std::to_string(group) + "/").c_str());
    fprintf(brief_f, "Output folder    : %s\n", out_dir.c_str());
    fprintf(brief_f,
            "------------------------------------------------------------\n");
    fprintf(brief_f, "Total matrices   : %zu\n", files.size());
    fprintf(brief_f,
            "Matrices hit     : %d      "
            "(contain at least one zero minor)\n",
            matrices_hit);
    fprintf(brief_f, "Hit ratio        : %d/%zu = %.2f%%\n", matrices_hit,
            files.size(), hit_pct);
    fprintf(brief_f, "Total minors tested  : %.0f\n", total_minors_tested);
    fprintf(brief_f, "Total zero minors    : %lld\n", total_zero_minors);
    fprintf(brief_f, "Total time           : %.3f s\n", folder_ms / 1000.0);
    fprintf(brief_f, "Avg time per matrix  : %.3f s\n",
            (files.size() > 0) ? folder_ms / 1000.0 / files.size() : 0.0);
    fprintf(brief_f,
            "============================================================\n");
    fclose(brief_f);
    printf("  Summary (brief)    -> %s\n", brief_path.c_str());
  }

  // Terminal totals
  printf("\n  -- group=%d  dev=%d complete --\n", group, dev);
  printf("  Time           : %.3f s\n", folder_ms / 1000.0);
  printf("  Matrices       : %zu\n", files.size());
  printf("  Minors tested  : %.0f\n", total_minors_tested);
  printf("  Zero minors    : %lld\n", total_zero_minors);
  printf("  Matrices hit   : %d / %zu\n", matrices_hit, files.size());
  fflush(stdout);
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char **argv) {
  int gmin = 25, gmax = 35;
  if (argc == 3) {
    gmin = atoi(argv[1]);
    gmax = atoi(argv[2]);
    if (gmin > gmax) {
      int tmp = gmin;
      gmin = gmax;
      gmax = tmp;
    }
  } else if (argc != 1) {
    printf("Usage: %s [gmin gmax]\n", argv[0]);
    printf("  Example: %s 32 35\n", argv[0]);
    printf("  Example: %s 25 27\n", argv[0]);
    return 1;
  }

  // ── Program header ──────────────────────────────────────────────────────
  printf("=================================================================\n");
  printf(" APM First-Hit Search — apm_research_hits_one\n");
  printf(" KERNEL_OUTPUT VERSION — prime groups from kernel_output/<g>/\n");
  printf(" GROUP-OUTER, DEVIATION-INNER LOOP\n");
  printf(" STOPS after the FIRST zero minor per matrix\n");
  printf(" PM block size    : %d (fixed)\n", PM_SIZE);
  printf(" Max index static : %d\n", MAX_IDX_STATIC);
  printf(
      "=================================================================\n\n");
  printf(" Group range     : %d to %d\n\n", gmin, gmax);
  fflush(stdout);

  // ── Read primes from files ──────────────────────────────────────────────
  printf("Reading primes from files...\n");
  load_primes_from_files();
  printf("\nPrime table:\n");
  for (int i = 0; i < NUM_FOLDER_PRIMES; i++) {
    printf("  %2d -> %15lld  %s\n", FOLDER_PRIMES[i].folder_id,
           FOLDER_PRIMES[i].prime, FOLDER_PRIMES[i].label);
  }
  printf("\n");
  fflush(stdout);

  // ── Check kernel_output directory ────────────────────────────────────────
  struct stat st;
  if (stat("kernel_output", &st) != 0) {
    printf("[FATAL] kernel_output/ directory not found. Nothing to do.\n");
    return 1;
  }

  double prog_start = now_ms();

  // =========================================================================
  // MAIN LOOP — group-outer, deviation-inner
  //
  // For each prime group g = 25, 26, ..., 35:
  //   Read prime, collect files, detect n
  //   For each deviation d = 2, 3, ..., n-3:
  //     Process all matrices
  // =========================================================================

  for (int gi = 0; gi < NUM_FOLDER_PRIMES; gi++) {
    int group = FOLDER_PRIMES[gi].folder_id;
    long long prime = FOLDER_PRIMES[gi].prime;
    if (group < gmin || group > gmax)
      continue;

    printf("\n");
    printf(
        "#################################################################\n");
    printf("# GROUP %d   prime = %lld\n", group, prime);
    printf(
        "#################################################################\n");
    fflush(stdout);

    // Collect files
    std::string input_dir = "kernel_output/" + std::to_string(group);
    std::string prefix = "kernel_" + std::to_string(group) + "_";
    std::vector<std::string> files = collect_files(input_dir, prefix);
    if (files.empty()) {
      std::string alt_prefix = std::to_string(group) + "_";
      files = collect_files(input_dir, alt_prefix);
      if (!files.empty()) {
        printf("  [INFO] No files matching prefix '%s'. Using prefix '%s' instead.\n",
               prefix.c_str(), alt_prefix.c_str());
        prefix = alt_prefix;
      }
    }

    if (files.empty()) {
      printf("  [SKIP] No matching files in %s\n", input_dir.c_str());
      continue;
    }

    // Parse first matrix to detect n
    MatrixData first;
    try {
      first = parse_matrix(files[0]);
    } catch (std::exception &ex) {
      printf("  [ERROR] Cannot parse first file: %s\n", ex.what());
      continue;
    }
    int n = first.n;

    int max_dev = n - (PM_SIZE + 1); // n - 3
    printf("  Matrix size n   = %d\n", n);
    printf("  Files found     = %zu\n", files.size());
    printf("  Deviation range = %d to %d\n", MIN_DEV, max_dev);
    fflush(stdout);

    if (max_dev < MIN_DEV) {
      printf("  [SKIP] max_dev=%d < MIN_DEV=%d — matrix too small\n", max_dev,
             MIN_DEV);
      continue;
    }

    double group_start = now_ms();

    // Deviation-inner loop
    for (int dev = MIN_DEV; dev <= max_dev; dev++) {
      // Check feasibility
      if (dev > n - PM_SIZE) {
        printf("  [SKIP] dev=%d needs pool >= %d but pool=%d\n", dev, dev,
               n - PM_SIZE);
        continue;
      }

      printf("\n");
      printf(
          "  "
          "===============================================================\n");
      printf("  [group=%d | dev=%d/%d]  minor=%dx%d\n", group, dev, max_dev,
             PM_SIZE + dev, PM_SIZE + dev);
      printf(
          "  "
          "===============================================================\n");
      fflush(stdout);

      // Allocate GPU buffers for this (n, dev) combination
      GPUBufs gpu;
      gpu.alloc(n, dev);

      run_group_deviation(group, prime, dev, files, n, gpu);

      // gpu freed by destructor
      printf("  [group=%d  dev=%d done]\n", group, dev);
      fflush(stdout);
    }

    double group_ms = now_ms() - group_start;
    printf("\n");
    printf(
        "#################################################################\n");
    printf("# GROUP %d COMPLETE   (%.2f min)\n", group, group_ms / 60000.0);
    printf(
        "#################################################################\n");
    fflush(stdout);
  }

  double total_ms = now_ms() - prog_start;
  printf(
      "\n=================================================================\n");
  printf(" ALL DONE\n");
  printf(" Total wall time : %.3f s  (%.2f min)\n", total_ms / 1000.0,
         total_ms / 60000.0);
  printf("\n Results layout:\n");
  for (int gi = 0; gi < NUM_FOLDER_PRIMES; gi++) {
    int g = FOLDER_PRIMES[gi].folder_id;
    // Detect n from first file (approximate)
    std::string idir = "kernel_output/" + std::to_string(g);
    std::string pfx = "kernel_" + std::to_string(g) + "_";
    std::vector<std::string> fs = collect_files(idir, pfx);
    int nn = 0;
    if (!fs.empty()) {
      try {
        MatrixData m = parse_matrix(fs[0]);
        nn = m.n;
      } catch (...) {
      }
    }
    int md = nn > 0 ? nn - (PM_SIZE + 1) : 5;
    for (int d = MIN_DEV; d <= md; d++)
      printf("   Results_hits_one/%d/deviation_%d/\n", g, d);
  }
  printf("=================================================================\n");

  return 0;
}

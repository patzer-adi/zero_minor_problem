// =============================================================================
// apm_research_4.cu
//
// APM — Almost Principal Minors, Exhaustive Search
// Extended version: handles p_1048573 folders AND 25_29/ multi-prime matrices
//
// What this program does:
//   THREE PHASES, in strict order:
//
//   Phase 1 — p_1048573  antioffset (12×12 matrices, fast)
//     For EACH deviation level D in {2, 3, 4, 5}:
//       For EACH matrix file in p_1048573_antioffset=0.2/:
//         For EACH principal block position s (0 to n-2):
//           Generate ALL C(n-2, D) index sets for deviation level D
//           For EVERY (row_set × col_set) pair:
//             Compute det(submatrix) mod prime
//             If det == 0 → record it (NEVER stop early, find ALL)
//         Write one result file per matrix
//       Write one SUMMARY.txt per folder
//
//   Phase 2 — 25_29/ subfolders (25, 26, 27, 28, 29)
//     Each subfolder has its own prime:
//       25 → 33554393     26 → 44923183     27 → 134217689
//       28 → 268435399    29 → 536870909
//     Each subfolder contains c1/c2/c3 × 100 kernel matrices (75×75).
//     Files ending in _RN.txt are IGNORED.
//     Same exhaustive search as Phase 1, deviation levels 2→5.
//     All antioffset → processed here.
//
//   Phase 3 — p_1048573  offset (larger matrices, slow)
//     Same algorithm, deviation levels 2→5, all matrices.
//
// Output folder structure created automatically:
//   Results/
//     deviation_2/
//       antioffset/          ← p_1048573 antioffset results
//       25_29_25/            ← subfolder 25 results (prime=33554393)
//       25_29_26/            ← subfolder 26 results (prime=44923183)
//       25_29_27/            ← subfolder 27 results
//       25_29_28/            ← subfolder 28 results
//       25_29_29/            ← subfolder 29 results
//       offset/              ← p_1048573 offset results
//     deviation_3/
//       ...  (same sub-structure)
//     deviation_4/
//       ...
//     deviation_5/
//       ...
//
// Each result file contains:
//   - Matrix info, prime, deviation level
//   - Timing: total matrix time, minors tested, avg time per minor
//   - Every zero minor found: size, row indices, col indices, time found
//   - Extracted submatrix with principal block highlighted
//
// COMPILE:
//   nvcc -O3 -std=c++14 -ccbin gcc-10           \
//        -Wno-deprecated-gpu-targets             \
//        -gencode arch=compute_50,code=sm_50     \
//        -gencode arch=compute_50,code=compute_50 \
//        apm_research_4.cu -o apm4d              \
//        -Xlinker -lstdc++
//
// RUN (all paths and primes are hardcoded):
//   ./apm4d
//
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

// =============================================================================
// Compile-time configuration
// =============================================================================

#define PM_SIZE 2     // principal minor block size — FIXED at 2 (thesis §5.3.3)
#define MIN_DEV 2     // minimum deviation to test
#define MAX_DEV 5     // maximum deviation to test
#define MAX_IDX 7     // PM_SIZE + MAX_DEV  (2 + 5 = 7)
#define COL_CHUNK 512 // col-sets per GPU kernel launch (controls VRAM usage)

// =============================================================================
// Hardcoded prime table for 25_29/ subfolders
// The .txt matrix files do NOT store the prime, so we map it here.
// =============================================================================

struct FolderPrime {
  int folder_id;     // 25, 26, 27, 28, 29
  long long prime;   // the prime used to generate matrices in that folder
  const char *label; // display label
};

static const FolderPrime FOLDER_PRIMES[] = {
    {25, 33554393LL, "25 (p=33554393)"},
    {26, 44923183LL, "26 (p=44923183)"},
    {27, 134217689LL, "27 (p=134217689)"},
    {28, 268435399LL, "28 (p=268435399)"},
    {29, 536870909LL, "29 (p=536870909)"},
};
static const int NUM_FOLDER_PRIMES = 5;

// Prime for p_1048573 folders
static const long long P_1048573 = 1048573LL;

// =============================================================================
// Data structures
// =============================================================================

// One row-set or col-set: sorted indices into the matrix
struct IndexSet {
  int k;            // number of indices  (= PM_SIZE + dev)
  int idx[MAX_IDX]; // sorted matrix indices
};

// A fully resolved zero minor (for saving to file)
struct ZeroMinor {
  int k;
  int dev;
  int s; // principal block start position
  int row_idx[MAX_IDX];
  int col_idx[MAX_IDX];
  double time_ms; // ms elapsed since start of this matrix
};

struct MatrixData {
  std::vector<int> data;
  int n;
  std::string filename;
};

// =============================================================================
// Device: modular arithmetic helpers
// =============================================================================

__device__ inline int mod_sub(int a, int b, int p) {
  int r = a - b;
  return (r < 0) ? r + p : r;
}

__device__ inline int mod_mul(int a, int b, int p) {
  return (int)((1LL * a * b) % p);
}

__device__ int mod_inv(int a, int p) {
  // Extended Euclidean algorithm.
  // Use long long throughout — intermediate values like (q * nt) can exceed
  // INT_MAX when p is large (e.g. 536870909 for folder 29: q*nt can reach
  // ~2.88e17 which overflows int32 and silently produces wrong inverses).
  long long t = 0, nt = 1, r = (long long)p, nr = (long long)a;
  while (nr) {
    long long q = r / nr, tmp;
    tmp = t;  t = nt;  nt = tmp - q * nt;
    tmp = r;  r = nr;  nr = tmp - q * nr;
  }
  return (int)((t < 0) ? t + p : t);
}

// =============================================================================
// Device: determinant mod p via Gaussian elimination
// mat is flat k×k row-major, k <= MAX_IDX (7)
// =============================================================================

__device__ int det_mod(const int *mat, int k, int p) {
  int a[MAX_IDX][MAX_IDX];
  for (int i = 0; i < k; i++)
    for (int j = 0; j < k; j++)
      a[i][j] = mat[i * k + j];

  int det = 1;
  for (int col = 0; col < k; col++) {
    // Find pivot
    int piv = -1;
    for (int row = col; row < k; row++)
      if (a[row][col]) {
        piv = row;
        break;
      }
    if (piv < 0)
      return 0; // singular

    if (piv != col) {
      for (int j = col; j < k; j++) {
        int tmp = a[col][j];
        a[col][j] = a[piv][j];
        a[piv][j] = tmp;
      }
      det = (p - det) % p; // row swap negates determinant
    }

    det = mod_mul(det, a[col][col], p);
    int inv = mod_inv(a[col][col], p);

    for (int row = col + 1; row < k; row++) {
      if (!a[row][col])
        continue;
      int f = mod_mul(a[row][col], inv, p);
      for (int j = col + 1; j < k; j++)
        a[row][j] = mod_sub(a[row][j], mod_mul(f, a[col][j], p), p);
      a[row][col] = 0;
    }
  }
  return det;
}

// =============================================================================
// CUDA kernel — finds ALL zero minors, not just the first one per chunk.
//
// Each thread handles ONE (row_set, col_set) pair.
// d_zero_flags[tid] is set to 1 if that pair's determinant is zero.
// The host then scans d_zero_flags[] to collect every hit.
//
// This replaces the previous atomicCAS approach that silently dropped all
// zero minors after the first one found in each chunk launch.
// =============================================================================

__global__ void
apm_kernel(const int *d_matrix,        // n*n matrix (device)
           int n,
           const IndexSet *d_row_sets, // all row index sets for this (s, dev)
           const IndexSet *d_col_sets, // col index sets for current chunk
           int num_row_sets,
           int num_col_sets,
           int k,
           int prime,
           int *d_zero_flags)          // output: 1 per job if det==0, else 0
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_jobs = num_row_sets * num_col_sets;
  if (tid >= total_jobs)
    return;

  int r = tid / num_col_sets;
  int c = tid % num_col_sets;

  const IndexSet &rs = d_row_sets[r];
  const IndexSet &cs = d_col_sets[c];

  // Extract k×k submatrix
  int sub[MAX_IDX * MAX_IDX];
  for (int i = 0; i < k; i++)
    for (int j = 0; j < k; j++)
      sub[i * k + j] = d_matrix[rs.idx[i] * n + cs.idx[j]];

  d_zero_flags[tid] = (det_mod(sub, k, prime) == 0) ? 1 : 0;
}

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

// Create a single directory (ignore if exists)
static void mkdir_safe(const std::string &p) { mkdir(p.c_str(), 0755); }

// Create Results/deviation_<d>/<tag>/ and return full path
static std::string make_out_dir(int dev, const std::string &tag) {
  mkdir_safe("Results");
  std::string d1 = "Results/deviation_" + std::to_string(dev);
  mkdir_safe(d1);
  std::string d2 = d1 + "/" + tag;
  mkdir_safe(d2);
  return d2;
}

// Check if a filename ends with a given suffix
static bool ends_with(const std::string &str, const std::string &suffix) {
  if (suffix.size() > str.size())
    return false;
  return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Collect all .txt files in a directory, sorted.
// SKIPS files ending in _RN.txt (these are row-number metadata files).
static std::vector<std::string> collect_files(const std::string &path) {
  std::vector<std::string> files;
  struct stat st;
  if (stat(path.c_str(), &st) != 0)
    return files;
  if (S_ISDIR(st.st_mode)) {
    DIR *d = opendir(path.c_str());
    if (!d)
      return files;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
      std::string nm = e->d_name;
      // Must end with .txt
      if (nm.size() > 4 && nm.substr(nm.size() - 4) == ".txt") {
        // SKIP files ending in _RN.txt
        if (ends_with(nm, "_RN.txt"))
          continue;
        files.push_back(path + "/" + nm);
      }
    }
    closedir(d);
  } else {
    files.push_back(path);
  }
  std::sort(files.begin(), files.end());
  return files;
}

// Parse matrix from Sage/Python [[...],[...],...] format
static MatrixData parse_matrix(const std::string &path) {
  std::ifstream ifs(path.c_str());
  if (!ifs.is_open())
    throw std::runtime_error("Cannot open: " + path);

  MatrixData md;
  md.filename = path.substr(path.find_last_of("/\\") + 1);

  std::vector<std::vector<int>> rows;
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
    std::vector<int> row;
    int v;
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
  md.data.assign((size_t)md.n * md.n, 0);
  for (int i = 0; i < md.n; i++)
    for (int j = 0; j < (int)rows[i].size() && j < md.n; j++)
      md.data[(size_t)i * md.n + j] = rows[i][j];
  return md;
}

// Generate all C(pool_sz, r) index sets for this (s, dev)
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
// Shared helper: write one zero minor's principal block + submatrix to FILE*
// Used by both write_result_file and the SUMMARY writer.
// =============================================================================

static void write_zero_minor_detail(FILE *f, const ZeroMinor &zm,
                                    const MatrixData &md) {
  // Find column width from the full matrix for consistent alignment
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

  // Find widest entry in just this submatrix for tighter column formatting
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
// Write result file for one matrix
// =============================================================================

static void write_result_file(const std::string &out_dir, const MatrixData &md,
                              long long prime, int dev,
                              const std::vector<ZeroMinor> &minors,
                              double matrix_ms, double minors_tested) {
  // Build output path: out_dir/<matrixname_without_ext>_result.txt
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

  // ── Helper: find the widest entry in the full matrix (for column alignment)
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

  // ── Write the full source matrix once at the top ──────────────────────────
  fprintf(f, "Full matrix (%dx%d) mod %lld:\n", md.n, md.n, prime);
  // Column index header
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
      // Write row indices
      fprintf(f, "  Row indices [%d]  : ", zm.k);
      for (int j = 0; j < zm.k; j++)
        fprintf(f, "%d%s", zm.row_idx[j], j < zm.k - 1 ? " " : "");
      fprintf(f, "\n");
      // Write col indices
      fprintf(f, "  Col indices [%d]  : ", zm.k);
      for (int j = 0; j < zm.k; j++)
        fprintf(f, "%d%s", zm.col_idx[j], j < zm.k - 1 ? " " : "");
      fprintf(f, "\n");
      fprintf(f, "  Time found       : %.4f ms  (since matrix start)\n",
              zm.time_ms);
      fprintf(f, "\n");
      // Print principal block + submatrix using shared helper
      write_zero_minor_detail(f, zm, md);
    }
  }

  fclose(f);
  printf("      → %s\n", outpath.c_str());
  fflush(stdout);
}

// =============================================================================
// GPU buffer manager — one instance per (dev, n) combination.
//
// Key design changes vs original:
//   - No MinorJob array (eliminated the ~82 GB allocation for 75×75 matrices)
//   - No ChunkResult / found_any (replaced by zero_flags per-thread array)
//   - zero_flags sized to COL_CHUNK * cap_rows (the max jobs per launch)
// =============================================================================

struct GPUBufs {
  IndexSet *row_sets;   // all C(pool, dev) row index sets
  IndexSet *col_sets;   // one chunk of col index sets  (size COL_CHUNK)
  int      *matrix;     // flat n×n matrix
  int      *zero_flags; // one int per (row,col) pair in this chunk launch

  size_t cap_rows;      // C(n-PM_SIZE, dev) — row set count upper bound
  size_t cap_flags;     // max jobs per launch = cap_rows * COL_CHUNK

  GPUBufs()
      : row_sets(NULL), col_sets(NULL), matrix(NULL), zero_flags(NULL),
        cap_rows(0), cap_flags(0) {}

  void alloc(int n, int dev) {
    free_all();

    // Compute C(pool, dev) — safe running-product formula
    int pool = n - PM_SIZE;
    long long c = 1;
    for (int i = 0; i < dev && i < pool; i++)
      c = c * (pool - i) / (i + 1);
    cap_rows  = (size_t)c;
    cap_flags = cap_rows * (size_t)COL_CHUNK;

    double row_mb   = cap_rows  * sizeof(IndexSet) / 1e6;
    double flags_mb = cap_flags * sizeof(int)       / 1e6;

    printf("    [GPU] Alloc: row_sets=%zu (%.1f MB)  col_sets=%d (%.2f MB)"
           "  zero_flags=%zu (%.1f MB)  matrix=%dx%d\n",
           cap_rows, row_mb,
           COL_CHUNK, COL_CHUNK * sizeof(IndexSet) / 1e6,
           cap_flags, flags_mb, n, n);
    fflush(stdout);

    CK(cudaMalloc(&row_sets,   cap_rows  * sizeof(IndexSet)));
    CK(cudaMalloc(&col_sets,   (size_t)COL_CHUNK * sizeof(IndexSet)));
    CK(cudaMalloc(&matrix,     (size_t)n * n * sizeof(int)));
    CK(cudaMalloc(&zero_flags, cap_flags * sizeof(int)));
  }

  void free_all() {
    if (row_sets)   { cudaFree(row_sets);   row_sets   = NULL; }
    if (col_sets)   { cudaFree(col_sets);   col_sets   = NULL; }
    if (matrix)     { cudaFree(matrix);     matrix     = NULL; }
    if (zero_flags) { cudaFree(zero_flags); zero_flags = NULL; }
    cap_rows = cap_flags = 0;
  }

  ~GPUBufs() { free_all(); }
};

// =============================================================================
// Search all zero minors in one matrix at one deviation level
// Returns: vector of all zero minors found
// Writes:  minors_tested (total number of det evaluations)
// =============================================================================

// =============================================================================
// Search all zero minors in one matrix at one deviation level.
//
// Key corrections vs original:
//   1. No MinorJob array — thread index encodes (r, c) directly.
//   2. zero_flags[] — one flag per thread — captures EVERY zero minor,
//      not just the first one per chunk.
//   3. Host scans zero_flags after each launch to collect all hits.
// =============================================================================

static std::vector<ZeroMinor>
search_matrix(GPUBufs &gpu,
              const int *h_mat,
              int n,
              int prime,      // safe: all our primes fit in int
              int dev,
              double &minors_tested,
              double matrix_start_ms)
{
  std::vector<ZeroMinor> found;
  minors_tested = 0.0;

  const int BLOCK = 256;
  const int k = PM_SIZE + dev;

  // Upload matrix once per matrix
  CK(cudaMemcpy(gpu.matrix, h_mat, (size_t)n * n * sizeof(int),
                cudaMemcpyHostToDevice));

  int pool_sz = n - PM_SIZE;

  // ── Loop over every principal block start position s ────────────────────
  for (int s = 0; s <= n - PM_SIZE; s++) {
    int principal[PM_SIZE];
    for (int i = 0; i < PM_SIZE; i++)
      principal[i] = s + i;

    // Build pool: indices NOT in the principal block
    std::vector<int> pool;
    pool.reserve(pool_sz);
    for (int idx = 0; idx < n; idx++) {
      bool in_pm = false;
      for (int j = 0; j < PM_SIZE; j++)
        if (idx == principal[j]) { in_pm = true; break; }
      if (!in_pm) pool.push_back(idx);
    }

    // Generate all C(pool_sz, dev) index sets (same for rows AND cols)
    std::vector<IndexSet> sets;
    gen_combos(pool, dev, principal, sets);
    int num_sets = (int)sets.size();
    if (num_sets == 0) continue;

    if ((size_t)num_sets > gpu.cap_rows) {
      fprintf(stderr,
              "  [WARN] s=%d dev=%d: %d sets exceeds GPU buffer %zu — skipping\n",
              s, dev, num_sets, gpu.cap_rows);
      continue;
    }

    // Upload all row/col index sets (identical pool for rows and cols)
    CK(cudaMemcpy(gpu.row_sets, sets.data(), num_sets * sizeof(IndexSet),
                  cudaMemcpyHostToDevice));

    // ── Chunk over col-sets to keep GPU memory bounded ─────────────────────
    for (int col_start = 0; col_start < num_sets; col_start += COL_CHUNK) {
      int col_end  = std::min(col_start + COL_CHUNK, num_sets);
      int num_cols = col_end - col_start;
      int num_jobs = num_sets * num_cols;

      minors_tested += (double)num_jobs;

      // Upload this col chunk
      CK(cudaMemcpy(gpu.col_sets, sets.data() + col_start,
                    num_cols * sizeof(IndexSet), cudaMemcpyHostToDevice));

      // Clear zero_flags for this launch
      CK(cudaMemset(gpu.zero_flags, 0, (size_t)num_jobs * sizeof(int)));

      // Launch: one thread per (row_set, col_set) pair
      int grid_x = (num_jobs + BLOCK - 1) / BLOCK;
      apm_kernel<<<dim3(grid_x, 1, 1), BLOCK>>>(
          gpu.matrix, n,
          gpu.row_sets, gpu.col_sets,
          num_sets, num_cols, k, prime,
          gpu.zero_flags);

      CK(cudaDeviceSynchronize());
      CK(cudaGetLastError());

      // Pull flags back and scan for ALL zeros — no silent drops
      std::vector<int> h_flags(num_jobs);
      CK(cudaMemcpy(h_flags.data(), gpu.zero_flags,
                    num_jobs * sizeof(int), cudaMemcpyDeviceToHost));

      for (int ji = 0; ji < num_jobs; ji++) {
        if (!h_flags[ji]) continue;
        int ri = ji / num_cols;
        int ci = ji % num_cols + col_start; // absolute index in sets[]

        ZeroMinor zm;
        zm.k   = k;
        zm.dev = dev;
        zm.s   = s;
        for (int j = 0; j < k; j++) zm.row_idx[j] = sets[ri].idx[j];
        for (int j = 0; j < k; j++) zm.col_idx[j] = sets[ci].idx[j];
        zm.time_ms = now_ms() - matrix_start_ms;
        found.push_back(zm);
      }
    } // col chunk loop
  } // s loop

  return found;
}

// =============================================================================
// Process one folder at one deviation level
// =============================================================================

static void run_folder(
    const std::string &folder_path,
    const std::string &folder_tag, // "offset", "antioffset", "25_29_25", etc.
    long long prime, int dev) {
  std::string out_dir = make_out_dir(dev, folder_tag);
  std::vector<std::string> files = collect_files(folder_path);

  if (files.empty()) {
    printf("  [SKIP] No .txt files (or all _RN.txt): %s\n",
           folder_path.c_str());
    return;
  }

  // ── Header ───────────────────────────────────────────────────────────────
  printf("\n");
  printf("  ╔══════════════════════════════════════════════════════════╗\n");
  printf("  ║  Deviation %d  |  %-12s  |  %zu matrices             \n", dev,
         folder_tag.c_str(), files.size());
  printf("  ║  Input : %s\n", folder_path.c_str());
  printf("  ║  Output: %s\n", out_dir.c_str());
  printf("  ║  Prime : %lld\n", prime);
  printf("  ╚══════════════════════════════════════════════════════════╝\n");
  fflush(stdout);

  // Parse first matrix to get n
  MatrixData first;
  try {
    first = parse_matrix(files[0]);
  } catch (std::exception &ex) {
    printf("  [ERROR] Cannot parse first file: %s\n", ex.what());
    return;
  }
  int n = first.n;
  printf("  Matrix size n = %d\n", n);

  // Check feasibility for this deviation
  if (dev > n - PM_SIZE) {
    printf("  [SKIP] dev=%d needs pool >= %d but pool = n - PM_SIZE = %d\n",
           dev, dev, n - PM_SIZE);
    return;
  }

  // Allocate GPU buffers (once for all matrices in this folder + dev)
  GPUBufs gpu;
  gpu.alloc(n, dev);

  // Open SUMMARY.txt for this folder
  std::string sum_path = out_dir + "/SUMMARY.txt";
  FILE *sum_f = fopen(sum_path.c_str(), "w");
  if (!sum_f) {
    fprintf(stderr, "  [WARN] Cannot open SUMMARY.txt: %s\n", sum_path.c_str());
  } else {
    fprintf(sum_f, "APM Summary\n");
    fprintf(sum_f, "Deviation level : %d\n", dev);
    fprintf(sum_f, "Folder          : %s\n", folder_path.c_str());
    fprintf(sum_f, "Prime           : %lld\n", prime);
    fprintf(sum_f, "Matrix size n   : %d\n", n);
    fprintf(sum_f, "PM block size   : %d\n", PM_SIZE);
    fprintf(sum_f, "Matrices        : %zu\n", files.size());
    fprintf(sum_f,
            "======================================================\n\n");
  }

  // Folder-level accumulators
  double folder_start_ms = now_ms();
  long long folder_zero_minors = 0;
  double folder_minors_total = 0.0;

  // We pass int to kernel, so cast prime to int here (safe for all our primes)
  int prime_int = (int)prime;

  // ── Per-matrix loop ───────────────────────────────────────────────────────
  for (int fi = 0; fi < (int)files.size(); fi++) {
    printf("\n  [%d/%zu] %s\n", fi + 1, files.size(), files[fi].c_str());
    fflush(stdout);

    MatrixData md;
    try {
      md = parse_matrix(files[fi]);
    } catch (std::exception &ex) {
      printf("    [ERROR] %s — skipping\n", ex.what());
      fflush(stdout);
      continue;
    }
    if (md.n != n) {
      printf("    [WARN] n=%d != expected %d — skipping\n", md.n, n);
      fflush(stdout);
      continue;
    }

    double matrix_start_ms = now_ms();
    double minors_tested = 0.0;

    printf("    Searching all zero minors at dev=%d ...\n", dev);
    fflush(stdout);

    std::vector<ZeroMinor> found = search_matrix(
        gpu, md.data.data(), n, prime_int, dev, minors_tested, matrix_start_ms);

    double matrix_ms = now_ms() - matrix_start_ms;
    folder_zero_minors += (long long)found.size();
    folder_minors_total += minors_tested;

    // ── Terminal summary per matrix ───────────────────────────────────────
    printf("    Time: %.3f s | Tested: %.0f minors | Zero minors: %zu\n",
           matrix_ms / 1000.0, minors_tested, found.size());

    if (found.empty()) {
      printf("    → No zero minor at deviation %d\n", dev);
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

    // ── Write per-matrix result file ──────────────────────────────────────
    write_result_file(out_dir, md, prime, dev, found, matrix_ms, minors_tested);

    // ── Append to SUMMARY.txt — ALL matrices (zeros=0 included) ─────────
    // This matches the reference SUMMARY.txt format which shows every
    // matrix with its timing, count tested, and zero count regardless.
    if (sum_f) {
      fprintf(sum_f, "[%d/%d] %s\n", fi + 1, (int)files.size(),
              md.filename.c_str());
      fprintf(sum_f, "  time=%.4f s | tested=%.0f | zeros=%zu\n",
              matrix_ms / 1000.0, minors_tested, found.size());
      for (int mi = 0; mi < (int)found.size(); mi++) {
        const ZeroMinor &zm = found[mi];
        fprintf(sum_f, "  #%d  k=%d  s=%d  rows=[", mi + 1, zm.k, zm.s);
        for (int j = 0; j < zm.k; j++)
          fprintf(sum_f, "%d%s", zm.row_idx[j], j < zm.k - 1 ? "," : "");
        fprintf(sum_f, "]  cols=[");
        for (int j = 0; j < zm.k; j++)
          fprintf(sum_f, "%d%s", zm.col_idx[j], j < zm.k - 1 ? "," : "");
        fprintf(sum_f, "]  t=%.4fms\n", zm.time_ms);
        write_zero_minor_detail(sum_f, zm, md);
      }
      fprintf(sum_f, "\n");
    }
  } // end matrix loop

  double folder_ms = now_ms() - folder_start_ms;

  // ── Folder totals to terminal ─────────────────────────────────────────────
  printf("\n  ── dev=%d | %s complete ──\n", dev, folder_tag.c_str());
  printf("  Time           : %.3f s\n", folder_ms / 1000.0);
  printf("  Matrices       : %zu\n", files.size());
  printf("  Minors tested  : %.0f\n", folder_minors_total);
  printf("  Zero minors    : %lld\n", folder_zero_minors);
  fflush(stdout);

  // ── Folder totals to SUMMARY.txt ─────────────────────────────────────────
  if (sum_f) {
    fprintf(sum_f, "======================================================\n");
    fprintf(sum_f, "FOLDER TOTALS\n");
    fprintf(sum_f, "Total time     : %.4f s\n", folder_ms / 1000.0);
    fprintf(sum_f, "Matrices       : %zu\n", files.size());
    fprintf(sum_f, "Minors tested  : %.0f\n", folder_minors_total);
    fprintf(sum_f, "Zero minors    : %lld\n", folder_zero_minors);
    fclose(sum_f);
    sum_f = NULL;
    printf("  Summary -> %s\n", sum_path.c_str());
  }
  fflush(stdout);
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char **argv) {
  (void)argc;
  (void)argv; // no arguments needed — all hardcoded

  // ── Folder paths ──────────────────────────────────────────────────────────
  std::string dir_antioffset = "p_1048573_antioffset=0.2";
  std::string dir_offset = "p_1048573_offset=1";
  std::string dir_25_29 = "25_29";

  // Auto-detect p_1048573 folders if naming differs slightly
  {
    struct stat st;
    bool off_ok = (stat(dir_offset.c_str(), &st) == 0);
    bool anti_ok = (stat(dir_antioffset.c_str(), &st) == 0);
    if (!off_ok || !anti_ok) {
      DIR *d = opendir(".");
      if (d) {
        struct dirent *e;
        while ((e = readdir(d)) != NULL) {
          std::string nm = e->d_name;
          if (!off_ok && nm.find("antioffset") == std::string::npos &&
              (nm.find("offset=1") != std::string::npos ||
               nm.find("offset_1") != std::string::npos))
            dir_offset = nm;
          if (!anti_ok && nm.find("antioffset") != std::string::npos)
            dir_antioffset = nm;
        }
        closedir(d);
      }
    }
  }

  // ── Program header ────────────────────────────────────────────────────────
  printf("=================================================================\n");
  printf(" APM Exhaustive Search — Extended Multi-Prime Version\n");
  printf(" THREE-PHASE EXECUTION ORDER:\n");
  printf("   Phase 1: p_1048573 antioffset  (dev %d→%d)\n", MIN_DEV, MAX_DEV);
  printf("   Phase 2: 25_29/{25..29}        (dev %d→%d, per-folder prime)\n",
         MIN_DEV, MAX_DEV);
  printf("   Phase 3: p_1048573 offset      (dev %d→%d)\n", MIN_DEV, MAX_DEV);
  printf("=================================================================\n");
  printf(" PM block size     : %d (fixed)\n", PM_SIZE);
  printf(" Deviation levels  : %d -> %d\n", MIN_DEV, MAX_DEV);
  printf(" COL_CHUNK         : %d\n", COL_CHUNK);
  printf(" p_1048573 prime   : %lld\n", P_1048573);
  printf(" Antioffset folder : %s\n", dir_antioffset.c_str());
  printf(" Offset folder     : %s\n", dir_offset.c_str());
  printf(" 25_29 base dir    : %s\n", dir_25_29.c_str());
  printf("\n Folder→Prime mapping for 25_29/:\n");
  for (int i = 0; i < NUM_FOLDER_PRIMES; i++)
    printf("   %s/%d/ → prime = %lld\n", dir_25_29.c_str(),
           FOLDER_PRIMES[i].folder_id, FOLDER_PRIMES[i].prime);
  printf(
      "=================================================================\n\n");
  fflush(stdout);

  double prog_start = now_ms();

  // =========================================================================
  // PHASE 1 — p_1048573 ANTIOFFSET
  //   12×12 matrices — completes quickly.
  //   Results go to: Results/deviation_<d>/antioffset/
  // =========================================================================
  {
    struct stat st;
    bool anti_ok = (stat(dir_antioffset.c_str(), &st) == 0);

    printf("\n");
    printf(
        "================================================================\n");
    printf("  PHASE 1 - p_1048573 ANTIOFFSET  (deviations %d -> %d)\n", MIN_DEV,
           MAX_DEV);
    printf("  Folder : %s\n", dir_antioffset.c_str());
    printf("  Prime  : %lld\n", P_1048573);
    printf(
        "================================================================\n");
    fflush(stdout);

    if (!anti_ok) {
      printf("  [SKIP] antioffset folder not found: %s\n",
             dir_antioffset.c_str());
    } else {
      double phase1_start = now_ms();
      for (int dev = MIN_DEV; dev <= MAX_DEV; dev++) {
        printf("\n  -- ANTIOFFSET  deviation=%d  minor size=%dx%d --\n", dev,
               PM_SIZE + dev, PM_SIZE + dev);
        fflush(stdout);
        run_folder(dir_antioffset, "antioffset", P_1048573, dev);
        printf("\n  [antioffset deviation_%d done]\n", dev);
        fflush(stdout);
      }
      double phase1_ms = now_ms() - phase1_start;
      printf("\n==============================================================="
             "=\n");
      printf("  PHASE 1 COMPLETE - p_1048573 ANTIOFFSET  (%.2f min)\n",
             phase1_ms / 60000.0);
      printf(
          "================================================================\n");
      fflush(stdout);
    }
  }

  // =========================================================================
  // PHASE 2 — 25_29/ SUBFOLDERS
  //   Each subfolder (25..29) has its own prime.
  //   Files ending in _RN.txt are skipped.
  //   Results go to: Results/deviation_<d>/25_29_<fid>/
  // =========================================================================
  {
    struct stat st;
    bool base_ok = (stat(dir_25_29.c_str(), &st) == 0);

    printf("\n");
    printf(
        "================================================================\n");
    printf("  PHASE 2 - 25_29/ SUBFOLDERS  (deviations %d -> %d)\n", MIN_DEV,
           MAX_DEV);
    printf("  Base dir : %s\n", dir_25_29.c_str());
    printf(
        "================================================================\n");
    fflush(stdout);

    if (!base_ok) {
      printf("  [SKIP] 25_29/ directory not found: %s\n", dir_25_29.c_str());
    } else {
      double phase2_start = now_ms();
      for (int fi = 0; fi < NUM_FOLDER_PRIMES; fi++) {
        int fid = FOLDER_PRIMES[fi].folder_id;
        long long fprime = FOLDER_PRIMES[fi].prime;
        std::string sub_dir = dir_25_29 + "/" + std::to_string(fid);
        std::string tag = "25_29_" + std::to_string(fid);

        printf("\n");
        printf("  ┌──────────────────────────────────────────────────────┐\n");
        printf("  │  25_29 subfolder: %d   prime: %lld\n", fid, fprime);
        printf("  │  Path: %s\n", sub_dir.c_str());
        printf("  └──────────────────────────────────────────────────────┘\n");
        fflush(stdout);

        struct stat sub_st;
        if (stat(sub_dir.c_str(), &sub_st) != 0) {
          printf("  [SKIP] Subfolder not found: %s\n", sub_dir.c_str());
          continue;
        }

        for (int dev = MIN_DEV; dev <= MAX_DEV; dev++) {
          printf("\n  -- 25_29/%d  deviation=%d  minor size=%dx%d --\n", fid,
                 dev, PM_SIZE + dev, PM_SIZE + dev);
          fflush(stdout);
          run_folder(sub_dir, tag, fprime, dev);
          printf("\n  [25_29_%d deviation_%d done]\n", fid, dev);
          fflush(stdout);
        }
      }
      double phase2_ms = now_ms() - phase2_start;
      printf("\n==============================================================="
             "=\n");
      printf("  PHASE 2 COMPLETE - 25_29/ SUBFOLDERS  (%.2f min)\n",
             phase2_ms / 60000.0);
      printf(
          "================================================================\n");
      fflush(stdout);
    }
  }

  // =========================================================================
  // PHASE 3 — p_1048573 OFFSET
  //   Larger matrices — takes longer.
  //   Results go to: Results/deviation_<d>/offset/
  // =========================================================================
  {
    struct stat st;
    bool off_ok = (stat(dir_offset.c_str(), &st) == 0);

    printf("\n");
    printf(
        "================================================================\n");
    printf("  PHASE 3 - p_1048573 OFFSET  (deviations %d -> %d)\n", MIN_DEV,
           MAX_DEV);
    printf("  Folder : %s\n", dir_offset.c_str());
    printf("  Prime  : %lld\n", P_1048573);
    printf(
        "================================================================\n");
    fflush(stdout);

    if (!off_ok) {
      printf("  [SKIP] offset folder not found: %s\n", dir_offset.c_str());
    } else {
      double phase3_start = now_ms();
      for (int dev = MIN_DEV; dev <= MAX_DEV; dev++) {
        printf("\n  -- OFFSET  deviation=%d  minor size=%dx%d --\n", dev,
               PM_SIZE + dev, PM_SIZE + dev);
        fflush(stdout);
        run_folder(dir_offset, "offset", P_1048573, dev);
        printf("\n  [offset deviation_%d done]\n", dev);
        fflush(stdout);
      }
      double phase3_ms = now_ms() - phase3_start;
      printf("\n==============================================================="
             "=\n");
      printf("  PHASE 3 COMPLETE - p_1048573 OFFSET  (%.2f min)\n",
             phase3_ms / 60000.0);
      printf(
          "================================================================\n");
      fflush(stdout);
    }
  }

  double total_ms = now_ms() - prog_start;
  printf(
      "\n=================================================================\n");
  printf(" ALL DONE\n");
  printf(" Total wall time : %.3f s  (%.2f min)\n", total_ms / 1000.0,
         total_ms / 60000.0);
  printf(" Results layout  :\n");
  for (int d = MIN_DEV; d <= MAX_DEV; d++) {
    printf("   Results/deviation_%d/antioffset/\n", d);
    for (int i = 0; i < NUM_FOLDER_PRIMES; i++)
      printf("   Results/deviation_%d/25_29_%d/\n", d,
             FOLDER_PRIMES[i].folder_id);
    printf("   Results/deviation_%d/offset/\n", d);
  }
  printf("=================================================================\n");

  return 0;
}

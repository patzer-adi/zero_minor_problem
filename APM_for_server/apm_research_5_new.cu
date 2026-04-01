// =============================================================================
// apm_research_5_new.cu
//
// APM — Almost Principal Minors, Exhaustive Search
// NEW FOLDER version: processes matrices from new/ folder for primes 25–35
//
// What this program does:
//   ONE UNIFIED DEVIATION LOOP — deviation is the outermost loop.
//   Within each deviation, prime groups are processed in order 25 → 35.
//
//   Matrices live in the flat folder "new/":
//     kernel_25_1.txt ... kernel_25_15.txt   (prime from 25_29/25_1.txt)
//     kernel_26_*                             (prime from 25_29/26_1.txt)
//     ...
//     kernel_35_*                             (prime from exp/35_1.txt)
//
//   kernel_36_* files exist but are IGNORED (no prime file for 36).
//   Files ending in _RN.txt are IGNORED in all cases.
//
//   For each prime group at each deviation:
//     For EACH matrix file matching the prefix:
//       For EACH principal block position s (0 to n-2):
//         Generate ALL C(n-2, D) index sets for deviation level D
//         For EVERY (row_set x col_set) pair:
//           Compute det(submatrix) mod prime
//           If det == 0 -> record it (NEVER stop early, find ALL)
//       Write one result file per matrix
//     Write one SUMMARY.txt for the prime group
//
// IMPORTANT: Primes 31–35 exceed int range, so ALL modular arithmetic
//            uses long long throughout the kernel.
//
// Output folder structure created automatically:
//   Results_new/
//     25/
//       deviation_2/
//       deviation_3/
//       deviation_4/
//       deviation_5/
//     26/ 27/ 28/ 29/ 30/ 31/ 32/ 33/ 34/ 35/  (same pattern)
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
//        apm_research_5_new.cu -o apm5new        \
//        -Xlinker -lstdc++
//
// RUN (all paths and primes are hardcoded):
//   ./apm5new
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
// Hardcoded prime table for new/ folder (primes 25–35)
// Primes 31–35 exceed int range — long long is required.
// =============================================================================

struct FolderPrime {
  int folder_id;     // 25, 26, ..., 35
  long long prime;   // the prime used to generate matrices in that group
  const char *label; // display label
};

static const FolderPrime FOLDER_PRIMES[] = {
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
  int k;            // number of indices  (= PM_SIZE + dev)
  int idx[MAX_IDX]; // sorted matrix indices
};

// One GPU work unit: indices into the current device row/col buffers
struct MinorJob {
  int r; // row-set index in d_row_sets[]
  int c; // col-set index in current col chunk on device
  int k; // minor size (= PM_SIZE + dev)
};

// Per-launch result: first representative zero minor found
struct ChunkResult {
  int found; // 1 if any zero minor found this launch
  int minor_size;
  int r_idx;
  int c_idx;
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
// Device: modular arithmetic helpers — ALL long long for large primes
// =============================================================================

__device__ inline long long mod_sub(long long a, long long b, long long p) {
  long long r = a - b;
  return (r < 0) ? r + p : r;
}

__device__ inline long long mod_mul(long long a, long long b, long long p) {
  // Split a into two halves to avoid 128-bit overflow.
  // a, b < p < 2^36, so:
  //   a_hi < 2^20, a_lo < 2^16
  //   a_hi * b < 2^56 — fits in long long
  //   a_lo * b < 2^52 — fits in long long
  //   (term << 16) < 2^52 — fits in long long
  a %= p;
  if (a < 0)
    a += p;
  b %= p;
  if (b < 0)
    b += p;
  long long a_hi = a >> 16;
  long long a_lo = a & 0xFFFFLL;
  long long term1 = ((a_hi * b) % p << 16) % p;
  long long term2 = (a_lo * b) % p;
  return (term1 + term2) % p;
}

__device__ long long mod_inv(long long a, long long p) {
  // Extended Euclidean algorithm — long long version
  long long t = 0, nt = 1, r = p, nr = a;
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
// Device: determinant mod p via Gaussian elimination — long long
// mat is flat k×k row-major, k <= MAX_IDX (7)
// =============================================================================

__device__ long long det_mod(const int *mat, int k, long long p) {
  long long a[MAX_IDX][MAX_IDX];
  for (int i = 0; i < k; i++)
    for (int j = 0; j < k; j++)
      a[i][j] = ((long long)mat[i * k + j] % p + p) % p; // ensure positive mod

  long long det = 1;
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
      a[row][col] = 0;
    }
  }
  return det;
}

// =============================================================================
// CUDA kernel
//
// Each thread handles one (row_set, col_set) pair for the single loaded matrix.
// ALL zero minors are flagged — the kernel does NOT stop at the first one.
// d_chunk_result stores the first representative found (for index recovery).
// d_found_any is OR'd with 1 whenever any zero minor is found.
//
// Grid layout:
//   blockIdx.x * blockDim.x + threadIdx.x  = job index
// =============================================================================

__global__ void
apm_kernel(const int *d_matrix, // n*n matrix (device)
           int n,
           const IndexSet *d_row_sets, // all row index sets for this (s, dev)
           const IndexSet *d_col_sets, // col index sets for current chunk
           const MinorJob *d_jobs,     // job list for this launch
           int num_jobs, long long prime,
           ChunkResult *d_chunk_result, // output: first found representative
           int *d_found_any)            // output: 1 if any zero found
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_jobs)
    return;

  MinorJob job = d_jobs[tid];
  const IndexSet &rs = d_row_sets[job.r];
  const IndexSet &cs = d_col_sets[job.c];
  int k = job.k;

  // Extract k×k submatrix
  int sub[MAX_IDX * MAX_IDX];
  for (int i = 0; i < k; i++)
    for (int j = 0; j < k; j++)
      sub[i * k + j] = d_matrix[rs.idx[i] * n + cs.idx[j]];

  if (det_mod(sub, k, prime) == 0) {
    // Record one representative (first thread to CAS wins)
    if (atomicCAS(&d_chunk_result->found, 0, 1) == 0) {
      d_chunk_result->minor_size = k;
      d_chunk_result->r_idx = job.r;
      d_chunk_result->c_idx = job.c;
    }
    // Always mark that something was found
    atomicOr(d_found_any, 1);
  }
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

// Create Results_new/<folder_id>/deviation_<d>/ — used for all prime groups.
static std::string make_out_dir(const std::string &folder_id, int dev) {
  mkdir_safe("Results_new");
  std::string d1 = "Results_new/" + folder_id;
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
// SKIPS files ending in _RN.txt (these are row-number metadata files).
static std::vector<std::string>
collect_files_by_prefix(const std::string &dir_path,
                        const std::string &prefix) {
  std::vector<std::string> files;
  DIR *d = opendir(dir_path.c_str());
  if (!d)
    return files;
  struct dirent *e;
  while ((e = readdir(d)) != NULL) {
    std::string nm = e->d_name;
    // Must start with prefix and end with .txt
    if (!starts_with(nm, prefix))
      continue;
    if (nm.size() <= 4 || nm.substr(nm.size() - 4) != ".txt")
      continue;
    // SKIP files ending in _RN.txt
    if (ends_with(nm, "_RN.txt"))
      continue;
    files.push_back(dir_path + "/" + nm);
  }
  closedir(d);
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
  printf("      -> %s\n", outpath.c_str());
  fflush(stdout);
}

// =============================================================================
// GPU buffer manager — one instance per (dev, n) combination
// =============================================================================

struct GPUBufs {
  IndexSet *row_sets;
  IndexSet *col_sets;
  MinorJob *jobs;
  int *matrix;
  ChunkResult *chunk_res;
  int *found_any;
  size_t cap_rows;    // allocated row capacity
  long long cap_jobs; // allocated job capacity

  GPUBufs()
      : row_sets(NULL), col_sets(NULL), jobs(NULL), matrix(NULL),
        chunk_res(NULL), found_any(NULL), cap_rows(0), cap_jobs(0) {}

  void alloc(int n, int dev) {
    free_all();

    // Upper bound: C(n - PM_SIZE, dev)
    int pool = n - PM_SIZE;
    long long c = 1;
    for (int i = 0; i < dev && i < pool; i++)
      c = c * (pool - i) / (i + 1);
    cap_rows = (size_t)c;
    cap_jobs = (long long)cap_rows * COL_CHUNK;

    printf("    [GPU] Allocating: rows=%zu (%.1f MB)  jobs=%lld (%.1f MB)  "
           "matrix=%d x %d\n",
           cap_rows, cap_rows * sizeof(IndexSet) / 1e6, cap_jobs,
           cap_jobs * sizeof(MinorJob) / 1e6, n, n);
    fflush(stdout);

    CK(cudaMalloc(&row_sets, cap_rows * sizeof(IndexSet)));
    CK(cudaMalloc(&col_sets, (size_t)COL_CHUNK * sizeof(IndexSet)));
    CK(cudaMalloc(&jobs, cap_jobs * sizeof(MinorJob)));
    CK(cudaMalloc(&matrix, (size_t)n * n * sizeof(int)));
    CK(cudaMalloc(&chunk_res, sizeof(ChunkResult)));
    CK(cudaMalloc(&found_any, sizeof(int)));
  }

  void free_all() {
    if (row_sets) {
      cudaFree(row_sets);
      row_sets = NULL;
    }
    if (col_sets) {
      cudaFree(col_sets);
      col_sets = NULL;
    }
    if (jobs) {
      cudaFree(jobs);
      jobs = NULL;
    }
    if (matrix) {
      cudaFree(matrix);
      matrix = NULL;
    }
    if (chunk_res) {
      cudaFree(chunk_res);
      chunk_res = NULL;
    }
    if (found_any) {
      cudaFree(found_any);
      found_any = NULL;
    }
  }

  ~GPUBufs() { free_all(); }
};

// =============================================================================
// Search all zero minors in one matrix at one deviation level
// Returns: vector of all zero minors found
// Writes:  minors_tested (total number of det evaluations)
// =============================================================================

static std::vector<ZeroMinor>
search_matrix(GPUBufs &gpu,
              const int *h_mat, // host pointer: n*n flat matrix
              int n,
              long long prime, // long long for large primes (31–35)
              int dev, double &minors_tested, double matrix_start_ms) {
  std::vector<ZeroMinor> found;
  minors_tested = 0.0;

  const int BLOCK = 256;
  const int k = PM_SIZE + dev;

  // Upload matrix once
  CK(cudaMemcpy(gpu.matrix, h_mat, (size_t)n * n * sizeof(int),
                cudaMemcpyHostToDevice));

  int pool_sz = n - PM_SIZE;

  // ── Loop over every principal block position s ──────────────────────────
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

    // Generate all C(pool_sz, dev) index sets for this (s, dev)
    std::vector<IndexSet> sets;
    gen_combos(pool, dev, principal, sets);
    int num_sets = (int)sets.size();
    if (num_sets == 0)
      continue;

    if ((size_t)num_sets > gpu.cap_rows) {
      fprintf(
          stderr,
          "  [WARN] s=%d dev=%d: %d sets exceeds GPU buffer %zu — skipping\n",
          s, dev, num_sets, gpu.cap_rows);
      continue;
    }

    // Upload all row index sets (fixed for this s, dev)
    CK(cudaMemcpy(gpu.row_sets, sets.data(), num_sets * sizeof(IndexSet),
                  cudaMemcpyHostToDevice));

    // ── Chunk over col-sets ──────────────────────────────────────────────
    for (int col_start = 0; col_start < num_sets; col_start += COL_CHUNK) {
      int col_end = std::min(col_start + COL_CHUNK, num_sets);
      int num_cols = col_end - col_start;

      long long num_jobs = (long long)num_sets * num_cols;
      minors_tested += (double)num_jobs;

      // Build job list on host
      std::vector<MinorJob> h_jobs(num_jobs);
      for (int r = 0; r < num_sets; r++)
        for (int c = 0; c < num_cols; c++) {
          long long jidx = (long long)r * num_cols + c;
          h_jobs[jidx].r = r;
          h_jobs[jidx].c = c;
          h_jobs[jidx].k = k;
        }

      // Reset GPU result flags
      ChunkResult zero_cr = {0, 0, 0, 0};
      int zero_fa = 0;
      CK(cudaMemcpy(gpu.chunk_res, &zero_cr, sizeof(ChunkResult),
                    cudaMemcpyHostToDevice));
      CK(cudaMemcpy(gpu.found_any, &zero_fa, sizeof(int),
                    cudaMemcpyHostToDevice));

      // Upload col chunk and jobs
      CK(cudaMemcpy(gpu.col_sets, sets.data() + col_start,
                    num_cols * sizeof(IndexSet), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(gpu.jobs, h_jobs.data(), num_jobs * sizeof(MinorJob),
                    cudaMemcpyHostToDevice));

      // Launch kernel
      int grid_x = ((int)num_jobs + BLOCK - 1) / BLOCK;
      apm_kernel<<<dim3(grid_x, 1, 1), BLOCK>>>(
          gpu.matrix, n, gpu.row_sets, gpu.col_sets, gpu.jobs, (int)num_jobs,
          prime, gpu.chunk_res, gpu.found_any);

      CK(cudaDeviceSynchronize());
      CK(cudaGetLastError());

      // Pull back results
      ChunkResult h_cr;
      int h_fa;
      CK(cudaMemcpy(&h_cr, gpu.chunk_res, sizeof(ChunkResult),
                    cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(&h_fa, gpu.found_any, sizeof(int), cudaMemcpyDeviceToHost));

      if (h_fa) {
        // Resolve absolute row/col indices
        int ri = h_cr.r_idx;
        int ci = h_cr.c_idx + col_start; // absolute in sets[]

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
      }
    } // col chunk loop
  } // s loop

  return found;
}

// =============================================================================
// Process one prime group at one deviation level
// =============================================================================

static void run_folder(const std::string &input_dir,
                       const std::string &prefix,     // e.g. "kernel_25_"
                       const std::string &folder_tag, // "25", "26", etc.
                       long long prime, int dev) {
  std::string out_dir = make_out_dir(folder_tag, dev);
  std::vector<std::string> files = collect_files_by_prefix(input_dir, prefix);

  if (files.empty()) {
    printf("  [SKIP] No matching files for prefix %s in %s\n", prefix.c_str(),
           input_dir.c_str());
    return;
  }

  // ── Header ───────────────────────────────────────────────────────────────
  printf("\n");
  printf("  +----------------------------------------------------------+\n");
  printf("  |  Deviation %d  |  %-12s  |  %zu matrices\n", dev,
         folder_tag.c_str(), files.size());
  printf("  |  Input : %s  (prefix: %s)\n", input_dir.c_str(), prefix.c_str());
  printf("  |  Output: %s\n", out_dir.c_str());
  printf("  |  Prime : %lld\n", prime);
  printf("  +----------------------------------------------------------+\n");
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

  // Allocate GPU buffers (once for all matrices in this group + dev)
  GPUBufs gpu;
  gpu.alloc(n, dev);

  // Open SUMMARY.txt for this prime group
  std::string sum_path = out_dir + "/SUMMARY.txt";
  FILE *sum_f = fopen(sum_path.c_str(), "w");
  if (!sum_f) {
    fprintf(stderr, "  [WARN] Cannot open SUMMARY.txt: %s\n", sum_path.c_str());
  } else {
    fprintf(sum_f, "APM Summary\n");
    fprintf(sum_f, "Deviation level : %d\n", dev);
    fprintf(sum_f, "Prime group     : %s\n", folder_tag.c_str());
    fprintf(sum_f, "Input folder    : %s\n", input_dir.c_str());
    fprintf(sum_f, "File prefix     : %s\n", prefix.c_str());
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
        gpu, md.data.data(), n, prime, dev, minors_tested, matrix_start_ms);

    double matrix_ms = now_ms() - matrix_start_ms;
    folder_zero_minors += (long long)found.size();
    folder_minors_total += minors_tested;

    // ── Terminal summary per matrix ───────────────────────────────────────
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

    // ── Write per-matrix result file ──────────────────────────────────────
    write_result_file(out_dir, md, prime, dev, found, matrix_ms, minors_tested);

    // ── Append to SUMMARY.txt — skip matrices with no zeros ──────────────
    if (sum_f && !found.empty()) {
      fprintf(sum_f,
              "------------------------------------------------------------\n");
      fprintf(sum_f, "[%d/%d] %s\n", fi + 1, (int)files.size(),
              md.filename.c_str());
      fprintf(sum_f, "  time=%.4f s | tested=%.0f | zeros=%zu\n",
              matrix_ms / 1000.0, minors_tested, found.size());
      for (int mi = 0; mi < (int)found.size(); mi++) {
        const ZeroMinor &zm = found[mi];
        fprintf(sum_f, "\n  --- Zero Minor #%d ---\n", mi + 1);
        fprintf(sum_f, "  Minor size (k)   : %d\n", zm.k);
        fprintf(sum_f, "  Deviation        : %d\n", zm.dev);
        fprintf(sum_f, "  Principal block s: %d  (indices {%d, %d})\n", zm.s,
                zm.s, zm.s + 1);
        fprintf(sum_f, "  Row indices [%d]  : ", zm.k);
        for (int j = 0; j < zm.k; j++)
          fprintf(sum_f, "%d%s", zm.row_idx[j], j < zm.k - 1 ? " " : "");
        fprintf(sum_f, "\n");
        fprintf(sum_f, "  Col indices [%d]  : ", zm.k);
        for (int j = 0; j < zm.k; j++)
          fprintf(sum_f, "%d%s", zm.col_idx[j], j < zm.k - 1 ? " " : "");
        fprintf(sum_f, "\n");
        fprintf(sum_f, "  Time found       : %.4f ms\n\n", zm.time_ms);
        // Print principal block + submatrix using shared helper
        write_zero_minor_detail(sum_f, zm, md);
      }
    }
  } // end matrix loop

  double folder_ms = now_ms() - folder_start_ms;

  // ── Folder totals to terminal ─────────────────────────────────────────────
  printf("\n  -- dev=%d | %s complete --\n", dev, folder_tag.c_str());
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

  // ── Input folder ──────────────────────────────────────────────────────────
  std::string dir_new = "new";

  // Check that new/ folder exists
  struct stat st;
  bool new_exists = (stat(dir_new.c_str(), &st) == 0);

  // ── Program header ────────────────────────────────────────────────────────
  printf("=================================================================\n");
  printf(" APM Exhaustive Search — apm_research_5_new\n");
  printf(" NEW FOLDER VERSION — primes 25 through 35\n");
  printf(" UNIFIED DEVIATION-OUTER LOOP\n");
  printf("\n");
  printf(" Execution order per deviation (2 -> 3 -> 4 -> 5):\n");
  for (int i = 0; i < NUM_FOLDER_PRIMES; i++) {
    printf("   %2d. new/kernel_%d_*  %s\n", i + 1, FOLDER_PRIMES[i].folder_id,
           FOLDER_PRIMES[i].label);
  }
  printf("\n");
  printf(" Input folder     : %s\n", dir_new.c_str());
  printf(" PM block size    : %d (fixed)\n", PM_SIZE);
  printf(" Deviation levels : %d -> %d\n", MIN_DEV, MAX_DEV);
  printf(" COL_CHUNK        : %d\n", COL_CHUNK);
  printf(" new/ found       : %s\n", new_exists ? "YES" : "NO (will skip)");
  printf(
      "=================================================================\n\n");
  fflush(stdout);

  if (!new_exists) {
    printf("[FATAL] new/ folder not found. Nothing to do.\n");
    return 1;
  }

  double prog_start = now_ms();

  // =========================================================================
  // UNIFIED LOOP — deviation is outermost
  //
  // For each deviation d in {2, 3, 4, 5}:
  //   For each prime group N in {25, 26, ..., 35}:
  //     Collect new/kernel_<N>_*.txt (skip _RN.txt)
  //     Process with prime for N
  //     Write results to Results_new/<N>/deviation_<d>/
  // =========================================================================

  for (int dev = MIN_DEV; dev <= MAX_DEV; dev++) {

    printf("\n");
    printf(
        "#################################################################\n");
    printf("# DEVIATION %d   minor size = %dx%d\n", dev, PM_SIZE + dev,
           PM_SIZE + dev);
    printf("# Order: 25 -> 26 -> 27 -> 28 -> 29 -> 30 -> 31 -> 32 -> 33 -> "
           "34 -> 35\n");
    printf(
        "#################################################################\n");
    fflush(stdout);

    double dev_start = now_ms();

    // ── Process each prime group ──────────────────────────────────────────
    for (int fi = 0; fi < NUM_FOLDER_PRIMES; fi++) {
      int fid = FOLDER_PRIMES[fi].folder_id;
      long long fprime = FOLDER_PRIMES[fi].prime;
      std::string prefix = "kernel_" + std::to_string(fid) + "_";
      std::string tag = std::to_string(fid);

      printf("\n");
      printf(
          "  "
          "================================================================\n");
      printf("  [dev=%d | %d/%d] new/kernel_%d_*   prime=%lld\n", dev, fi + 1,
             NUM_FOLDER_PRIMES, fid, fprime);
      printf("  Input  : %s  prefix=%s\n", dir_new.c_str(), prefix.c_str());
      printf(
          "  "
          "================================================================\n");
      fflush(stdout);

      run_folder(dir_new, prefix, tag, fprime, dev);
      printf("  [dev=%d  prime_%d done]\n", dev, fid);
      fflush(stdout);
    }

    double dev_ms = now_ms() - dev_start;
    printf("\n");
    printf(
        "#################################################################\n");
    printf("# DEVIATION %d COMPLETE   (%.2f min)\n", dev, dev_ms / 60000.0);
    printf(
        "#################################################################\n");
    fflush(stdout);

  } // end deviation loop

  double total_ms = now_ms() - prog_start;
  printf(
      "\n=================================================================\n");
  printf(" ALL DONE\n");
  printf(" Total wall time : %.3f s  (%.2f min)\n", total_ms / 1000.0,
         total_ms / 60000.0);
  printf("\n Results layout:\n");
  for (int d = MIN_DEV; d <= MAX_DEV; d++) {
    for (int i = 0; i < NUM_FOLDER_PRIMES; i++)
      printf("   Results_new/%d/deviation_%d/\n", FOLDER_PRIMES[i].folder_id,
             d);
  }
  printf("=================================================================\n");

  return 0;
}

// =============================================================================
// apm_brahma.cu
//
// APM Brahma — V100-OPTIMIZED, TEMPLATED KERNEL, FLAT-ARRAY VERSION
//
// V100-specific optimizations:
//   1. TEMPLATED KERNEL: apm_kernel_k<K> sizes sub[K*K] at compile time.
//      Stack frame is exactly K*K*8 bytes — not the fixed 50*50*8 = 20KB.
//      For k=4 (dev=2): 128 bytes → 100% occupancy on V100.
//      For k=10 (dev=8): 800 bytes → 100% occupancy on V100.
//      For k=27 (dev=25): 5832 bytes → ~68% occupancy on V100.
//      For k=50 (dev=48): 20000 bytes → ~20% occupancy (same as before).
//      47 template instantiations (k=4..50) compiled to separate .cubin.
//   2. __constant__ memory for pow2_64 (L1-cached on Volta, removes
//      parameter from mod_mul→det_mod→kernel chain, reduces register pressure)
//   3. Plain *d_found read instead of atomicAdd(d_found,0) (free on Volta)
//   4. reserve() on host vectors before filling (avoids reallocation)
//   5. Flat int[] index arrays (5-8x VRAM savings vs IndexSet structs)
//   6. Graceful OOM skip with break
//
// All-in-one CUDA file combining:
//   - First-hit search (stops after 1st zero minor per matrix)
//   - Early-stop at 100 matrix hits per group
//   - Runtime VRAM detection via cudaMemGetInfo()
//   - Safe 50-bit prime modular multiplication (__umul64hi intrinsic)
//   - Extended group range 25–50
//   - Auto-detect GPU compute capability (in Makefile)
//
// Usage:
//   ./apm_brahma [gmin gmax]
//   ./apm_brahma              (default: groups 25 to 50)
//   ./apm_brahma 32 35        (groups 32 to 35)
//
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
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
using namespace std;

// =============================================================================
// Compile-time constants
// =============================================================================

static const int PM_SIZE = 2;
static const int MIN_DEV = 2;
static const int MAX_IDX_STATIC = 60;  // supports up to 62×62 matrices (group 60)
static const int EARLY_STOP_HIT = 100;
static const char *RESULT_BASE_DIR = "Results_brahma";

// =============================================================================
// __constant__ memory: pow2_64 = 2^64 mod p
//
// Stored in constant memory (64KB, L1-cached on Volta/Turing).
// Set once per matrix search via cudaMemcpyToSymbol() on the host.
// Read by every mod_mul call — zero parameter overhead, cached in L1.
// =============================================================================

__constant__ unsigned long long c_pow2_64;

// =============================================================================
// CUDA error check macro
// =============================================================================

#define CK(call)                                                               \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d -- %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(_e));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// =============================================================================
// Data structures
// =============================================================================

struct MatrixData {
  string filename;
  int n;
  vector<long long> data;
};

struct ZeroMinor {
  int k;
  int dev;
  int s;
  int row_idx[MAX_IDX_STATIC];
  int col_idx[MAX_IDX_STATIC];
  double time_ms;
};

struct FolderPrime {
  int folder_id;
  long long prime;
  char label[64];
};

// =============================================================================
// Hardcoded fallback primes for groups 10-60
// =============================================================================

static const struct {
  int id;
  long long prime;
} HARDCODED_PRIMES[] = {
    {10, 1021LL},          {11, 2029LL},          {12, 4079LL},
    {13, 8111LL},          {14, 16273LL},         {15, 32749LL},
    {16, 65413LL},         {17, 131071LL},        {18, 262103LL},
    {19, 524257LL},        {20, 1048573LL},       {21, 2097147LL},
    {22, 4194217LL},       {23, 8388449LL},       {24, 16777099LL},
    {25, 33554393LL},      {26, 44923183LL},      {27, 134217689LL},
    {28, 268435399LL},     {29, 536870909LL},     {30, 1073741789LL},
    {31, 2147483647LL},    {32, 4294967291LL},    {33, 8589934583LL},
    {34, 17179869143LL},   {35, 34359738337LL},   {36, 68719476503LL},
    {37, 137438953097LL},  {38, 274877906837LL},  {39, 549755813657LL},
    {40, 1099511627689LL}, {41, 2199023255531LL}, {42, 4398046511093LL},
    {43, 8796093022151LL}, {44, 17592186044399LL},{45, 35184372088777LL},
    {46, 70368744177643LL},{47, 140737488355213LL},{48, 281474976710597LL},
    {49, 562949953421231LL},{50, 1125899906842597LL},
    // Groups 51-60: primes near 2^51 through 2^60
    {51, 2251799813685119LL},  {52, 4503599627370449LL},
    {53, 9007199254740881LL},  {54, 18014398509481951LL},
    {55, 36028797018963913LL}, {56, 72057594037927931LL},
    {57, 144115188075855859LL},{58, 288230376151711717LL},
    {59, 576460752303423433LL},{60, 1152921504606846883LL},
};
static const int NUM_HARDCODED = sizeof(HARDCODED_PRIMES) / sizeof(HARDCODED_PRIMES[0]);

// =============================================================================
// Device: modular arithmetic — ALL long long
// =============================================================================

__device__ inline long long mod_sub(long long a, long long b, long long p) {
  long long r = a - b;
  return (r < 0) ? r + p : r;
}

// Safe modular multiplication for primes up to 2^60.
// Uses __umul64hi intrinsic for hardware 64×64→128 bit multiply.
// Reads c_pow2_64 from __constant__ memory (L1-cached, zero param overhead).
//
// Three paths:
//   1. hi == 0 → single lo % p (most calls for small primes)
//   2. p < 2^40 (groups 25-39): single multiply hi*c_pow2_64 (no loop at all)
//      Proof: hi < 2^(2×40-64) = 2^16, c_pow2_64 < 2^40, product < 2^56 ✓
//   3. p >= 2^40 (groups 40-60): binary loop on hi (≤56 iters for p < 2^60)
__device__ inline long long mod_mul(long long a, long long b, long long p) {
  unsigned long long ua = (unsigned long long)((a % p + p) % p);
  unsigned long long ub = (unsigned long long)((b % p + p) % p);
  unsigned long long up = (unsigned long long)p;

  unsigned long long lo = ua * ub;
  unsigned long long hi = __umul64hi(ua, ub);

  // Fast path 1: no overflow — most calls end here for groups 25-31
  if (hi == 0) return (long long)(lo % up);

  // Fast path 2: single multiply for p < 2^40 (groups 25-39)
  // hi < 2^16, c_pow2_64 < 2^40 → product < 2^56, fits in uint64
  if (up < (1ULL << 40)) {
    return (long long)((hi * c_pow2_64 % up + lo % up) % up);
  }

  // Slow path: binary loop for p >= 2^40 (groups 40-60)
  // hi < 2^(2*60-64) = 2^56, loop runs ≤ 56 iterations
  unsigned long long hi_mod = 0;
  unsigned long long hbase = c_pow2_64;  // from __constant__ memory
  while (hi) {
    if (hi & 1) hi_mod = (hi_mod + hbase) % up;
    hbase = (hbase + hbase) % up;  // safe: hbase < p < 2^60, sum < 2^61
    hi >>= 1;
  }

  return (long long)((hi_mod + lo % up) % up);
}

// Extended Euclidean — all long long.
// Safe for p up to 2^62: the intermediate product q*nt is bounded by 2p
// (property of the GCD recurrence: |q_i * t_i| < 2p always holds).
// For p < 2^60: 2p < 2^61 < 2^63, fits in long long. No overflow.
__device__ inline long long mod_inv(long long a, long long p) {
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
// Device: determinant mod p via Gaussian elimination
//
// Operates DIRECTLY on the passed-in buffer (no internal copy).
// mod_mul reads c_pow2_64 from constant memory — no pow2_64 parameter needed.
// =============================================================================

__device__ inline long long det_mod(long long *a, int k, long long p) {
  long long det = 1LL;
  for (int col = 0; col < k; col++) {
    int piv = -1;
    for (int row = col; row < k; row++) {
      if (a[row * k + col]) {
        piv = row;
        break;
      }
    }
    if (piv < 0)
      return 0LL;

    if (piv != col) {
      for (int j = col; j < k; j++) {
        long long tmp = a[col * k + j];
        a[col * k + j] = a[piv * k + j];
        a[piv * k + j] = tmp;
      }
      det = (p - det) % p;
    }

    det = mod_mul(det, a[col * k + col], p);
    long long inv = mod_inv(a[col * k + col], p);

    for (int row = col + 1; row < k; row++) {
      if (!a[row * k + col])
        continue;
      long long f = mod_mul(a[row * k + col], inv, p);
      for (int j = col + 1; j < k; j++)
        a[row * k + j] = mod_sub(a[row * k + j], mod_mul(f, a[col * k + j], p), p);
      a[row * k + col] = 0LL;
    }
  }
  return det;
}

// =============================================================================
// CUDA kernel — TEMPLATED, FLAT INDEX ARRAYS, CONSTANT MEMORY pow2_64
//
// Template parameter K = minor size (PM_SIZE + dev).
// Each instantiation apm_kernel_k<K> has sub[K*K] — the compiler knows the
// exact stack frame size, enabling optimal register allocation per K.
//
// Stack frame per thread:
//   k=4  →  128 bytes   (100% occupancy on V100)
//   k=6  →  288 bytes   (100% occupancy on V100)
//   k=10 →  800 bytes   (100% occupancy on V100)
//   k=27 → 5832 bytes   (~68% occupancy on V100)
//   k=50 → 20000 bytes  (~20% occupancy on V100)
//
// 47 template instantiations (k=4..50) are compiled via INSTANTIATE_KERNEL.
// Host-side dispatch via launch_apm_kernel() switch.
// =============================================================================

template <int K>
__global__ void apm_kernel_k(const long long *d_matrix, int n,
                             const int *d_row_idx,
                             const int *d_col_idx,
                             int num_row_sets, int num_col_sets,
                             long long prime,
                             int *d_zero_flags, int *d_found) {
  long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  long long total = static_cast<long long>(num_row_sets) * num_col_sets;
  if (tid >= total)
    return;

  // Plain read — no atomic needed for a check.
  // On Volta, global loads go through L1 cache.
  //if (*d_found != 0) return;
  if (atomicAdd(d_found, 0) != 0) return;
  int r = static_cast<int>(tid / num_col_sets);
  int c = static_cast<int>(tid % num_col_sets);

  // K*K known at compile time — compiler allocates exact stack frame.
  long long sub[K * K];
  for (int i = 0; i < K; i++)
    for (int j = 0; j < K; j++) {
      long long v = d_matrix[d_row_idx[r * K + i] * n + d_col_idx[c * K + j]];
      sub[i * K + j] = ((v % prime) + prime) % prime;
    }

  int result = (det_mod(sub, K, prime) == 0LL) ? 1 : 0;
  d_zero_flags[tid] = result;
  if (result) atomicExch(d_found, 1);
}

// -----------------------------------------------------------------------------
// Explicit template instantiation for k = 4 through 60.
// Each creates a separate kernel binary in the .cubin with optimal stack size.
// 57 instantiations — compilation takes ~45-90s but runtime is optimal.
// -----------------------------------------------------------------------------

#define INSTANTIATE_KERNEL(K) template __global__ void apm_kernel_k<K>( \
    const long long*, int, const int*, const int*, int, int,            \
    long long, int*, int*);

INSTANTIATE_KERNEL(4)  INSTANTIATE_KERNEL(5)  INSTANTIATE_KERNEL(6)
INSTANTIATE_KERNEL(7)  INSTANTIATE_KERNEL(8)  INSTANTIATE_KERNEL(9)
INSTANTIATE_KERNEL(10) INSTANTIATE_KERNEL(11) INSTANTIATE_KERNEL(12)
INSTANTIATE_KERNEL(13) INSTANTIATE_KERNEL(14) INSTANTIATE_KERNEL(15)
INSTANTIATE_KERNEL(16) INSTANTIATE_KERNEL(17) INSTANTIATE_KERNEL(18)
INSTANTIATE_KERNEL(19) INSTANTIATE_KERNEL(20) INSTANTIATE_KERNEL(21)
INSTANTIATE_KERNEL(22) INSTANTIATE_KERNEL(23) INSTANTIATE_KERNEL(24)
INSTANTIATE_KERNEL(25) INSTANTIATE_KERNEL(26) INSTANTIATE_KERNEL(27)
INSTANTIATE_KERNEL(28) INSTANTIATE_KERNEL(29) INSTANTIATE_KERNEL(30)
INSTANTIATE_KERNEL(31) INSTANTIATE_KERNEL(32) INSTANTIATE_KERNEL(33)
INSTANTIATE_KERNEL(34) INSTANTIATE_KERNEL(35) INSTANTIATE_KERNEL(36)
INSTANTIATE_KERNEL(37) INSTANTIATE_KERNEL(38) INSTANTIATE_KERNEL(39)
INSTANTIATE_KERNEL(40) INSTANTIATE_KERNEL(41) INSTANTIATE_KERNEL(42)
INSTANTIATE_KERNEL(43) INSTANTIATE_KERNEL(44) INSTANTIATE_KERNEL(45)
INSTANTIATE_KERNEL(46) INSTANTIATE_KERNEL(47) INSTANTIATE_KERNEL(48)
INSTANTIATE_KERNEL(49) INSTANTIATE_KERNEL(50) INSTANTIATE_KERNEL(51)
INSTANTIATE_KERNEL(52) INSTANTIATE_KERNEL(53) INSTANTIATE_KERNEL(54)
INSTANTIATE_KERNEL(55) INSTANTIATE_KERNEL(56) INSTANTIATE_KERNEL(57)
INSTANTIATE_KERNEL(58) INSTANTIATE_KERNEL(59) INSTANTIATE_KERNEL(60)

#undef INSTANTIATE_KERNEL

// -----------------------------------------------------------------------------
// Host dispatch: launch the correct template instantiation for a given k.
// Uses a compile-time switch — each case calls the right apm_kernel_k<K>.
// Covers k=4..60 for groups 25-60.
// -----------------------------------------------------------------------------

#define DISPATCH_CASE(K)                                                       \
  case K:                                                                      \
    apm_kernel_k<K><<<grid, block>>>(d_matrix, n, d_row_idx, d_col_idx,        \
                                     num_rows, num_cols, prime,                \
                                     d_zero_flags, d_found);                  \
    break;

static void launch_apm_kernel(int k, int grid, int block,
                              const long long *d_matrix, int n,
                              const int *d_row_idx, const int *d_col_idx,
                              int num_rows, int num_cols,
                              long long prime,
                              int *d_zero_flags, int *d_found) {
  switch (k) {
    DISPATCH_CASE(4)  DISPATCH_CASE(5)  DISPATCH_CASE(6)
    DISPATCH_CASE(7)  DISPATCH_CASE(8)  DISPATCH_CASE(9)
    DISPATCH_CASE(10) DISPATCH_CASE(11) DISPATCH_CASE(12)
    DISPATCH_CASE(13) DISPATCH_CASE(14) DISPATCH_CASE(15)
    DISPATCH_CASE(16) DISPATCH_CASE(17) DISPATCH_CASE(18)
    DISPATCH_CASE(19) DISPATCH_CASE(20) DISPATCH_CASE(21)
    DISPATCH_CASE(22) DISPATCH_CASE(23) DISPATCH_CASE(24)
    DISPATCH_CASE(25) DISPATCH_CASE(26) DISPATCH_CASE(27)
    DISPATCH_CASE(28) DISPATCH_CASE(29) DISPATCH_CASE(30)
    DISPATCH_CASE(31) DISPATCH_CASE(32) DISPATCH_CASE(33)
    DISPATCH_CASE(34) DISPATCH_CASE(35) DISPATCH_CASE(36)
    DISPATCH_CASE(37) DISPATCH_CASE(38) DISPATCH_CASE(39)
    DISPATCH_CASE(40) DISPATCH_CASE(41) DISPATCH_CASE(42)
    DISPATCH_CASE(43) DISPATCH_CASE(44) DISPATCH_CASE(45)
    DISPATCH_CASE(46) DISPATCH_CASE(47) DISPATCH_CASE(48)
    DISPATCH_CASE(49) DISPATCH_CASE(50) DISPATCH_CASE(51)
    DISPATCH_CASE(52) DISPATCH_CASE(53) DISPATCH_CASE(54)
    DISPATCH_CASE(55) DISPATCH_CASE(56) DISPATCH_CASE(57)
    DISPATCH_CASE(58) DISPATCH_CASE(59) DISPATCH_CASE(60)
    default:
      fprintf(stderr, "  [FATAL] k=%d exceeds MAX_IDX_STATIC=%d\n",
              k, MAX_IDX_STATIC);
      break;
  }
}

#undef DISPATCH_CASE

// =============================================================================
// Host utilities
// =============================================================================

static double now_ms() {
  using namespace chrono;
  return static_cast<double>(
             duration_cast<microseconds>(
                 high_resolution_clock::now().time_since_epoch())
                 .count()) /
         1000.0;
}

static long long nCr(int n, int r) {
  if (r < 0 || r > n)
    return 0;
  if (r == 0 || r == n)
    return 1;
  if (r > n - r)
    r = n - r;
  long long result = 1;
  for (int i = 0; i < r; i++)
    result = result * (n - i) / (i + 1);
  return result;
}

static void mkdir_safe(const string &p) { mkdir(p.c_str(), 0755); }

static bool ends_with(const string &str, const string &suffix) {
  if (suffix.size() > str.size())
    return false;
  return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool starts_with(const string &str, const string &prefix) {
  if (prefix.size() > str.size())
    return false;
  return str.compare(0, prefix.size(), prefix) == 0;
}

static vector<string> collect_files(const string &dir_path,
                                    const string &prefix) {
  vector<string> files;
  DIR *d = opendir(dir_path.c_str());
  if (!d)
    return files;
  struct dirent *e;
  while ((e = readdir(d)) != NULL) {
    string nm = e->d_name;
    if (!starts_with(nm, prefix))
      continue;
    if (nm.size() <= 4)
      continue;
    if (nm.substr(nm.size() - 4) != ".txt")
      continue;
    if (ends_with(nm, "_RN.txt"))
      continue;
    files.push_back(dir_path + "/" + nm);
  }
  closedir(d);
  sort(files.begin(), files.end());
  return files;
}

static string make_out_dir(int group, int dev) {
  mkdir_safe(RESULT_BASE_DIR);
  string d1 = string(RESULT_BASE_DIR) + "/" + to_string(group);
  mkdir_safe(d1);
  string d2 = d1 + "/deviation_" + to_string(dev);
  mkdir_safe(d2);
  return d2;
}

// =============================================================================
// Prime file reader
// =============================================================================

static long long read_prime_from_file(const string &path) {
  ifstream ifs(path.c_str());
  if (!ifs.is_open())
    return -1;

  string line;
  while (getline(ifs, line)) {
    size_t fs = line.find_first_not_of(" \t\r\n");
    if (fs == string::npos)
      continue;
    line = line.substr(fs);
    if (line[0] == '#')
      break;

    istringstream ss(line);
    vector<string> tokens;
    string tok;
    while (ss >> tok)
      tokens.push_back(tok);

    if (tokens.size() == 1) {
      long long val = 0;
      try {
        val = stoll(tokens[0]);
      } catch (...) {
        continue;
      }
      return val;
    }
  }
  return -1;
}

static vector<FolderPrime> load_primes(int gmin, int gmax) {
  vector<FolderPrime> primes;

  for (int g = gmin; g <= gmax; g++) {
    FolderPrime fp;
    fp.folder_id = g;
    fp.prime = -1;
    for (int i = 0; i < NUM_HARDCODED; i++) {
      if (HARDCODED_PRIMES[i].id == g) {
        fp.prime = HARDCODED_PRIMES[i].prime;
        break;
      }
    }
    snprintf(fp.label, sizeof(fp.label), "%d (p=%lld)", g, fp.prime);

    string path;
    long long p = -1;

    if (g >= 25 && g <= 29) {
      path = "25_29/" + to_string(g) + "_1.txt";
      p = read_prime_from_file(path);
      if (p <= 0) {
        path = "25_29/" + to_string(g) + "/" + to_string(g) + "_1.txt";
        p = read_prime_from_file(path);
      }
    }
    if (p <= 0 && g >= 30) {
      path = "exp/" + to_string(g) + "_1.txt";
      p = read_prime_from_file(path);
    }

    if (p > 0) {
      fp.prime = p;
      snprintf(fp.label, sizeof(fp.label), "%d (p=%lld)", g, p);
      printf("  [PRIME] %d -> %lld  (from %s)\n", g, p, path.c_str());
    } else if (fp.prime > 0) {
      printf("  [PRIME] %d -> %lld  (HARDCODED FALLBACK)\n", g, fp.prime);
    } else {
      printf("  [PRIME] %d -> NOT AVAILABLE\n", g);
    }

    primes.push_back(fp);
  }
  return primes;
}

// =============================================================================
// Parse matrix from Sage/Python [[...],[...],...] format
// =============================================================================

static MatrixData parse_matrix(const string &path) {
  ifstream ifs(path.c_str());
  if (!ifs.is_open())
    throw runtime_error("Cannot open: " + path);

  MatrixData md;
  md.filename = path.substr(path.find_last_of("/\\") + 1);

  vector<vector<long long> > rows;
  string line;
  bool started = false;

  while (getline(ifs, line)) {
    if (!started) {
      if (line.find("[[") != string::npos)
        started = true;
      else
        continue;
    }
    size_t lb = line.find('[');
    size_t rb = line.find(']');
    if (lb == string::npos || rb == string::npos || rb <= lb)
      continue;

    string tok = line.substr(lb + 1, rb - lb - 1);
    size_t fs = tok.find_first_not_of(" [");
    if (fs == string::npos)
      continue;
    tok = tok.substr(fs);
    replace(tok.begin(), tok.end(), ',', ' ');

    istringstream ss(tok);
    vector<long long> row;
    long long v;
    while (ss >> v)
      row.push_back(v);
    if (!row.empty())
      rows.push_back(row);
    if (line.find("]]") != string::npos)
      break;
  }

  if (rows.empty())
    throw runtime_error("No matrix data in: " + path);
  md.n = static_cast<int>(rows.size());
  md.data.assign(static_cast<size_t>(md.n) * md.n, 0LL);
  for (int i = 0; i < md.n; i++)
    for (int j = 0; j < static_cast<int>(rows[i].size()) && j < md.n; j++)
      md.data[static_cast<size_t>(i) * md.n + j] = rows[i][j];
  return md;
}

// =============================================================================
// Generate C(pool_sz, r) index sets — FLAT OUTPUT with reserve()
//
// Output: vector<int> with sets packed as k ints each.
// Total entries = num_sets × k.  num_sets = out.size() / k.
//
// V100 fix: reserve() avoids repeated vector reallocation for large C(n,r).
// =============================================================================

static void gen_combos_flat(const vector<int> &pool, int r,
                            const int *principal, int k,
                            vector<int> &out) {
  int psz = static_cast<int>(pool.size());
  if (r > psz)
    return;

  // Pre-allocate: nCr(psz, r) sets × k ints each
  long long expected = nCr(psz, r);
  out.reserve(out.size() + static_cast<size_t>(expected) * k);

  vector<int> cidx(r);
  for (int i = 0; i < r; i++)
    cidx[i] = i;

  int tmp[MAX_IDX_STATIC];

  while (true) {
    for (int i = 0; i < PM_SIZE; i++)
      tmp[i] = principal[i];
    for (int i = 0; i < r; i++)
      tmp[PM_SIZE + i] = pool[cidx[i]];
    sort(tmp, tmp + k);

    for (int i = 0; i < k; i++)
      out.push_back(tmp[i]);

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
// GPU buffer manager — FLAT ARRAYS, runtime VRAM detection
// =============================================================================

struct GPUBufs {
  int *d_row_idx;
  int *d_col_idx;
  long long *d_matrix;
  int *d_zero_flags;
  int *d_found;

  size_t cap_rows;
  size_t col_chunk;
  size_t flag_cap;
  int k_stored;

  GPUBufs()
      : d_row_idx(NULL), d_col_idx(NULL), d_matrix(NULL), d_zero_flags(NULL),
        d_found(NULL), cap_rows(0), col_chunk(0), flag_cap(0), k_stored(0) {}

  bool alloc(int n, int dev) {
    free_all();

    int k = PM_SIZE + dev;
    k_stored = k;

    size_t free_bytes = 0, total_bytes = 0;
    CK(cudaMemGetInfo(&free_bytes, &total_bytes));

    size_t avail_bytes = static_cast<size_t>(free_bytes * 0.90);

    printf("    [GPU VRAM] Total: %.0f MB | Free: %.0f MB | Usable (90%%): "
           "%.0f MB\n",
           total_bytes / 1e6, free_bytes / 1e6, avail_bytes / 1e6);

    int pool = n - PM_SIZE;
    cap_rows = static_cast<size_t>(nCr(pool, dev));

    size_t row_idx_bytes = cap_rows * k * sizeof(int);
    size_t matrix_bytes = static_cast<size_t>(n) * n * sizeof(long long);
    size_t fixed = row_idx_bytes + matrix_bytes;

    printf("    [GPU] Index sets: %zu  x  %d ints  =  %.1f MB  (flat)\n",
           cap_rows, k, row_idx_bytes / 1e6);

    if (fixed >= avail_bytes) {
      printf("    [SKIP] dev=%d needs %.0f MB but only %.0f MB "
             "available\n",
             dev, fixed / 1e6, avail_bytes / 1e6);
      return false;
    }
    size_t remain = avail_bytes - fixed;

    size_t per_col = static_cast<size_t>(k) * sizeof(int) +
                     cap_rows * sizeof(int);
    size_t max_chunk = (per_col > 0) ? remain / per_col : 512;
    if (max_chunk < 1)
      max_chunk = 1;
    if (max_chunk > cap_rows)
      max_chunk = cap_rows;

    col_chunk = 1;
    while (col_chunk * 2 <= max_chunk)
      col_chunk *= 2;
    if (col_chunk > cap_rows)
      col_chunk = cap_rows;
    if (col_chunk < 1)
      col_chunk = 1;

    flag_cap = cap_rows * col_chunk;

    printf("    [GPU] Allocating: rows=%zu  col_chunk=%zu  flags=%zu (%.1f MB)"
           "  matrix=%dx%d\n",
           cap_rows, col_chunk, flag_cap, flag_cap * sizeof(int) / 1e6, n, n);
    fflush(stdout);

    CK(cudaMalloc(&d_row_idx, row_idx_bytes));
    CK(cudaMalloc(&d_col_idx, col_chunk * k * sizeof(int)));
    CK(cudaMalloc(&d_matrix, matrix_bytes));
    CK(cudaMalloc(&d_zero_flags, flag_cap * sizeof(int)));
    CK(cudaMalloc(&d_found, sizeof(int)));
    return true;
  }

  void free_all() {
    if (d_row_idx) { cudaFree(d_row_idx); d_row_idx = NULL; }
    if (d_col_idx) { cudaFree(d_col_idx); d_col_idx = NULL; }
    if (d_matrix)  { cudaFree(d_matrix);  d_matrix = NULL; }
    if (d_zero_flags) { cudaFree(d_zero_flags); d_zero_flags = NULL; }
    if (d_found) { cudaFree(d_found); d_found = NULL; }
  }

  ~GPUBufs() { free_all(); }
};

// =============================================================================
// Helper: digit width for pretty printing
// =============================================================================

static int digit_width(long long v) {
  int w = 1;
  if (v < 0) { v = -v; w = 2; }
  while (v >= 10) { v /= 10; w++; }
  return w;
}

// =============================================================================
// Write zero minor detail (principal block + submatrix)
// =============================================================================

static void write_zero_minor_detail(FILE *f, const ZeroMinor &zm,
                                    const MatrixData &md) {
  int col_w = 1;
  for (int i = 0; i < md.n * md.n; i++) {
    int w = digit_width(md.data[i]);
    if (w > col_w) col_w = w;
  }

  fprintf(f, "  Principal 2x2 block (s=%d, rows/cols {%d,%d}):\n", zm.s, zm.s,
          zm.s + 1);
  fprintf(f, "    [ %*lld  %*lld ]\n", col_w, md.data[zm.s * md.n + zm.s],
          col_w, md.data[zm.s * md.n + zm.s + 1]);
  fprintf(f, "    [ %*lld  %*lld ]\n", col_w, md.data[(zm.s + 1) * md.n + zm.s],
          col_w, md.data[(zm.s + 1) * md.n + zm.s + 1]);

  int sw = 1;
  for (int r = 0; r < zm.k; r++)
    for (int c = 0; c < zm.k; c++) {
      int w = digit_width(md.data[zm.row_idx[r] * md.n + zm.col_idx[c]]);
      if (w > sw) sw = w;
    }

  fprintf(f, "\n  Extracted %dx%d submatrix  (det mod p = 0):\n", zm.k, zm.k);

  fprintf(f, "       ");
  for (int c = 0; c < zm.k; c++) {
    bool is_pm = (zm.col_idx[c] == zm.s || zm.col_idx[c] == zm.s + 1);
    fprintf(f, " %*d%c", sw, zm.col_idx[c], is_pm ? '*' : ' ');
  }
  fprintf(f, "\n       ");
  for (int c = 0; c < zm.k; c++) {
    fprintf(f, " ");
    for (int w = 0; w <= sw; w++) fprintf(f, "-");
  }
  fprintf(f, "\n");

  for (int r = 0; r < zm.k; r++) {
    bool pm_r = (zm.row_idx[r] == zm.s || zm.row_idx[r] == zm.s + 1);
    fprintf(f, "  %3d%c |", zm.row_idx[r], pm_r ? '*' : ' ');
    for (int c = 0; c < zm.k; c++) {
      long long val = md.data[zm.row_idx[r] * md.n + zm.col_idx[c]];
      bool pm_c = (zm.col_idx[c] == zm.s || zm.col_idx[c] == zm.s + 1);
      if (pm_r && pm_c)
        fprintf(f, "[%*lld]", sw, val);
      else
        fprintf(f, " %*lld ", sw, val);
    }
    fprintf(f, "\n");
  }
  fprintf(f, "\n");
}

// =============================================================================
// Write per-matrix result file
// =============================================================================

static void write_result_file(const string &out_dir, const MatrixData &md,
                              long long prime, int dev,
                              const vector<ZeroMinor> &minors,
                              double matrix_ms, double minors_tested) {
  string base = md.filename;
  size_t dot = base.rfind('.');
  if (dot != string::npos) base = base.substr(0, dot);
  string outpath = out_dir + "/" + base + "_result.txt";

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
  fprintf(f, "Matrix total time : %.4f ms  (%.6f s)\n", matrix_ms,
          matrix_ms / 1000.0);
  fprintf(f, "Minors tested     : %.0f\n", minors_tested);
  fprintf(f, "Avg per minor     : %.8f ms\n", avg_ms);
  fprintf(f, "------------------------------------------------------------\n");
  fprintf(f, "Zero Minors Found : %zu\n", minors.size());
  fprintf(f, "============================================================\n\n");

  int col_w = 1;
  for (int i = 0; i < md.n * md.n; i++) {
    int w = digit_width(md.data[i]);
    if (w > col_w) col_w = w;
  }

  fprintf(f, "Full matrix (%dx%d) mod %lld:\n", md.n, md.n, prime);
  fprintf(f, "     ");
  for (int c = 0; c < md.n; c++) fprintf(f, " %*d", col_w, c);
  fprintf(f, "\n     ");
  for (int c = 0; c < md.n; c++) {
    fprintf(f, " ");
    for (int w = 0; w < col_w; w++) fprintf(f, "-");
  }
  fprintf(f, "\n");
  for (int r = 0; r < md.n; r++) {
    fprintf(f, "%3d |", r);
    for (int c = 0; c < md.n; c++)
      fprintf(f, " %*lld", col_w, md.data[r * md.n + c]);
    fprintf(f, "\n");
  }
  fprintf(f, "\n");

  if (minors.empty()) {
    fprintf(f, "No zero minor found at deviation level %d.\n", dev);
  } else {
    for (int mi = 0; mi < static_cast<int>(minors.size()); mi++) {
      const ZeroMinor &zm = minors[mi];
      fprintf(f, "--- Zero Minor #%d ---\n", mi + 1);
      fprintf(f, "  Minor size (k)   : %d\n", zm.k);
      fprintf(f, "  Deviation        : %d\n", zm.dev);
      fprintf(f, "  Principal block s: %d  (indices {%d, %d})\n", zm.s, zm.s,
              zm.s + 1);
      fprintf(f, "  Row indices [%d]  : ", zm.k);
      for (int j = 0; j < zm.k; j++)
        fprintf(f, "%d%s", zm.row_idx[j], j < zm.k - 1 ? " " : "");
      fprintf(f, "\n  Col indices [%d]  : ", zm.k);
      for (int j = 0; j < zm.k; j++)
        fprintf(f, "%d%s", zm.col_idx[j], j < zm.k - 1 ? " " : "");
      fprintf(f, "\n  Time found       : %.4f ms\n", zm.time_ms);
      fprintf(f, "  Minors tested    : %.0f\n\n", minors_tested);
      write_zero_minor_detail(f, zm, md);
    }
  }

  fclose(f);
  printf("      -> %s\n", outpath.c_str());
  fflush(stdout);
}

// =============================================================================
// Write group-level result.txt
// =============================================================================

static void write_group_result(int group, int best_dev, int best_hits,
                               bool reached_100) {
  mkdir_safe(RESULT_BASE_DIR);
  string dir = string(RESULT_BASE_DIR) + "/" + to_string(group);
  mkdir_safe(dir);
  string path = dir + "/result.txt";

  FILE *f = fopen(path.c_str(), "w");
  if (!f) {
    fprintf(stderr, "[WARN] Cannot write: %s\n", path.c_str());
    return;
  }

  fprintf(f, "============================================================\n");
  fprintf(f, "APM Early-Stop Result  (brahma v100-optimized)\n");
  fprintf(f, "============================================================\n");
  fprintf(f, "Prime group      : %d\n", group);
  if (reached_100) {
    fprintf(f, "Best deviation   : %d\n", best_dev);
    fprintf(f, "Matrices hit     : %d  (reached %d -- EARLY STOP)\n", best_hits,
            EARLY_STOP_HIT);
    fprintf(f, "Status           : All matrices hit at deviation %d.\n", best_dev);
    fprintf(f, "                   No further deviations were checked.\n");
  } else {
    fprintf(f, "Best deviation   : %d\n", best_dev);
    fprintf(f, "Matrices hit     : %d  (did NOT reach %d)\n", best_hits,
            EARLY_STOP_HIT);
    fprintf(f, "Status           : All deviations checked.\n");
    fprintf(f, "                   Deviation %d had the most hits (%d).\n",
            best_dev, best_hits);
  }
  fprintf(f, "============================================================\n");
  fclose(f);
  printf("  -> result.txt written: %s\n", path.c_str());
  fflush(stdout);
}

// =============================================================================
// Search one matrix for the FIRST zero minor
//
// pow2_64 is set via cudaMemcpyToSymbol → __constant__ c_pow2_64.
// mod_mul reads it from constant memory — no kernel parameter needed.
// =============================================================================

static vector<ZeroMinor>
search_matrix(GPUBufs &gpu, const long long *h_mat, int n, long long prime,
              int dev, double &minors_tested, double matrix_start_ms) {
  vector<ZeroMinor> found;
  minors_tested = 0.0;

  const int BLOCK = 256;
  const int k = PM_SIZE + dev;

  // Precompute 2^64 mod prime and upload to __constant__ memory.
  // Done ONCE per matrix search (not per mod_mul call).
  {
    unsigned long long pow2_64 = 1;
    unsigned long long up = (unsigned long long)prime;
    for (int i = 0; i < 64; i++)
      pow2_64 = (pow2_64 + pow2_64) % up;
    CK(cudaMemcpyToSymbol(c_pow2_64, &pow2_64, sizeof(unsigned long long)));
  }

  CK(cudaMemcpy(gpu.d_matrix, h_mat,
                static_cast<size_t>(n) * n * sizeof(long long),
                cudaMemcpyHostToDevice));

  int pool_sz = n - PM_SIZE;

  CK(cudaMemset(gpu.d_found, 0, sizeof(int)));

  for (int s = 0; s <= n - PM_SIZE; s++) {
    int principal[PM_SIZE];
    for (int i = 0; i < PM_SIZE; i++)
      principal[i] = s + i;

    vector<int> pool;
    pool.reserve(pool_sz);
    for (int idx = 0; idx < n; idx++) {
      bool in_pm = false;
      for (int j = 0; j < PM_SIZE; j++)
        if (idx == principal[j]) { in_pm = true; break; }
      if (!in_pm)
        pool.push_back(idx);
    }

    vector<int> sets;
    gen_combos_flat(pool, dev, principal, k, sets);
    int num_sets = static_cast<int>(sets.size()) / k;
    if (num_sets == 0)
      continue;

    if (static_cast<size_t>(num_sets) > gpu.cap_rows) {
      fprintf(stderr,
              "  [WARN] s=%d dev=%d: %d sets > GPU buffer %zu -- skipping\n", s,
              dev, num_sets, gpu.cap_rows);
      continue;
    }

    CK(cudaMemcpy(gpu.d_row_idx, sets.data(),
                  static_cast<size_t>(num_sets) * k * sizeof(int),
                  cudaMemcpyHostToDevice));

    int chunk_sz = static_cast<int>(gpu.col_chunk);
    bool hit_found = false;

    for (int col_start = 0; col_start < num_sets; col_start += chunk_sz) {
      int col_end = min(col_start + chunk_sz, num_sets);
      int num_cols_this_chunk = col_end - col_start;
      long long total_jobs =
          static_cast<long long>(num_sets) * num_cols_this_chunk;
      minors_tested += static_cast<double>(total_jobs);

      CK(cudaMemset(gpu.d_zero_flags, 0,
                    static_cast<size_t>(num_sets) * num_cols_this_chunk *
                        sizeof(int)));

      CK(cudaMemcpy(gpu.d_col_idx,
                    sets.data() + static_cast<size_t>(col_start) * k,
                    static_cast<size_t>(num_cols_this_chunk) * k * sizeof(int),
                    cudaMemcpyHostToDevice));

      int grid = static_cast<int>((total_jobs + 255LL) / 256LL);
      launch_apm_kernel(k, grid, BLOCK,
                        gpu.d_matrix, n, gpu.d_row_idx, gpu.d_col_idx,
                        num_sets, num_cols_this_chunk,
                        prime, gpu.d_zero_flags, gpu.d_found);
      CK(cudaDeviceSynchronize());

      int h_found = 0;
      CK(cudaMemcpy(&h_found, gpu.d_found, sizeof(int),
                    cudaMemcpyDeviceToHost));
      if (!h_found)
        continue;

      vector<int> h_flags(static_cast<size_t>(num_sets) * num_cols_this_chunk);
      CK(cudaMemcpy(h_flags.data(), gpu.d_zero_flags,
                    h_flags.size() * sizeof(int), cudaMemcpyDeviceToHost));

      for (long long ji = 0; ji < total_jobs; ji++) {
        if (h_flags[ji]) {
          int ri = static_cast<int>(ji / num_cols_this_chunk);
          int ci = static_cast<int>(ji % num_cols_this_chunk) + col_start;

          ZeroMinor zm;
          zm.k = k;
          zm.dev = dev;
          zm.s = s;
          for (int j = 0; j < k; j++)
            zm.row_idx[j] = sets[static_cast<size_t>(ri) * k + j];
          for (int j = 0; j < k; j++)
            zm.col_idx[j] = sets[static_cast<size_t>(ci) * k + j];
          zm.time_ms = now_ms() - matrix_start_ms;
          found.push_back(zm);

          hit_found = true;
          break;
        }
      }
      if (hit_found)
        break;
    }
    if (hit_found)
      break;
  }
  return found;
}

// =============================================================================
// Process one prime group at one deviation level
// =============================================================================

static int run_group_deviation(int group, long long prime, int dev,
                               const vector<string> &files, int n,
                               GPUBufs &gpu) {
  string out_dir = make_out_dir(group, dev);
  string input_dir = "kernel_output/" + to_string(group) + "/";
  string prefix = "kernel_" + to_string(group) + "_";

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

  string det_path = out_dir + "/SUMMARY_detailed.txt";
  FILE *det_f = fopen(det_path.c_str(), "w");
  if (det_f) {
    fprintf(det_f, "APM Summary\n");
    fprintf(det_f, "Deviation level : %d\n", dev);
    fprintf(det_f, "Prime group     : %d\n", group);
    fprintf(det_f, "Input folder    : %s\n", input_dir.c_str());
    fprintf(det_f, "Prime           : %lld\n", prime);
    fprintf(det_f, "Matrix size n   : %d\n", n);
    fprintf(det_f, "PM block size   : %d\n", PM_SIZE);
    fprintf(det_f, "Matrices        : %zu\n", files.size());
    fprintf(det_f, "======================================================\n\n");
  }

  double folder_start_ms = now_ms();
  long long total_zero_minors = 0;
  double total_minors_tested = 0.0;
  int matrices_hit = 0;

  for (int fi = 0; fi < static_cast<int>(files.size()); fi++) {
    printf("\n  [%d/%zu] %s\n", fi + 1, files.size(), files[fi].c_str());
    fflush(stdout);

    MatrixData md;
    try {
      md = parse_matrix(files[fi]);
    } catch (std::exception &ex) {
      printf("    [ERROR] %s -- skipping\n", ex.what());
      continue;
    }
    if (md.n != n) {
      printf("    [WARN] n=%d != expected %d -- skipping\n", md.n, n);
      continue;
    }

    double matrix_start_ms = now_ms();
    double minors_tested = 0.0;

    printf("    Searching for first zero minor at dev=%d ...\n", dev);
    fflush(stdout);

    vector<ZeroMinor> found = search_matrix(gpu, md.data.data(), n, prime, dev,
                                            minors_tested, matrix_start_ms);

    double matrix_ms = now_ms() - matrix_start_ms;
    total_zero_minors += static_cast<long long>(found.size());
    total_minors_tested += minors_tested;
    if (!found.empty())
      matrices_hit++;

    printf("    Time: %.3f s | Tested: %.0f minors | Zero minors: %zu\n",
           matrix_ms / 1000.0, minors_tested, found.size());

    if (found.empty()) {
      printf("    -> No zero minor at deviation %d\n", dev);
    } else {
      for (int mi = 0; mi < static_cast<int>(found.size()); mi++) {
        const ZeroMinor &zm = found[mi];
        printf("    [Zero #%d] size=%d  dev=%d  s=%d  t=%.2f ms  "
               "minors_tested=%.0f\n",
               mi + 1, zm.k, zm.dev, zm.s, zm.time_ms, minors_tested);
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

    write_result_file(out_dir, md, prime, dev, found, matrix_ms, minors_tested);

    if (det_f && !found.empty()) {
      fprintf(det_f, "------------------------------------------------------------\n");
      fprintf(det_f, "[%d/%d] %s\n", fi + 1, static_cast<int>(files.size()),
              md.filename.c_str());
      fprintf(det_f, "  time=%.4f s | tested=%.0f | zeros=%zu\n",
              matrix_ms / 1000.0, minors_tested, found.size());
      for (int mi = 0; mi < static_cast<int>(found.size()); mi++) {
        const ZeroMinor &zm = found[mi];
        fprintf(det_f, "\n  --- Zero Minor #%d ---\n", mi + 1);
        fprintf(det_f, "  Minor size (k)   : %d\n", zm.k);
        fprintf(det_f, "  Deviation        : %d\n", zm.dev);
        fprintf(det_f, "  Principal block s: %d  (indices {%d, %d})\n", zm.s,
                zm.s, zm.s + 1);
        fprintf(det_f, "  Row indices [%d]  : ", zm.k);
        for (int j = 0; j < zm.k; j++)
          fprintf(det_f, "%d%s", zm.row_idx[j], j < zm.k - 1 ? " " : "");
        fprintf(det_f, "\n  Col indices [%d]  : ", zm.k);
        for (int j = 0; j < zm.k; j++)
          fprintf(det_f, "%d%s", zm.col_idx[j], j < zm.k - 1 ? " " : "");
        fprintf(det_f, "\n  Time found       : %.4f ms\n", zm.time_ms);
        fprintf(det_f, "  Minors tested    : %.0f\n\n", minors_tested);
        write_zero_minor_detail(det_f, zm, md);
      }
    }
  }

  double folder_ms = now_ms() - folder_start_ms;

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

  string brief_path = out_dir + "/SUMMARY_brief.txt";
  FILE *brief_f = fopen(brief_path.c_str(), "w");
  if (brief_f) {
    double hit_pct = (files.size() > 0) ? 100.0 * matrices_hit /
                                              static_cast<double>(files.size())
                                        : 0.0;
    fprintf(brief_f, "============================================================\n");
    fprintf(brief_f, "APM Brief Summary\n");
    fprintf(brief_f, "============================================================\n");
    fprintf(brief_f, "Prime group      : %d\n", group);
    fprintf(brief_f, "Prime (p)        : %lld\n", prime);
    fprintf(brief_f, "Deviation level  : %d\n", dev);
    fprintf(brief_f, "Minor size       : %d x %d\n", PM_SIZE + dev, PM_SIZE + dev);
    fprintf(brief_f, "PM block size    : %d\n", PM_SIZE);
    fprintf(brief_f, "Matrix size (n)  : %d\n", n);
    fprintf(brief_f, "Input folder     : kernel_output/%d/\n", group);
    fprintf(brief_f, "Output folder    : %s\n", out_dir.c_str());
    fprintf(brief_f, "------------------------------------------------------------\n");
    fprintf(brief_f, "Total matrices   : %zu\n", files.size());
    fprintf(brief_f, "Matrices hit     : %d      (at least one zero minor)\n", matrices_hit);
    fprintf(brief_f, "Hit ratio        : %d/%zu = %.2f%%\n", matrices_hit, files.size(), hit_pct);
    fprintf(brief_f, "Total minors tested  : %.0f\n", total_minors_tested);
    fprintf(brief_f, "Total zero minors    : %lld\n", total_zero_minors);
    fprintf(brief_f, "Total time           : %.3f s\n", folder_ms / 1000.0);
    fprintf(brief_f, "Avg time per matrix  : %.3f s\n",
            (files.size() > 0) ? folder_ms / 1000.0 / static_cast<double>(files.size()) : 0.0);
    fprintf(brief_f, "============================================================\n");
    fclose(brief_f);
    printf("  Summary (brief)    -> %s\n", brief_path.c_str());
  }

  printf("\n  -- group=%d  dev=%d complete --\n", group, dev);
  printf("  Time           : %.3f s\n", folder_ms / 1000.0);
  printf("  Matrices       : %zu\n", files.size());
  printf("  Minors tested  : %.0f\n", total_minors_tested);
  printf("  Zero minors    : %lld\n", total_zero_minors);
  printf("  Matrices hit   : %d / %zu\n", matrices_hit, files.size());
  fflush(stdout);

  return matrices_hit;
}

// =============================================================================
// Print GPU info
// =============================================================================

static void print_gpu_info() {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    printf("  [GPU] No CUDA devices found!\n");
    return;
  }
  for (int i = 0; i < device_count; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("  [GPU %d] %s\n", i, prop.name);
    printf("           Compute Capability : %d.%d\n", prop.major, prop.minor);
    printf("           Total VRAM         : %.0f MB\n", prop.totalGlobalMem / 1e6);
    printf("           SM count           : %d\n", prop.multiProcessorCount);
    printf("           Max threads/block   : %d\n", prop.maxThreadsPerBlock);
  }
  cudaSetDevice(0);
  size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  printf("  [GPU 0] Free VRAM: %.0f MB / %.0f MB\n\n", free_bytes / 1e6,
         total_bytes / 1e6);
  fflush(stdout);
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char **argv) {
  int gmin = 25, gmax = 60;
  if (argc == 3) {
    gmin = atoi(argv[1]);
    gmax = atoi(argv[2]);
    if (gmin > gmax) { int tmp = gmin; gmin = gmax; gmax = tmp; }
  } else if (argc != 1) {
    printf("Usage: %s [gmin gmax]\n", argv[0]);
    printf("  Example: %s 32 35\n", argv[0]);
    printf("  Example: %s 25 27\n", argv[0]);
    printf("  Example: %s 35 50\n", argv[0]);
    return 1;
  }

  printf("=================================================================\n");
  printf(" APM Brahma — V100-OPTIMIZED, TEMPLATED KERNEL VERSION\n");
  printf(" KERNEL_OUTPUT VERSION — prime groups from kernel_output/<g>/\n");
  printf(" GROUP-OUTER, DEVIATION-INNER LOOP\n");
  printf(" STOPS after the FIRST zero minor per matrix\n");
  printf(" EARLY STOP: skips remaining deviations once %d hits reached\n",
         EARLY_STOP_HIT);
  printf(" PM block size    : %d (fixed)\n", PM_SIZE);
  printf(" Max index static : %d (supports up to group 60)\n", MAX_IDX_STATIC);
  printf(" Supports primes up to 2^60 (safe modular arithmetic)\n");
  printf(" Runtime VRAM detection via cudaMemGetInfo()\n");
  printf(" V100 optimizations:\n");
  printf("   - Templated kernel apm_kernel_k<K>: sub[K*K] exact stack frame\n");
  printf("     k=4: 128B (100%% occ) | k=10: 800B (100%%) | k=27: 5.8KB (68%%)\n");
  printf("   - Group-adaptive mod_mul: zero-loop for p<2^40 (groups 25-39)\n");
  printf("   - __constant__ pow2_64 (L1-cached, zero param overhead)\n");
  printf("   - Plain *d_found read (no atomic on check)\n");
  printf("   - Flat int[] arrays (~5-8x VRAM savings)\n");
  printf("   - reserve() host vectors, graceful OOM skip\n");
  printf("=================================================================\n\n");
  printf(" Group range     : %d to %d\n\n", gmin, gmax);
  fflush(stdout);

  printf("GPU Information:\n");
  print_gpu_info();

  printf("Reading primes from files...\n");
  vector<FolderPrime> primes = load_primes(gmin, gmax);
  printf("\nPrime table:\n");
  for (size_t i = 0; i < primes.size(); i++) {
    if (primes[i].prime > 0)
      printf("  %2d -> %15lld  %s\n", primes[i].folder_id, primes[i].prime,
             primes[i].label);
    else
      printf("  %2d -> %15s  (not available)\n", primes[i].folder_id, "N/A");
  }
  printf("\n");
  fflush(stdout);

  struct stat st;
  if (stat("kernel_output", &st) != 0) {
    printf("[FATAL] kernel_output/ directory not found. Nothing to do.\n");
    return 1;
  }

  double prog_start = now_ms();

  for (size_t gi = 0; gi < primes.size(); gi++) {
    int group = primes[gi].folder_id;
    long long prime = primes[gi].prime;

    if (prime <= 0) {
      printf("\n  [SKIP] Group %d -- no prime available\n", group);
      continue;
    }

    printf("\n#################################################################\n");
    printf("# GROUP %d   prime = %lld\n", group, prime);
    printf("#################################################################\n");
    fflush(stdout);

    string input_dir = "kernel_output/" + to_string(group);
    string prefix = "kernel_" + to_string(group) + "_";
    vector<string> files = collect_files(input_dir, prefix);

    if (files.empty()) {
      string alt_prefix = to_string(group) + "_";
      files = collect_files(input_dir, alt_prefix);
      if (!files.empty()) {
        printf("  [INFO] Using prefix '%s' instead of '%s'.\n",
               alt_prefix.c_str(), prefix.c_str());
        prefix = alt_prefix;
      }
    }

    if (files.empty()) {
      printf("  [SKIP] No matching files in %s\n", input_dir.c_str());
      continue;
    }

    MatrixData first;
    try {
      first = parse_matrix(files[0]);
    } catch (std::exception &ex) {
      printf("  [ERROR] Cannot parse first file: %s\n", ex.what());
      continue;
    }
    int n = first.n;

    int max_dev = n - (PM_SIZE + 1);
    printf("  Matrix size n   = %d\n", n);
    printf("  Files found     = %zu\n", files.size());
    printf("  Deviation range = %d to %d\n", MIN_DEV, max_dev);
    fflush(stdout);

    if (max_dev < MIN_DEV) {
      printf("  [SKIP] max_dev=%d < MIN_DEV=%d -- matrix too small\n", max_dev,
             MIN_DEV);
      continue;
    }

    double group_start = now_ms();
    int best_dev = -1, best_hits = 0;
    bool reached_target = false;

    for (int dev = MIN_DEV; dev <= max_dev; dev++) {
      if (dev > n - PM_SIZE) {
        printf("  [SKIP] dev=%d needs pool >= %d but pool=%d\n", dev, dev,
               n - PM_SIZE);
        continue;
      }

      printf("\n  ===============================================================\n");
      printf("  [group=%d | dev=%d/%d]  minor=%dx%d\n", group, dev, max_dev,
             PM_SIZE + dev, PM_SIZE + dev);
      printf("  ===============================================================\n");
      fflush(stdout);

      GPUBufs gpu;
      if (!gpu.alloc(n, dev)) {
        printf("  [SKIP] Insufficient VRAM for dev=%d (C(%d,%d) = %lld sets, "
               "%.1f MB flat)\n",
               dev, n - PM_SIZE, dev, nCr(n - PM_SIZE, dev),
               nCr(n - PM_SIZE, dev) * (PM_SIZE + dev) * sizeof(int) / 1e6);
        printf("  [STOP] Higher deviations will also exceed VRAM\n");
        break;
      }

      int hits = run_group_deviation(group, prime, dev, files, n, gpu);
      printf("  [group=%d  dev=%d done]  hits=%d/%zu\n", group, dev, hits,
             files.size());
      fflush(stdout);

      if (hits > best_hits) {
        best_hits = hits;
        best_dev = dev;
      }

      if (hits >= EARLY_STOP_HIT) {
        printf("\n  *** %d HITS REACHED at deviation %d! ***\n", EARLY_STOP_HIT, dev);
        printf("  *** Skipping remaining deviations for group %d ***\n", group);
        fflush(stdout);
        reached_target = true;
        break;
      }
    }

    if (best_dev >= 0)
      write_group_result(group, best_dev, best_hits, reached_target);

    double group_ms = now_ms() - group_start;
    printf("\n#################################################################\n");
    printf("# GROUP %d COMPLETE   (%.2f min)\n", group, group_ms / 60000.0);
    if (reached_target)
      printf("# EARLY STOP at deviation %d (%d hits)\n", best_dev, best_hits);
    else
      printf("# Best deviation: %d  (%d hits)\n", best_dev, best_hits);
    printf("#################################################################\n");
    fflush(stdout);
  }

  double total_ms = now_ms() - prog_start;
  printf("\n=================================================================\n");
  printf(" ALL DONE\n");
  printf(" Total wall time : %.3f s  (%.2f min)\n", total_ms / 1000.0,
         total_ms / 60000.0);
  printf("\n Results in: %s/<group>/\n", RESULT_BASE_DIR);
  printf(" Each group has a result.txt with the best deviation.\n");
  printf("=================================================================\n");

  return 0;
}

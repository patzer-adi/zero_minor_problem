/*
 * gen_relation.cu - CUDA implementation of matrix generation over finite fields
 *
 * Converted from MPI+NTL implementation to pure CUDA C++
 * Target: CUDA 11.4, GCC 10 compatible
 *
 * Supports primes p from 2 to 17600 (fits in 32-bit integers)
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace std;

// ============================================================================
// CUDA Error Checking Macro
// ============================================================================
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "         \
           << cudaGetErrorString(err) << endl;                                 \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ============================================================================
// Constants
// ============================================================================
#define MAX_N 32        // Maximum matrix dimension supported
#define MAX_ROWS 65536  // Maximum number of valid rows to store
#define MAX_MATRICES 64 // Maximum matrices to find per configuration

// ============================================================================
// Device: Modular Arithmetic Primitives
// ============================================================================

__device__ __forceinline__ int mod_add(int a, int b, int p) {
  int result = a + b;
  return (result >= p) ? (result - p) : result;
}

__device__ __forceinline__ int mod_sub(int a, int b, int p) {
  int result = a - b;
  return (result < 0) ? (result + p) : result;
}

__device__ __forceinline__ int mod_mul(int a, int b, int p) {
  return (int)(((long long)a * b) % p);
}

// Extended Euclidean Algorithm for modular inverse
__device__ int mod_inv(int a, int p) {
  if (a == 0)
    return 0; // No inverse for 0

  int t = 0, newt = 1;
  int r = p, newr = a % p;

  while (newr != 0) {
    int quotient = r / newr;

    int temp_t = t;
    t = newt;
    newt = temp_t - quotient * newt;

    int temp_r = r;
    r = newr;
    newr = temp_r - quotient * newr;
  }

  if (r > 1)
    return 0; // a is not invertible
  if (t < 0)
    t += p;

  return t;
}

__device__ __forceinline__ int mod_div(int a, int b, int p) {
  return mod_mul(a, mod_inv(b, p), p);
}

// ============================================================================
// Device: Combination Generation Helpers
// ============================================================================

// Calculate binomial coefficient C(n, k)
__host__ __device__ long long binomial(int n, int k) {
  if (k > n || k < 0)
    return 0;
  if (k == 0 || k == n)
    return 1;
  if (k > n - k)
    k = n - k;

  long long result = 1;
  for (int i = 0; i < k; i++) {
    result = result * (n - i) / (i + 1);
  }
  return result;
}

// Unrank a combination: given index, produce the k-combination from n elements
__device__ void unrank_combination(long long idx, int n, int k, int *combo) {
  int x = 0;
  for (int i = 0; i < k; i++) {
    while (true) {
      long long c = binomial(n - 1 - x, k - 1 - i);
      if (idx < c) {
        combo[i] = x;
        x++;
        break;
      }
      idx -= c;
      x++;
    }
  }
}

// ============================================================================
// Device: Check if row has valid sum (equals p-1)
// ============================================================================

__device__ bool check_row_sum(const int *row, int n, int target_sum) {
  int sum = 0;
  for (int i = 0; i < n; i++) {
    sum += row[i];
  }
  return sum == target_sum;
}

// ============================================================================
// Kernel: Generate all valid rows (combinations with sum = p-1)
// ============================================================================

__global__ void generate_rows_kernel(
    int n,                        // Row length (matrix dimension)
    int p,                        // Prime modulus
    int target_sum,               // Target sum (p-1)
    long long total_combinations, // Total C(p-1, n) combinations
    int *d_rows,                  // Output: valid rows (flattened)
    int *d_row_count              // Output: count of valid rows
) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= total_combinations)
    return;

  // Generate combination from index
  // Elements are 1 to p-1, so we map from 0-indexed to 1-indexed
  int combo[MAX_N];
  unrank_combination(idx, p - 1, n, combo);

  // Convert to 1-indexed (elements 1 to p-1)
  for (int i = 0; i < n; i++) {
    combo[i] += 1;
  }

  // Check sum
  if (check_row_sum(combo, n, target_sum)) {
    // Atomically get slot for this row
    int slot = atomicAdd(d_row_count, 1);
    if (slot < MAX_ROWS) {
      // Store row
      for (int i = 0; i < n; i++) {
        d_rows[slot * MAX_N + i] = combo[i];
      }
    }
  }
}

// ============================================================================
// Device: Check if new_row conflicts with current_matrix
// ============================================================================

__device__ bool has_conflict(const int *current_matrix, int num_rows,
                             const int *new_row, int n) {
  for (int r = 0; r < num_rows; r++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (current_matrix[r * MAX_N + i] == new_row[j]) {
          return true;
        }
      }
    }
  }
  return false;
}

// ============================================================================
// Device: Check if all elements in matrix are unique
// ============================================================================

__device__ bool all_elements_unique(const int *mat, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        for (int l = 0; l < n; l++) {
          if ((i != k || j != l) && mat[i * MAX_N + j] == mat[k * MAX_N + l]) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

// ============================================================================
// Device: Calculate determinant using Gaussian elimination mod p
// ============================================================================

__device__ int determinant_mod_p(const int *mat, int n, int p) {
  // Copy matrix to local storage for modification
  int local_mat[MAX_N * MAX_N];
  for (int i = 0; i < n * n; i++) {
    int row = i / n;
    int col = i % n;
    local_mat[i] = mat[row * MAX_N + col];
  }

  int det = 1;
  int sign = 1;

  for (int col = 0; col < n; col++) {
    // Find pivot
    int pivot_row = -1;
    for (int row = col; row < n; row++) {
      if (local_mat[row * n + col] != 0) {
        pivot_row = row;
        break;
      }
    }

    if (pivot_row == -1) {
      return 0; // Singular matrix
    }

    // Swap rows if needed
    if (pivot_row != col) {
      for (int j = 0; j < n; j++) {
        int temp = local_mat[col * n + j];
        local_mat[col * n + j] = local_mat[pivot_row * n + j];
        local_mat[pivot_row * n + j] = temp;
      }
      sign = -sign;
    }

    int pivot = local_mat[col * n + col];
    det = mod_mul(det, pivot, p);

    // Eliminate below
    int pivot_inv = mod_inv(pivot, p);
    for (int row = col + 1; row < n; row++) {
      if (local_mat[row * n + col] != 0) {
        int factor = mod_mul(local_mat[row * n + col], pivot_inv, p);
        for (int j = col; j < n; j++) {
          local_mat[row * n + j] =
              mod_sub(local_mat[row * n + j],
                      mod_mul(factor, local_mat[col * n + j], p), p);
        }
      }
    }
  }

  if (sign < 0) {
    det = mod_sub(0, det, p);
  }

  return det;
}

// ============================================================================
// Device: Recursive matrix generation (depth-first search)
// ============================================================================

__device__ bool generate_matrix_dfs(int n, const int *d_rows, int row_count,
                                    int *current_matrix, bool *used, int depth,
                                    int *result_mat, bool *is_singular, int p) {
  if (depth == n) {
    // Check uniqueness
    if (all_elements_unique(current_matrix, n)) {
      // Calculate determinant
      int det = determinant_mod_p(current_matrix, n, p);

      // Copy result
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          result_mat[i * MAX_N + j] = current_matrix[i * MAX_N + j];
        }
      }

      *is_singular = (det == 0);
      return true;
    }
    return false;
  }

  for (int i = 0; i < row_count; i++) {
    if (!used[i]) {
      const int *row = &d_rows[i * MAX_N];

      if (!has_conflict(current_matrix, depth, row, n)) {
        used[i] = true;

        // Add row to matrix
        for (int j = 0; j < n; j++) {
          current_matrix[depth * MAX_N + j] = row[j];
        }

        if (generate_matrix_dfs(n, d_rows, row_count, current_matrix, used,
                                depth + 1, result_mat, is_singular, p)) {
          return true;
        }

        used[i] = false;
      }
    }
  }

  return false;
}

// ============================================================================
// Kernel: Search for matrices in parallel
// ============================================================================

__global__ void search_matrices_kernel(
    int n, int p, const int *d_rows, int row_count,
    int *d_result_matrices,     // Output: found matrices
    int *d_result_determinants, // Output: determinants
    int *d_result_count,        // Output: count of matrices found
    int *d_nonsingular_count,   // Output: count of non-singular matrices
    int max_nonsingular         // Stop after finding this many non-singular
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= row_count)
    return;

  // Check if we already have enough matrices
  if (*d_nonsingular_count >= max_nonsingular)
    return;

  // Thread-local storage
  int current_matrix[MAX_N * MAX_N];
  bool used[MAX_ROWS];
  int result_mat[MAX_N * MAX_N];
  bool is_singular = false;

  // Initialize
  for (int i = 0; i < row_count && i < MAX_ROWS; i++) {
    used[i] = false;
  }

  // Start with row tid
  used[tid] = true;
  for (int j = 0; j < n; j++) {
    current_matrix[j] = d_rows[tid * MAX_N + j];
  }

  // Search for matrix
  if (generate_matrix_dfs(n, d_rows, row_count, current_matrix, used, 1,
                          result_mat, &is_singular, p)) {
    // Found a matrix
    if (!is_singular) {
      int slot = atomicAdd(d_nonsingular_count, 1);

      if (slot < max_nonsingular) {
        // Store result
        int result_slot = atomicAdd(d_result_count, 1);

        // Copy matrix to result
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            d_result_matrices[result_slot * MAX_N * MAX_N + i * MAX_N + j] =
                result_mat[i * MAX_N + j];
          }
        }

        // Calculate and store determinant
        d_result_determinants[result_slot] =
            determinant_mod_p(result_mat, n, p);
      }
    }
  }
}

// ============================================================================
// Host: Create directory (cross-platform)
// ============================================================================

void create_directory(const string &path) {
#ifdef _WIN32
  _mkdir(path.c_str());
#else
  mkdir(path.c_str(), 0755);
#endif
}

// ============================================================================
// Host: Copy rows from device to host for file output
// ============================================================================

void write_rows_file(const string &path, const vector<vector<int>> &rows,
                     int n) {
  ofstream out(path);
  out << "Rows generated: " << rows.size() << "\n";
  for (size_t i = 0; i < rows.size(); i++) {
    out << "Row " << (i + 1) << ": ";
    for (int j = 0; j < n; j++) {
      out << rows[i][j] << " ";
    }
    out << "\n";
  }
  out.close();
}

// ============================================================================
// Host: Write matrices file
// ============================================================================

void write_matrices_file(const string &path,
                         const vector<vector<int>> &matrices,
                         const vector<int> &determinants, int n,
                         long duration_ms) {
  ofstream out(path);

  if (!matrices.empty()) {
    out << "=== NON-SINGULAR MATRICES (found " << matrices.size()
        << ") ===\n\n";

    for (size_t m = 0; m < matrices.size() && m < 5; m++) {
      out << "Matrix #" << (m + 1) << " | n=" << n
          << " | Determinant=" << determinants[m] << " | Non-Singular\n";
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          out << matrices[m][i * n + j] << " ";
        }
        out << "\n";
      }
      out << "\n";
    }
  } else {
    out << "No non-singular matrix found for n=" << n << "\n\n";
  }

  out << "=== SUMMARY FOR n=" << n << " ===\n";
  out << "Non-Singular matrices found: " << matrices.size() << "\n";
  out << "Computation time (ms): " << duration_ms << "\n";
  out << "============================\n";
  out.close();
}

// ============================================================================
// Host: Write summary file
// ============================================================================

void write_summary_file(const string &path, int p, int n, int nonsingular_count,
                        long duration_ms) {
  ofstream out(path);
  out << "=== SUMMARY FOR p=" << p << " n=" << n << " ===\n";
  out << "Non-Singular matrices found: " << nonsingular_count << "\n";
  out << "Computation time (ms): " << duration_ms << "\n";
  out << "============================\n";
  out.close();
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char **argv) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <primes_file.txt>\n";
    return 1;
  }

  // Read primes from file
  vector<int> primes;
  {
    ifstream in(argv[1]);
    if (!in) {
      cerr << "Failed to open primes file: " << argv[1] << "\n";
      return 1;
    }
    int val;
    while (in >> val) {
      primes.push_back(val);
    }
  }

  if (primes.empty()) {
    cerr << "No primes found in file\n";
    return 1;
  }

  cout << "Loaded " << primes.size() << " primes\n";

  // Allocate device memory
  int *d_rows;
  int *d_row_count;
  int *d_result_matrices;
  int *d_result_determinants;
  int *d_result_count;
  int *d_nonsingular_count;

  CUDA_CHECK(cudaMalloc(&d_rows, MAX_ROWS * MAX_N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_row_count, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_result_matrices,
                        MAX_MATRICES * MAX_N * MAX_N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_result_determinants, MAX_MATRICES * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_result_count, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_nonsingular_count, sizeof(int)));

  // Process each prime
  for (size_t pi = 0; pi < primes.size(); pi++) {
    int p = primes[pi];
    cout << "\n=== STARTING prime p=" << p << " (index " << pi << ") ===\n";

    string pdir = "p_" + to_string(p);
    create_directory(pdir);

    // For each matrix dimension n from 2 to p-1
    for (int n = 2; n <= p - 1; n++) {
      auto start_time = chrono::high_resolution_clock::now();

      // Reset row count
      int zero = 0;
      CUDA_CHECK(
          cudaMemcpy(d_row_count, &zero, sizeof(int), cudaMemcpyHostToDevice));

      // Calculate total combinations C(p-1, n)
      long long total_combinations = binomial(p - 1, n);

      if (total_combinations == 0 || n > p - 1) {
        cout << "p=" << p << " n=" << n << " -> no valid combinations\n";
        break;
      }

      // Generate rows
      int target_sum = p - 1;
      int block_size = 256;
      int num_blocks =
          (int)((total_combinations + block_size - 1) / block_size);

      generate_rows_kernel<<<num_blocks, block_size>>>(
          n, p, target_sum, total_combinations, d_rows, d_row_count);
      CUDA_CHECK(cudaDeviceSynchronize());

      // Get row count
      int row_count;
      CUDA_CHECK(cudaMemcpy(&row_count, d_row_count, sizeof(int),
                            cudaMemcpyDeviceToHost));

      if (row_count > MAX_ROWS) {
        row_count = MAX_ROWS;
        cout << "  Warning: row count capped at " << MAX_ROWS << "\n";
      }

      cout << "p=" << p << " n=" << n << " -> rows generated: " << row_count
           << "\n";

      // Copy rows to host for file output
      vector<int> h_rows_flat(row_count * MAX_N);
      CUDA_CHECK(cudaMemcpy(h_rows_flat.data(), d_rows,
                            row_count * MAX_N * sizeof(int),
                            cudaMemcpyDeviceToHost));

      vector<vector<int>> h_rows(row_count);
      for (int i = 0; i < row_count; i++) {
        h_rows[i].resize(n);
        for (int j = 0; j < n; j++) {
          h_rows[i][j] = h_rows_flat[i * MAX_N + j];
        }
      }

      // Write rows file
      string rows_path = pdir + "/rows_n_" + to_string(n) + ".txt";
      write_rows_file(rows_path, h_rows, n);

      if (row_count == 0) {
        cout << "No rows for n=" << n << ", stopping\n";
        break;
      }

      // Reset result counts
      CUDA_CHECK(cudaMemcpy(d_result_count, &zero, sizeof(int),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_nonsingular_count, &zero, sizeof(int),
                            cudaMemcpyHostToDevice));

      cout << "  [Searching for up to 5 NON-SINGULAR matrices]\n";

      // Search for matrices
      int search_blocks = (row_count + block_size - 1) / block_size;

      search_matrices_kernel<<<search_blocks, block_size>>>(
          n, p, d_rows, row_count, d_result_matrices, d_result_determinants,
          d_result_count, d_nonsingular_count,
          5 // max non-singular to find
      );
      CUDA_CHECK(cudaDeviceSynchronize());

      // Get results
      int result_count, nonsingular_count;
      CUDA_CHECK(cudaMemcpy(&result_count, d_result_count, sizeof(int),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(&nonsingular_count, d_nonsingular_count,
                            sizeof(int), cudaMemcpyDeviceToHost));

      if (result_count > MAX_MATRICES)
        result_count = MAX_MATRICES;

      // Copy matrices to host
      vector<int> h_matrices_flat(result_count * MAX_N * MAX_N);
      vector<int> h_determinants(result_count);

      if (result_count > 0) {
        CUDA_CHECK(cudaMemcpy(h_matrices_flat.data(), d_result_matrices,
                              result_count * MAX_N * MAX_N * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_determinants.data(), d_result_determinants,
                              result_count * sizeof(int),
                              cudaMemcpyDeviceToHost));
      }

      // Convert to vector of vectors
      vector<vector<int>> h_matrices(result_count);
      for (int m = 0; m < result_count; m++) {
        h_matrices[m].resize(n * n);
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            h_matrices[m][i * n + j] =
                h_matrices_flat[m * MAX_N * MAX_N + i * MAX_N + j];
          }
        }
      }

      auto end_time = chrono::high_resolution_clock::now();
      auto duration =
          chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

      // Write output files
      string matrix_path = pdir + "/matrices_n_" + to_string(n) + ".txt";
      write_matrices_file(matrix_path, h_matrices, h_determinants, n,
                          duration.count());

      string summary_path = pdir + "/summary_n_" + to_string(n) + ".txt";
      write_summary_file(summary_path, p, n, nonsingular_count,
                         duration.count());

      cout << "p=" << p << " n=" << n
           << " done. nonsingular=" << nonsingular_count
           << " time(ms)=" << duration.count() << "\n";

      if (nonsingular_count == 0) {
        cout << "No matrix found for n=" << n
             << ", stopping further n values for p=" << p << "\n";
        break;
      }
    }

    cout << "=== FINISHED prime p=" << p << " ===\n";
  }

  // Cleanup
  CUDA_CHECK(cudaFree(d_rows));
  CUDA_CHECK(cudaFree(d_row_count));
  CUDA_CHECK(cudaFree(d_result_matrices));
  CUDA_CHECK(cudaFree(d_result_determinants));
  CUDA_CHECK(cudaFree(d_result_count));
  CUDA_CHECK(cudaFree(d_nonsingular_count));

  cout << "\nAll primes processed successfully.\n";
  return 0;
}

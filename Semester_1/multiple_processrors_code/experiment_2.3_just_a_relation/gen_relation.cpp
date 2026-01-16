#include <NTL/mat_ZZ_p.h>
#include <NTL/vector.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <mpi.h>
#include <experimental/filesystem>
#include <string>

using namespace std;
using namespace NTL;
namespace fs = std::experimental::filesystem;

// Check if new_row has a conflict with current_matrix
bool has_conflict(const vector<Vec<ZZ_p>>& current_matrix, const Vec<ZZ_p>& new_row) {
    for (const auto& row : current_matrix)
        for (long i = 0; i < row.length(); i++)
            for (long j = 0; j < new_row.length(); j++)
                if (row[i] == new_row[j]) return true;
                return false;
}

// Check if all elements in matrix are unique
bool all_elements_unique(const Mat<ZZ_p>& mat) {
    long n = mat.NumRows();
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            for (long k = 0; k < n; k++)
                for (long l = 0; l < n; l++)
                    if ((i != k || j != l) && mat[i][j] == mat[k][l])
                        return false;
    return true;
}

// Generate all valid rows recursively (no MPI calls inside)
void generate_rows_recursive(long n, long p, long target_sum, long start,
                             vector<long>& current, vector<Vec<ZZ_p>>& rows) {
    if (current.size() == n) {
        long sum = 0;
        for (auto x : current) sum += x;
        if (sum == target_sum) {
            Vec<ZZ_p> row;
            row.SetLength(n);
            for (long i = 0; i < n; i++) row[i] = conv<ZZ_p>(current[i]);
            rows.push_back(row);
        }
        return;
    }

    for (long i = start; i <= p - 1; i++) {
        bool used = false;
        for (auto x : current) if (x == i) { used = true; break; }
        if (used) continue;

        current.push_back(i);
        generate_rows_recursive(n, p, target_sum, i + 1, current, rows);
        current.pop_back();
    }
                             }

                             // Generate one valid matrix recursively (no MPI calls inside)
                             bool generate_one_matrix(long n, const vector<Vec<ZZ_p>>& rows, vector<Vec<ZZ_p>>& current_matrix,
                                                      vector<bool>& used, Mat<ZZ_p>& result_mat, bool& is_singular) {
                                 if (current_matrix.size() == n) {
                                     Mat<ZZ_p> mat;
                                     mat.SetDims(n, n);
                                     for (long i = 0; i < n; i++)
                                         for (long j = 0; j < n; j++)
                                             mat[i][j] = current_matrix[i][j];

                                     if (all_elements_unique(mat)) {
                                         ZZ_p det = determinant(mat);
                                         result_mat = mat;
                                         is_singular = (det == 0);
                                         return true;
                                     }
                                     return false;
                                 }

                                 for (size_t i = 0; i < rows.size(); i++) {
                                     if (!used[i] && !has_conflict(current_matrix, rows[i])) {
                                         used[i] = true;
                                         current_matrix.push_back(rows[i]);

                                         if (generate_one_matrix(n, rows, current_matrix, used, result_mat, is_singular))
                                             return true;

                                         current_matrix.pop_back();
                                         used[i] = false;
                                     }
                                 }
                                 return false;
                                                      }

                                                      int main(int argc, char** argv) {
                                                          if (argc < 2) {
                                                              cerr << "Usage: " << argv[0] << " <primes_file.txt>\n";
                                                              return 1;
                                                          }

                                                          MPI_Init(&argc, &argv);
                                                          int rank, size;
                                                          MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                                                          MPI_Comm_size(MPI_COMM_WORLD, &size);

                                                          vector<long> primes;
                                                          if (rank == 0) {
                                                              ifstream in(argv[1]);
                                                              if (!in) {
                                                                  cerr << "Failed to open primes file\n";
                                                                  MPI_Abort(MPI_COMM_WORLD, 1);
                                                              }
                                                              long val;
                                                              while (in >> val) primes.push_back(val);
                                                          }

                                                          long prime_count = primes.size();
                                                          MPI_Bcast(&prime_count, 1, MPI_LONG, 0, MPI_COMM_WORLD);
                                                          if (rank != 0) primes.resize(prime_count);
                                                          if (prime_count > 0) MPI_Bcast(primes.data(), prime_count, MPI_LONG, 0, MPI_COMM_WORLD);

                                                          for (long pi = 0; pi < prime_count; ++pi) {
                                                              long p = primes[pi];
                                                              if (rank == 0) cout << "\n=== STARTING prime p=" << p << " (index " << pi << ") ===\n";

                                                              ZZ_p::init(ZZ(p));
                                                              string pdir = "p_" + to_string(p);
                                                              if (rank == 0) fs::create_directories(pdir);
                                                              MPI_Barrier(MPI_COMM_WORLD);

                                                              for (long n = 2; n <= p - 1; n++) {
                                                                  auto start_time = chrono::high_resolution_clock::now();

                                                                  vector<Vec<ZZ_p>> rows;
                                                                  vector<long> current;
                                                                  generate_rows_recursive(n, p, p - 1, 1, current, rows);

                                                                  if (rank == 0) {
                                                                      cout << "p=" << p << " n=" << n << " -> rows generated: " << rows.size() << "\n";

                                                                      string rows_path = pdir + "/rows_n_" + to_string(n) + ".txt";
                                                                      ofstream row_file(rows_path);
                                                                      row_file << "Rows generated: " << rows.size() << "\n";
                                                                      for (size_t i = 0; i < rows.size(); i++) {
                                                                          row_file << "Row " << i + 1 << ": ";
                                                                          for (long j = 0; j < rows[i].length(); j++) row_file << rows[i][j] << " ";
                                                                          row_file << "\n";
                                                                      }
                                                                      row_file.close();
                                                                  }

                                                                  if (rows.empty()) break;

                                                                  // Search for up to 5 NON-SINGULAR matrices ONLY
                                                                  vector<Mat<ZZ_p>> local_nonsingular_matrices;
                                                                  long local_attempts = 0;

                                                                  if (rank == 0) {
                                                                      cout << "  [Searching for up to 5 NON-SINGULAR matrices]\n";
                                                                  }

                                                                  // Keep searching through all starting rows
                                                                  for (size_t t = rank; t < rows.size(); t += size) {
                                                                      if (local_nonsingular_matrices.size() >= 5) break; // Found 5 non-singular, stop

                                                                      vector<Vec<ZZ_p>> current_matrix;
                                                                      vector<bool> used(rows.size(), false);
                                                                      current_matrix.push_back(rows[t]);
                                                                      used[t] = true;

                                                                      Mat<ZZ_p> temp_mat;
                                                                      bool temp_singular = false;

                                                                      if (generate_one_matrix(n, rows, current_matrix, used, temp_mat, temp_singular)) {
                                                                          local_attempts++;

                                                                          if (!temp_singular) {
                                                                              local_nonsingular_matrices.push_back(temp_mat);
                                                                              if (local_nonsingular_matrices.size() <= 5) {
                                                                                  cout << "  [Rank " << rank << " found NON-SINGULAR #" << local_nonsingular_matrices.size() << " at attempt " << local_attempts << "]\n";
                                                                              }
                                                                          }
                                                                          // Skip singular matrices completely
                                                                      }
                                                                  }

                                                                  // Gather counts from all ranks
                                                                  long local_nonsingular_count = local_nonsingular_matrices.size();

                                                                  vector<long> all_nonsingular_counts(size);

                                                                  MPI_Gather(&local_nonsingular_count, 1, MPI_LONG, all_nonsingular_counts.data(), 1, MPI_LONG, 0, MPI_COMM_WORLD);

                                                                  // Collect all non-singular matrices to rank 0
                                                                  vector<Mat<ZZ_p>> all_nonsingular_matrices;

                                                                  if (rank == 0) {
                                                                      // Add rank 0's matrices
                                                                      all_nonsingular_matrices.insert(all_nonsingular_matrices.end(),
                                                                                                      local_nonsingular_matrices.begin(),
                                                                                                      local_nonsingular_matrices.end());

                                                                      // Receive from other ranks
                                                                      for (int r = 1; r < size; r++) {
                                                                          // Receive non-singular matrices
                                                                          for (long m = 0; m < all_nonsingular_counts[r]; m++) {
                                                                              vector<long> mat_data(n * n);
                                                                              MPI_Recv(mat_data.data(), n * n, MPI_LONG, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                                                                              Mat<ZZ_p> received_mat;
                                                                              received_mat.SetDims(n, n);
                                                                              for (long i = 0; i < n; i++)
                                                                                  for (long j = 0; j < n; j++)
                                                                                      received_mat[i][j] = conv<ZZ_p>(mat_data[i * n + j]);
                                                                              all_nonsingular_matrices.push_back(received_mat);
                                                                          }
                                                                      }
                                                                  } else {
                                                                      // Send non-singular matrices to rank 0
                                                                      for (const auto& mat : local_nonsingular_matrices) {
                                                                          vector<long> mat_data(n * n);
                                                                          for (long i = 0; i < n; i++)
                                                                              for (long j = 0; j < n; j++)
                                                                                  mat_data[i * n + j] = to_long(rep(mat[i][j]));
                                                                          MPI_Send(mat_data.data(), n * n, MPI_LONG, 0, 0, MPI_COMM_WORLD);
                                                                      }
                                                                  }

                                                                  auto end_time = chrono::high_resolution_clock::now();
                                                                  auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

                                                                  // Broadcast whether we found any non-singular matrices
                                                                  int found_flag = 0;
                                                                  if (rank == 0) {
                                                                      found_flag = (!all_nonsingular_matrices.empty()) ? 1 : 0;
                                                                  }
                                                                  MPI_Bcast(&found_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
                                                                  bool found = (found_flag == 1);

                                                                  if (rank == 0) {
                                                                      string matrix_path = pdir + "/matrices_n_" + to_string(n) + ".txt";
                                                                      ofstream matrix_file(matrix_path);

                                                                      if (!all_nonsingular_matrices.empty()) {
                                                                          matrix_file << "=== NON-SINGULAR MATRICES (found " << all_nonsingular_matrices.size() << ") ===\n\n";

                                                                          long count = min((long)all_nonsingular_matrices.size(), 5L);
                                                                          for (long m = 0; m < count; m++) {
                                                                              ZZ_p det = determinant(all_nonsingular_matrices[m]);
                                                                              matrix_file << "Matrix #" << (m + 1) << " | n=" << n << " | Determinant=" << det << " | Non-Singular\n";
                                                                              for (long i = 0; i < all_nonsingular_matrices[m].NumRows(); i++) {
                                                                                  for (long j = 0; j < all_nonsingular_matrices[m].NumCols(); j++)
                                                                                      matrix_file << all_nonsingular_matrices[m][i][j] << " ";
                                                                                  matrix_file << "\n";
                                                                              }
                                                                              matrix_file << "\n";
                                                                          }
                                                                      } else {
                                                                          matrix_file << "No non-singular matrix found for n=" << n << "\n\n";
                                                                      }

                                                                      matrix_file << "=== SUMMARY FOR n=" << n << " ===\n";
                                                                      matrix_file << "Non-Singular matrices found: " << all_nonsingular_matrices.size() << "\n";
                                                                      matrix_file << "Computation time (ms): " << duration.count() << "\n";
                                                                      matrix_file << "============================\n";
                                                                      matrix_file.close();

                                                                      string summary_path = pdir + "/summary_n_" + to_string(n) + ".txt";
                                                                      ofstream summary_file(summary_path);
                                                                      summary_file << "=== SUMMARY FOR p=" << p << " n=" << n << " ===\n";
                                                                      summary_file << "Non-Singular matrices found: " << all_nonsingular_matrices.size() << "\n";
                                                                      summary_file << "Computation time (ms): " << duration.count() << "\n";
                                                                      summary_file << "============================\n";
                                                                      summary_file.close();

                                                                      cout << "p=" << p << " n=" << n << " done. nonsingular=" << all_nonsingular_matrices.size()
                                                                      << " time(ms)=" << duration.count() << "\n";
                                                                  }

                                                                  MPI_Barrier(MPI_COMM_WORLD);

                                                                  if (!found) {
                                                                      if (rank == 0) cout << "No matrix found for n=" << n << ", stopping further n values for p=" << p << "\n";
                                                                      break;
                                                                  }
                                                              }

                                                              if (rank == 0) cout << "=== FINISHED prime p=" << p << " ===\n";
                                                              MPI_Barrier(MPI_COMM_WORLD);
                                                          }

                                                          MPI_Finalize();
                                                          return 0;
                                                      }


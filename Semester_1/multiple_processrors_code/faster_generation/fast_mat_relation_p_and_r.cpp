#include <NTL/mat_ZZ_p.h>
#include <NTL/vector.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <mpi.h>
#include <experimental/filesystem>
#include <string>
#include <cstdint>
#include <functional>
#include <cmath>

using namespace std;
using namespace NTL;
namespace fs = std::experimental::filesystem;
using u64 = uint64_t;
using Clock = chrono::high_resolution_clock;

// =============== BITSET UTILITIES ===============
struct Row {
    vector<u64> mask;      // bitmask for fast conflict checking
    Vec<ZZ_p> elems;       // actual NTL vector for matrix building
};

inline void set_bit(vector<u64>& mask, int idx) {
    mask[idx >> 6] |= (1ULL << (idx & 63));
}

inline bool test_bit(const vector<u64>& mask, int idx) {
    return (mask[idx >> 6] >> (idx & 63)) & 1ULL;
}

inline bool intersects(const vector<u64>& a, const vector<u64>& b) {
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] & b[i]) return true;
        return false;
}

inline void or_into(vector<u64>& a, const vector<u64>& b) {
    for (size_t i = 0; i < a.size(); ++i) a[i] |= b[i];
}

inline int count_bits(const vector<u64>& mask) {
    int cnt = 0;
    for (u64 word : mask) cnt += __builtin_popcountll(word);
    return cnt;
}

// =============== OPTIMIZED ROW GENERATION ===============
void generate_rows_optimized(long n, long p, vector<Row>& out_rows) {
    vector<long> pool;
    for (long i = 1; i <= p - 1; ++i) pool.push_back(i);

    int W = (p + 63) / 64;  // number of 64-bit words needed
    vector<long> current;
    current.reserve(n);

    // Precompute prefix sums for pruning
    vector<long> prefix_sum(p);
    for (long i = 0; i < p - 1; ++i) {
        prefix_sum[i] = (i == 0) ? pool[i] : prefix_sum[i-1] + pool[i];
    }

    long target = p - 1;  // Sum must equal exactly p - 1 (not modulo)

    function<void(int, int, long)> dfs = [&](int start, int depth, long cur_sum) {
        if (depth == n) {
            if (cur_sum == target) {  // FIXED: exact equality, not modulo
                Row r;
                r.mask.assign(W, 0ULL);
                r.elems.SetLength(n);

                for (int i = 0; i < n; ++i) {
                    r.elems[i] = conv<ZZ_p>(current[i]);
                    set_bit(r.mask, current[i] - 1);  // 0-based indexing
                }
                out_rows.push_back(move(r));
            }
            return;
        }

        int needed = n - depth;
        int available = (int)pool.size() - start;

        if (available < needed) return;  // not enough elements left

        for (int i = start; i <= (int)pool.size() - needed; ++i) {
            // Pruning: check if we can reach target sum
            long min_add = 0, max_add = 0;
            for (int j = 0; j < needed; ++j) {
                min_add += pool[i + j];
                max_add += pool[pool.size() - 1 - j];
            }

            long min_total = cur_sum + min_add;
            long max_total = cur_sum + max_add;

            // Check if target is reachable modulo p
            bool reachable = false;
            if (max_total - min_total >= p) {
                reachable = true;  // range spans multiple periods
            } else {
                long min_mod = min_total % p;
                long max_mod = max_total % p;
                long target_mod = target % p;

                if (min_mod <= max_mod) {
                    reachable = (target_mod >= min_mod && target_mod <= max_mod);
                } else {
                    reachable = (target_mod >= min_mod || target_mod <= max_mod);
                }
            }

            if (!reachable) continue;

            current.push_back(pool[i]);
            dfs(i + 1, depth + 1, cur_sum + pool[i]);
            current.pop_back();
        }
    };

    dfs(0, 0, 0);
}

// =============== FAST PACKING SEARCH ===============
bool find_packing(const vector<Row>& rows, int n, int start_idx,
                  vector<int>& solution, vector<u64>& used_mask) {
    if (start_idx < 0 || start_idx >= (int)rows.size()) return false;

    // Choose first row
    or_into(used_mask, rows[start_idx].mask);
    solution.push_back(start_idx);

    function<bool(int, int)> dfs = [&](int depth, int last_idx) -> bool {
        if (depth == n) return true;

        // Try rows in order (this avoids counting permutations)
        for (int idx = last_idx + 1; idx < (int)rows.size(); ++idx) {
            if (!intersects(rows[idx].mask, used_mask)) {
                vector<u64> old_mask = used_mask;
                or_into(used_mask, rows[idx].mask);
                solution.push_back(idx);

                if (dfs(depth + 1, idx)) return true;

                solution.pop_back();
                used_mask = move(old_mask);
            }
        }
        return false;
    };

    return dfs(1, start_idx);
                  }

                  // =============== PARALLEL PACKING SEARCH ===============
                  bool parallel_find_packing(const vector<Row>& rows, int n, MPI_Comm comm,
                                             vector<int>& solution_out, double time_limit = 120.0) {
                      int rank, size;
                      MPI_Comm_rank(comm, &rank);
                      MPI_Comm_size(comm, &size);

                      int R = (int)rows.size();
                      if (R < n) return false;

                      int W = rows.empty() ? 0 : (int)rows[0].mask.size();
                      auto tstart = Clock::now();

                      bool local_found = false;

                      // Each rank tries different starting rows
                      for (int start_idx = rank; start_idx < R && !local_found; start_idx += size) {
                          auto now = Clock::now();
                          double elapsed = chrono::duration<double>(now - tstart).count();
                          if (elapsed > time_limit) break;

                          vector<u64> used_mask(W, 0ULL);
                          vector<int> sol;

                          if (find_packing(rows, n, start_idx, sol, used_mask)) {
                              solution_out = sol;
                              local_found = true;
                              break;
                          }

                          // Periodic check if anyone found solution
                          if ((start_idx / size) % 50 == 0) {
                              int any_found = local_found ? 1 : 0;
                              int global_any = 0;
                              MPI_Allreduce(&any_found, &global_any, 1, MPI_INT, MPI_LOR, comm);
                              if (global_any) return true;
                          }
                      }

                      int local_flag = local_found ? 1 : 0;
                      int global_flag = 0;
                      MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_LOR, comm);

                      return global_flag != 0;
                                             }

                                             // =============== BINARY SEARCH FOR MAX D ===============
                                             int find_max_d(long p, MPI_Comm comm, double time_per_n = 60.0) {
                                                 int rank;
                                                 MPI_Comm_rank(comm, &rank);

                                                 int ub = (int)floor(sqrt((double)(p - 1)));
                                                 int lo = 2, hi = ub;
                                                 int best = 1;

                                                 while (lo <= hi) {
                                                     int mid = (lo + hi) / 2;

                                                     if (rank == 0) {
                                                         cout << "  [Binary Search] Testing n=" << mid
                                                         << " (range: [" << lo << ", " << hi << "])" << endl;
                                                     }

                                                     vector<Row> rows;
                                                     auto t0 = Clock::now();
                                                     generate_rows_optimized(mid, p, rows);
                                                     auto t1 = Clock::now();
                                                     double gen_time = chrono::duration<double>(t1 - t0).count();

                                                     if (rank == 0) {
                                                         cout << "    Generated " << rows.size() << " rows in "
                                                         << gen_time << "s" << endl;
                                                     }

                                                     bool feasible = false;
                                                     if (!rows.empty() && (int)rows.size() >= mid) {
                                                         vector<int> sol;
                                                         feasible = parallel_find_packing(rows, mid, comm, sol, time_per_n);
                                                     }

                                                     if (rank == 0) {
                                                         cout << "    Result: " << (feasible ? "FEASIBLE" : "NOT FEASIBLE") << endl;
                                                     }

                                                     if (feasible) {
                                                         best = mid;
                                                         lo = mid + 1;
                                                     } else {
                                                         hi = mid - 1;
                                                     }
                                                 }

                                                 return best;
                                             }

                                             // =============== BUILD AND ANALYZE MATRICES ===============
                                             void generate_and_analyze_matrices(long n, const vector<Row>& rows,
                                                                                int rank, int size,
                                                                                const string& pdir,
                                                                                long& local_singular,
                                                                                long& local_non_singular) {
                                                 string filename = pdir + "/matrices_n_" + to_string(n) +
                                                 "_rank_" + to_string(rank) + ".txt";
                                                 ofstream matrix_file(filename);

                                                 int W = rows.empty() ? 0 : (int)rows[0].mask.size();

                                                 // Each rank processes different starting rows
                                                 for (int start_idx = rank; start_idx < (int)rows.size(); start_idx += size) {
                                                     vector<u64> used_mask(W, 0ULL);
                                                     vector<int> solution;

                                                     if (find_packing(rows, n, start_idx, solution, used_mask)) {
                                                         // Build matrix from solution
                                                         Mat<ZZ_p> mat;
                                                         mat.SetDims(n, n);

                                                         for (int i = 0; i < n; ++i) {
                                                             for (int j = 0; j < n; ++j) {
                                                                 mat[i][j] = rows[solution[i]].elems[j];
                                                             }
                                                         }

                                                         ZZ_p det = determinant(mat);

                                                         matrix_file << "n=" << n << " | Determinant=" << det
                                                         << " | " << (det == 0 ? "Singular" : "Non-Singular") << "\n";

                                                         for (long i = 0; i < n; ++i) {
                                                             for (long j = 0; j < n; ++j) {
                                                                 matrix_file << mat[i][j] << " ";
                                                             }
                                                             matrix_file << "\n";
                                                         }
                                                         matrix_file << "\n";

                                                         if (det == 0) local_singular++;
                                                         else local_non_singular++;
                                                     }
                                                 }

                                                 matrix_file.close();
                                                                                }

                                                                                // =============== MAIN ===============
                                                                                int main(int argc, char** argv) {
                                                                                    if (argc < 2) {
                                                                                        cerr << "Usage: " << argv[0] << " <primes_file.txt> [--binary-search]\n";
                                                                                        return 1;
                                                                                    }

                                                                                    MPI_Init(&argc, &argv);
                                                                                    int rank, size;
                                                                                    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                                                                                    MPI_Comm_size(MPI_COMM_WORLD, &size);

                                                                                    bool use_binary_search = false;
                                                                                    for (int i = 2; i < argc; ++i) {
                                                                                        if (string(argv[i]) == "--binary-search") {
                                                                                            use_binary_search = true;
                                                                                        }
                                                                                    }

                                                                                    vector<long> primes;

                                                                                    if (rank == 0) {
                                                                                        ifstream in(argv[1]);
                                                                                        if (!in) {
                                                                                            cerr << "Failed to open primes file: " << argv[1] << "\n";
                                                                                            MPI_Abort(MPI_COMM_WORLD, 1);
                                                                                        }
                                                                                        long val;
                                                                                        while (in >> val) primes.push_back(val);
                                                                                        in.close();

                                                                                        if (primes.empty()) {
                                                                                            cerr << "No primes found in file.\n";
                                                                                            MPI_Abort(MPI_COMM_WORLD, 1);
                                                                                        }
                                                                                    }

                                                                                    long prime_count = primes.size();
                                                                                    MPI_Bcast(&prime_count, 1, MPI_LONG, 0, MPI_COMM_WORLD);
                                                                                    if (rank != 0) primes.resize(prime_count);
                                                                                    if (prime_count > 0) {
                                                                                        MPI_Bcast(primes.data(), prime_count, MPI_LONG, 0, MPI_COMM_WORLD);
                                                                                    }

                                                                                    for (long pi = 0; pi < prime_count; ++pi) {
                                                                                        long p = primes[pi];

                                                                                        if (rank == 0) {
                                                                                            cout << "\n=== STARTING prime p=" << p << " ===\n";
                                                                                        }

                                                                                        ZZ_p::init(ZZ(p));

                                                                                        string pdir = "p_" + to_string(p);
                                                                                        if (rank == 0) fs::create_directories(pdir);
                                                                                        MPI_Barrier(MPI_COMM_WORLD);

                                                                                        if (use_binary_search) {
                                                                                            // Fast mode: just find maximum d
                                                                                            int max_d = find_max_d(p, MPI_COMM_WORLD, 120.0);

                                                                                            if (rank == 0) {
                                                                                                cout << "Maximum dimension for p=" << p << " is d=" << max_d << "\n";

                                                                                                ofstream summary(pdir + "/max_dimension.txt");
                                                                                                summary << "p = " << p << "\n";
                                                                                                summary << "max_d = " << max_d << "\n";
                                                                                                summary << "log2(p) = " << log2(p) << "\n";
                                                                                                summary << "sqrt(p-1) = " << sqrt(p-1) << "\n";
                                                                                                summary.close();
                                                                                            }
                                                                                        } else {
                                                                                            // Detailed mode: generate all matrices for each n
                                                                                            for (long n = 2; n <= min(p - 1, (long)floor(sqrt(p - 1))); ++n) {
                                                                                                auto start_time = Clock::now();

                                                                                                vector<Row> rows;
                                                                                                generate_rows_optimized(n, p, rows);

                                                                                                if (rows.empty()) {
                                                                                                    if (rank == 0) {
                                                                                                        cout << "No valid rows for n=" << n << ". Stopping.\n";
                                                                                                    }
                                                                                                    break;
                                                                                                }

                                                                                                if (rank == 0) {
                                                                                                    cout << "n=" << n << ": generated " << rows.size() << " rows\n";
                                                                                                }

                                                                                                long local_singular = 0, local_non_singular = 0;
                                                                                                generate_and_analyze_matrices(n, rows, rank, size, pdir,
                                                                                                                              local_singular, local_non_singular);

                                                                                                long total_singular = 0, total_non_singular = 0;
                                                                                                MPI_Reduce(&local_singular, &total_singular, 1, MPI_LONG,
                                                                                                           MPI_SUM, 0, MPI_COMM_WORLD);
                                                                                                MPI_Reduce(&local_non_singular, &total_non_singular, 1, MPI_LONG,
                                                                                                           MPI_SUM, 0, MPI_COMM_WORLD);

                                                                                                if (rank == 0) {
                                                                                                    auto end_time = Clock::now();
                                                                                                    auto duration = chrono::duration_cast<chrono::milliseconds>(
                                                                                                        end_time - start_time);

                                                                                                    // Merge rank files
                                                                                                    string final_path = pdir + "/matrices_n_" + to_string(n) + ".txt";
                                                                                                    ofstream final_file(final_path);

                                                                                                    for (int r = 0; r < size; ++r) {
                                                                                                        string temp_file = pdir + "/matrices_n_" + to_string(n) +
                                                                                                        "_rank_" + to_string(r) + ".txt";
                                                                                                        ifstream in(temp_file);
                                                                                                        if (in) {
                                                                                                            final_file << in.rdbuf();
                                                                                                            in.close();
                                                                                                            fs::remove(temp_file);
                                                                                                        }
                                                                                                    }

                                                                                                    final_file << "\n=== SUMMARY ===\n";
                                                                                                    final_file << "Total matrices: "
                                                                                                    << (total_singular + total_non_singular) << "\n";
                                                                                                    final_file << "Singular: " << total_singular << "\n";
                                                                                                    final_file << "Non-Singular: " << total_non_singular << "\n";
                                                                                                    final_file << "Time (ms): " << duration.count() << "\n";
                                                                                                    final_file.close();

                                                                                                    cout << "p=" << p << " n=" << n << " complete. "
                                                                                                    << "Matrices=" << (total_singular + total_non_singular)
                                                                                                    << " Time=" << duration.count() << "ms\n";

                                                                                                    if (total_singular + total_non_singular == 0) {
                                                                                                        cout << "No matrices found. Stopping at n=" << n << "\n";
                                                                                                        break;
                                                                                                    }
                                                                                                }

                                                                                                MPI_Barrier(MPI_COMM_WORLD);
                                                                                            }
                                                                                        }

                                                                                        if (rank == 0) {
                                                                                            cout << "=== FINISHED prime p=" << p << " ===\n";
                                                                                        }
                                                                                        MPI_Barrier(MPI_COMM_WORLD);
                                                                                    }

                                                                                    MPI_Finalize();
                                                                                    return 0;
                                                                                }

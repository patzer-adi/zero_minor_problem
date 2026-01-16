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
    vector<u64> mask;
    Vec<ZZ_p> elems;
};

inline void set_bit(vector<u64>& mask, int idx) {
    mask[idx >> 6] |= (1ULL << (idx & 63));
}

inline bool intersects(const vector<u64>& a, const vector<u64>& b) {
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] & b[i]) return true;
    return false;
}

inline void or_into(vector<u64>& a, const vector<u64>& b) {
    for (size_t i = 0; i < a.size(); ++i) a[i] |= b[i];
}

// =============== ROW GENERATION ===============
void generate_rows_optimized(long n, long p, vector<Row>& out_rows) {
    vector<long> pool;
    for (long i = 1; i <= p - 1; ++i) pool.push_back(i);

    int W = (p + 63) / 64;
    vector<long> current;
    current.reserve(n);
    long target = p - 1;

    function<void(int, int, long)> dfs = [&](int start, int depth, long cur_sum) {
        if (depth == n) {
            if (cur_sum == target) {
                Row r;
                r.mask.assign(W, 0ULL);
                r.elems.SetLength(n);
                for (int i = 0; i < n; ++i) {
                    r.elems[i] = conv<ZZ_p>(current[i]);
                    set_bit(r.mask, current[i] - 1);
                }
                out_rows.push_back(move(r));
            }
            return;
        }

        int needed = n - depth;
        if ((int)pool.size() - start < needed) return;

        for (int i = start; i <= (int)pool.size() - needed; ++i) {
            long min_add = 0, max_add = 0;
            for (int j = 0; j < needed; ++j) {
                min_add += pool[i + j];
                max_add += pool[pool.size() - 1 - j];
            }

            if (target < cur_sum + min_add || target > cur_sum + max_add) continue;

            current.push_back(pool[i]);
            dfs(i + 1, depth + 1, cur_sum + pool[i]);
            current.pop_back();
        }
    };

    dfs(0, 0, 0);
}

// =============== GENERATE ALL MATRICES (EXHAUSTIVE) ===============
void generate_all_matrices(long n, const vector<Row>& rows,
                          int rank, int size,
                          const string& pdir,
                          long& local_singular,
                          long& local_non_singular,
                          vector<Mat<ZZ_p>>& samples,
                          int max_samples = 5) {
    string filename = pdir + "/matrices_n_" + to_string(n) +
                      "_rank_" + to_string(rank) + ".txt";
    ofstream matrix_file(filename);

    int W = rows.empty() ? 0 : (int)rows[0].mask.size();
    int samples_collected = 0;

    vector<int> current_indices;
    vector<u64> used_mask(W, 0ULL);

    // Recursive function to explore ALL valid combinations
    function<void(int, int)> explore_all = [&](int depth, int last_idx) {
        if (depth == n) {
            Mat<ZZ_p> mat;
            mat.SetDims(n, n);

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    mat[i][j] = rows[current_indices[i]].elems[j];
                }
            }

            ZZ_p det = determinant(mat);

            // Only write non-singular matrices
            if (det != 0) {
                matrix_file << "n=" << n << " | Determinant=" << det << " | Non-Singular\n";

                for (long i = 0; i < n; ++i) {
                    for (long j = 0; j < n; ++j) {
                        matrix_file << mat[i][j] << " ";
                    }
                    matrix_file << "\n";
                }
                matrix_file << "\n";

                local_non_singular++;
                if (samples_collected < max_samples) {
                    samples.push_back(mat);
                    samples_collected++;
                }
            } else {
                local_singular++;
            }
            return;
        }

        for (int idx = last_idx + 1; idx < (int)rows.size(); ++idx) {
            if (!intersects(rows[idx].mask, used_mask)) {
                vector<u64> old_mask = used_mask;
                or_into(used_mask, rows[idx].mask);
                current_indices.push_back(idx);

                explore_all(depth + 1, idx);

                current_indices.pop_back();
                used_mask = move(old_mask);
            }
        }
    };

    // Distribute starting rows across ranks
    for (int start_idx = rank; start_idx < (int)rows.size(); start_idx += size) {
        used_mask.assign(W, 0ULL);
        current_indices.clear();

        or_into(used_mask, rows[start_idx].mask);
        current_indices.push_back(start_idx);

        explore_all(1, start_idx);
    }

    matrix_file.close();
}

// =============== BINARY SEARCH (ONE MATRIX PER TEST) ===============
bool find_one_packing(const vector<Row>& rows, int n, int start_idx,
                     vector<int>& solution, vector<u64>& used_mask) {
    if (start_idx < 0 || start_idx >= (int)rows.size()) return false;

    or_into(used_mask, rows[start_idx].mask);
    solution.push_back(start_idx);

    function<bool(int, int)> dfs = [&](int depth, int last_idx) -> bool {
        if (depth == n) return true;

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

    for (int start_idx = rank; start_idx < R && !local_found; start_idx += size) {
        auto now = Clock::now();
        if (chrono::duration<double>(now - tstart).count() > time_limit) break;

        vector<u64> used_mask(W, 0ULL);
        vector<int> sol;

        if (find_one_packing(rows, n, start_idx, sol, used_mask)) {
            solution_out = sol;
            local_found = true;
            break;
        }

        if ((start_idx / size) % 50 == 0) {
            int any = local_found ? 1 : 0, global = 0;
            MPI_Allreduce(&any, &global, 1, MPI_INT, MPI_LOR, comm);
            if (global) return true;
        }
    }

    int local_flag = local_found ? 1 : 0, global_flag = 0;
    MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_LOR, comm);
    return global_flag != 0;
}

int find_max_d(long p, MPI_Comm comm, double time_per_n = 60.0) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int ub = (int)floor(sqrt((double)(p - 1)));
    int lo = 2, hi = ub, best = 1;

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

        if (rank == 0) {
            cout << "    Generated " << rows.size() << " rows in "
                 << chrono::duration<double>(t1 - t0).count() << "s" << endl;
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
            cerr << "Failed to open: " << argv[1] << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        long val;
        while (in >> val) primes.push_back(val);
        in.close();

        if (primes.empty()) {
            cerr << "No primes found\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        cout << "Running with " << size << " MPI ranks\n";
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
            // DETAILED MODE: Generate ALL matrices
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
                vector<Mat<ZZ_p>> local_samples;

                // Generate ALL matrices
                generate_all_matrices(n, rows, rank, size, pdir,
                                    local_singular, local_non_singular,
                                    local_samples, 5);

                long total_singular = 0, total_non_singular = 0;
                MPI_Reduce(&local_singular, &total_singular, 1, MPI_LONG,
                          MPI_SUM, 0, MPI_COMM_WORLD);
                MPI_Reduce(&local_non_singular, &total_non_singular, 1, MPI_LONG,
                          MPI_SUM, 0, MPI_COMM_WORLD);

                if (rank == 0) {
                    auto end_time = Clock::now();
                    auto duration = chrono::duration_cast<chrono::milliseconds>(
                        end_time - start_time);

                    // Save samples
                    string samples_path = pdir + "/sample_nonsingular_n_" + to_string(n) + ".txt";
                    ofstream samples_file(samples_path);
                    samples_file << "=== SAMPLE NON-SINGULAR MATRICES ===\n";
                    samples_file << "p=" << p << ", n=" << n << "\n\n";

                    for (size_t s = 0; s < local_samples.size() && s < 5; ++s) {
                        samples_file << "Sample " << (s + 1) << " (rank 0):\n";
                        ZZ_p det = determinant(local_samples[s]);
                        samples_file << "Determinant: " << det << "\n";
                        for (long i = 0; i < n; ++i) {
                            for (long j = 0; j < n; ++j) {
                                samples_file << local_samples[s][i][j] << " ";
                            }
                            samples_file << "\n";
                        }
                        samples_file << "\n";
                    }

                    // Receive samples from other ranks
                    for (int r = 1; r < size; ++r) {
                        int num_samples = 0;
                        MPI_Recv(&num_samples, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                        for (int s = 0; s < num_samples; ++s) {
                            long mat_n = 0;
                            MPI_Recv(&mat_n, 1, MPI_LONG, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                            vector<long> mat_data(mat_n * mat_n);
                            MPI_Recv(mat_data.data(), mat_n * mat_n, MPI_LONG, r, 2,
                                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                            Mat<ZZ_p> mat;
                            mat.SetDims(mat_n, mat_n);
                            for (long i = 0; i < mat_n; ++i) {
                                for (long j = 0; j < mat_n; ++j) {
                                    mat[i][j] = conv<ZZ_p>(mat_data[i * mat_n + j]);
                                }
                            }

                            samples_file << "Sample " << (s + 1) << " (rank " << r << "):\n";
                            ZZ_p det = determinant(mat);
                            samples_file << "Determinant: " << det << "\n";
                            for (long i = 0; i < mat_n; ++i) {
                                for (long j = 0; j < mat_n; ++j) {
                                    samples_file << mat[i][j] << " ";
                                }
                                samples_file << "\n";
                            }
                            samples_file << "\n";
                        }
                    }
                    samples_file.close();

                    // Merge files
                    string final_path = pdir + "/matrices_n_" + to_string(n) + ".txt";
                    ofstream final_file(final_path);

                    for (int r = 0; r < size; ++r) {
                        string temp = pdir + "/matrices_n_" + to_string(n) +
                                    "_rank_" + to_string(r) + ".txt";
                        ifstream in(temp);
                        if (in) {
                            final_file << in.rdbuf();
                            in.close();
                            fs::remove(temp);
                        }
                    }

                    final_file << "\n=== SUMMARY ===\n";
                    final_file << "Total matrices: " << (total_singular + total_non_singular) << "\n";
                    final_file << "Singular: " << total_singular << "\n";
                    final_file << "Non-Singular: " << total_non_singular << "\n";
                    final_file << "Time (ms): " << duration.count() << "\n";
                    final_file << "Samples: " << samples_path << "\n";
                    final_file.close();

                    cout << "p=" << p << " n=" << n << " done. "
                         << "Total=" << (total_singular + total_non_singular)
                         << " Singular=" << total_singular
                         << " NonSing=" << total_non_singular
                         << " Time=" << duration.count() << "ms\n";

                    if (total_singular + total_non_singular == 0) {
                        cout << "No matrices. Stopping.\n";
                        break;
                    }
                } else {
                    // Send samples
                    int num = min((int)local_samples.size(), 5);
                    MPI_Send(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

                    for (int s = 0; s < num; ++s) {
                        long mat_n = n;
                        MPI_Send(&mat_n, 1, MPI_LONG, 0, 1, MPI_COMM_WORLD);

                        vector<long> data(n * n);
                        for (long i = 0; i < n; ++i) {
                            for (long j = 0; j < n; ++j) {
                                data[i * n + j] = conv<long>(rep(local_samples[s][i][j]));
                            }
                        }
                        MPI_Send(data.data(), n * n, MPI_LONG, 0, 2, MPI_COMM_WORLD);
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

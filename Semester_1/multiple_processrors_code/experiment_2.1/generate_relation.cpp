// rows_generate_fixed.cpp
// MPI + NTL program to find maximum d for many primes (reads primes file on rank 0).
// C++11-compatible. Writes found matrix (rows in 1..p-1) to disk (rank 0).

#include <NTL/mat_ZZ_p.h>
#include <NTL/ZZ.h>

#include <mpi.h>

#include <sys/stat.h>
#include <sys/types.h>

#include <climits>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdint>

using namespace std;
using namespace NTL;

// Forward declarations
long mod_inverse(long a, long p);

// POSIX directory create
void create_directory(const string &path) {
    mkdir(path.c_str(), 0755);
}

// ---------------- Mask helpers (support p > 64) ----------------
typedef uint64_t Word;
typedef vector<Word> Mask;

static inline Mask make_mask_for_p(size_t p) {
    return Mask((p + 63) / 64, 0ULL);
}
static inline void mask_set(Mask &m, size_t idx) {
    m[idx >> 6] |= (Word(1) << (idx & 63));
}
static inline bool mask_intersect(const Mask &a, const Mask &b) {
    size_t sz = a.size();
    for (size_t i = 0; i < sz; ++i)
        if ((a[i] & b[i]) != 0) return true;
        return false;
}
static inline void mask_or_into(Mask &a, const Mask &b) {
    size_t sz = a.size();
    for (size_t i = 0; i < sz; ++i)
        a[i] |= b[i];
}

// ---------------- Row generation (no artificial cap) ----------------
// Generates all strictly increasing rows of length n with entries in [1..p-1]
// whose sum equals target_sum. Stores them in valid_rows (may be huge).
void generate_rows_with_pruning(long n, long p, long target_sum, long pos,
                                long current_sum, long last_used,
                                vector<long>& current_row,
                                vector<vector<long>>& valid_rows)
{
    if (pos == n) {
        if (current_sum == target_sum) {
            valid_rows.push_back(current_row);
        }
        return;
    }

    long remaining_positions = n - pos;
    // minimum if we pick smallest possible strictly increasing values
    long min_possible = current_sum + remaining_positions * (last_used + 1);
    // maximum achievable by picking largest possible strictly decreasing from (p-1)
    long max_possible = current_sum;
    for (long i = 0; i < remaining_positions; ++i) {
        max_possible += (p - 1 - i);
    }

    if (min_possible > target_sum || max_possible < target_sum) {
        return; // prune
    }

    for (long val = last_used + 1; val < p; ++val) {
        current_row[pos] = val;
        generate_rows_with_pruning(n, p, target_sum, pos + 1,
                                   current_sum + val, val,
                                   current_row, valid_rows);
    }
}

// ---------------- Fast rank (mod p) quick check ----------------
// A simple Gaussian elimination mod p returning whether full rank (n).
// Use int64_t for intermediate multiplications to avoid overflow.
bool is_likely_full_rank(const vector<vector<long>>& matrix, long p) {
    size_t n = matrix.size();
    if (n == 0) return false;
    // copy to mutable mat with values in [0,p-1]
    vector<vector<long>> mat(n, vector<long>(n));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            long v = matrix[i][j] % p;
            if (v < 0) v += p;
            mat[i][j] = v;
        }

        for (size_t col = 0; col < n; ++col) {
            // find pivot at/after row = col
            size_t pivot_row = (size_t)-1;
            for (size_t row = col; row < n; ++row) {
                if (mat[row][col] % p != 0) { pivot_row = row; break; }
            }
            if (pivot_row == (size_t)-1) return false; // rank deficient

            if (pivot_row != col) swap(mat[pivot_row], mat[col]);

            long inv_pivot = mod_inverse(mat[col][col], p);
            if (inv_pivot == 0) return false; // no inverse (shouldn't happen)

            // eliminate below
            for (size_t row = col + 1; row < n; ++row) {
                if (mat[row][col] == 0) continue;
                long factor = (long)((int64_t)mat[row][col] * inv_pivot % p);
                for (size_t j = col; j < n; ++j) {
                    int64_t prod = (int64_t)factor * (int64_t)mat[col][j];
                    long t = (long)(( (int64_t)mat[row][j] - (prod % p) ) % p);
                    if (t < 0) t += p;
                    mat[row][j] = t;
                }
            }
        }
        return true;
}

// ---------------- Modular inverse (extended Euclid) ----------------
long mod_inverse(long a, long p) {
    long t = 0, newt = 1;
    long r = p, newr = a % p;
    if (newr < 0) newr += p;
    while (newr != 0) {
        long q = r / newr;
        long tmp = t - q * newt; t = newt; newt = tmp;
        tmp = r - q * newr; r = newr; newr = tmp;
    }
    if (r != 1) {
        // no inverse (shouldn't happen for prime p except when a==0)
        return 0;
    }
    if (t < 0) t += p;
    return t;
}

// ---------------- DFS search (bitmask pruning) ----------------
// Note: global_found_flag must be updated atomically at higher level.
// We use the convention: set *global_found_flag = 1 to stop others.
bool search_non_singular_dfs(long n, long p,
                             const vector<vector<long>>& rows,
                             const vector<Mask>& masks,
                             vector<int>& current_indices,
                             vector<char>& used_rows,
                             Mask& used_values_mask,
                             volatile int *global_found_flag)
{
    if (*global_found_flag) return false;

    if ((int)current_indices.size() == n) {
        // Build matrix and check quickly then with NTL determinant
        vector<vector<long>> M(n, vector<long>(n));
        for (int i = 0; i < n; ++i) M[i] = rows[current_indices[i]];

        if (!is_likely_full_rank(M, p)) return false;

        // NTL determinant check: uses ZZ_p, so initialize must be called beforehand
        Mat<ZZ_p> A;
        A.SetDims(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                A[i][j] = conv<ZZ_p>(M[i][j]);

        ZZ_p det = determinant(A);
        if (det != 0) {
            *const_cast<int*>( (const int*)global_found_flag ) = 1; // set flag
            return true;
        }
        return false;
    }

    int start_idx = current_indices.empty() ? 0 : current_indices.back() + 1;
    int R = (int)rows.size();

    for (int i = start_idx; i < R; ++i) {
        if (*global_found_flag) return false;
        if (used_rows[i]) continue;
        if (mask_intersect(used_values_mask, masks[i])) continue;

        // Choose row i
        used_rows[i] = 1;
        current_indices.push_back(i);

        Mask old_mask = used_values_mask;
        mask_or_into(used_values_mask, masks[i]);

        if (search_non_singular_dfs(n, p, rows, masks, current_indices, used_rows, used_values_mask, global_found_flag))
            return true;

        // backtrack
        used_values_mask = old_mask;
        current_indices.pop_back();
        used_rows[i] = 0;
    }

    return false;
}

// ---------------- Find max dimension for prime p ----------------
long find_max_dimension_for_prime(long p, int rank, int size, const string &log_dir)
{
    // Must initialize ZZ_p on each process for determinant usage
    if (p <= 1) return 0;
    ZZ_p::init(ZZ(p));

    // Estimate upper bound (as in your earlier code)
    long k = (long)floor(log2((double)p));
    long estimated_max = max(2L, k - 2);
    long upper_bound = min(estimated_max + 2, p - 1);

    if (rank == 0) {
        cout << "  p=" << p << " estimated_max=" << estimated_max << " test from " << upper_bound << " down\n";
    }

    for (long n = upper_bound; n >= 2; --n) {
        auto tstart = chrono::high_resolution_clock::now();

        // generate rows (only on rank 0)
        vector<vector<long>> rows;
        if (rank == 0) {
            vector<long> cur(n);
            long target_sum = p - 1;
            generate_rows_with_pruning(n, p, target_sum, 0, 0, 0, cur, rows);
            cout << "    d=" << n << ": rows generated = " << rows.size() << "\n";
        }

        // Broadcast rows: flatten into contiguous buffer
        int R = (int)rows.size();
        MPI_Bcast(&R, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (R == 0) {
            // no rows for this n, try next smaller n
            MPI_Barrier(MPI_COMM_WORLD);
            continue;
        }

        if (rank != 0) rows.assign(R, vector<long>(n));
        // pack on rank 0 into buffer
        if (rank == 0) {
            vector<long> buf;
            buf.reserve((size_t)R * (size_t)n);
            for (int i = 0; i < R; ++i)
                for (int j = 0; j < n; ++j)
                    buf.push_back(rows[i][j]);

            MPI_Bcast(buf.data(), R * n, MPI_LONG, 0, MPI_COMM_WORLD);
            // rows already available on rank 0
        } else {
            vector<long> recvbuf((size_t)R * (size_t)n);
            MPI_Bcast(recvbuf.data(), R * n, MPI_LONG, 0, MPI_COMM_WORLD);
            for (int i = 0; i < R; ++i) {
                for (int j = 0; j < n; ++j) {
                    rows[i][j] = recvbuf[i * n + j];
                }
            }
        }

        // Build masks
        vector<Mask> masks;
        masks.reserve(R);
        for (int i = 0; i < R; ++i) {
            Mask m = make_mask_for_p((size_t)p);
            for (long v : rows[i]) {
                // rows use values in [1..p-1], convert to zero-based index
                if (v <= 0 || v >= p) continue; // safety
                mask_set(m, (size_t)(v - 1));
            }
            masks.push_back(m);
        }

        // Parallel search: each process tries different starting rows
        volatile int global_found = 0;
        int local_found = 0;
        vector<int> best_indices;

        // distribute starting indices over all ranks
        for (int start = rank; start < R && !global_found; start += size) {
            vector<int> chosen;
            chosen.push_back(start);
            vector<char> used_rows(R, 0);
            used_rows[start] = 1;
            Mask used_values = make_mask_for_p((size_t)p);
            mask_or_into(used_values, masks[start]);

            if (search_non_singular_dfs((int)n, p, rows, masks, chosen, used_rows, used_values, &global_found)) {
                local_found = 1;
                best_indices = chosen; // chosen contains the full path because recursion returns true without backtracking
                break;
            }
        }

        // Allreduce to know if ANY found
        int any_local = local_found ? 1 : 0;
        int any_global = 0;
        MPI_Allreduce(&any_local, &any_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        auto tend = chrono::high_resolution_clock::now();
        auto dur = chrono::duration_cast<chrono::milliseconds>(tend - tstart).count();
        if (rank == 0) {
            cout << "    d=" << n << (any_global ? " FOUND" : " NOT_FOUND") << " time=" << dur << "ms\n";
        }

        if (any_global) {
            // ---- New: determine which rank found the solution ----
            int my_rank_val = local_found ? rank : INT_MAX;
            int winner_rank = INT_MAX;
            MPI_Allreduce(&my_rank_val, &winner_rank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

            // Winner sends its indices to rank 0
            if (rank == winner_rank) {
                int count = (int)best_indices.size();
                MPI_Send(&count, 1, MPI_INT, 0, 100, MPI_COMM_WORLD);
                if (count > 0) {
                    MPI_Send(best_indices.data(), count, MPI_INT, 0, 101, MPI_COMM_WORLD);
                }
            }

            // Rank 0 receives indices and writes matrix using original 1..p-1 values
            if (rank == 0) {
                int count = 0;
                MPI_Recv(&count, 1, MPI_INT, winner_rank, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                vector<int> indices;
                if (count > 0) {
                    indices.resize(count);
                    MPI_Recv(indices.data(), count, MPI_INT, winner_rank, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                // Write matrix file
                string mat_fn = log_dir + "/matrix_d" + to_string(n) + ".txt";
                ofstream mat(mat_fn.c_str());
                mat << "Matrix for p=" << p << ", d=" << n << "\n";
                mat << "Rows (each row has " << n << " entries in range 1.." << (p-1) << "):\n\n";
                for (size_t ii = 0; ii < indices.size(); ++ii) {
                    int ridx = indices[ii];
                    if (ridx < 0 || ridx >= R) continue;
                    for (int j = 0; j < (int)n; ++j) {
                        mat << rows[ridx][j] << (j+1 == (int)n ? "" : " ");
                    }
                    mat << "\n";
                }
                mat.close();

                // also write the summary file
                string outfn = log_dir + "/max_dimension_d" + to_string(n) + ".txt";
                ofstream out(outfn.c_str());
                out << "Prime p=" << p << "\n";
                out << "Maximum dimension d=" << n << "\n";
                out << "Rows generated: " << R << "\n";
                out << "Search time (ms): " << dur << "\n";
                out << "Matrix saved in: " << mat_fn << "\n";
                out.close();
            }

            MPI_Barrier(MPI_COMM_WORLD);
            return n;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    return 0;
}

// ---------------- Main ----------------
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank=0, size=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            cerr << "Usage: mpirun -np <procs> ./rows_generate <primes_file.txt>\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Rank 0 reads primes file.
    vector<long> primes;
    if (rank == 0) {
        ifstream fin(argv[1]);
        if (!fin) {
            cerr << "Failed to open primes file: " << argv[1] << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        long x;
        while (fin >> x) {
            primes.push_back(x);
        }
        fin.close();
        if (primes.empty()) {
            cerr << "No primes found in file: " << argv[1] << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast count and primes
    int prime_count = (int)primes.size();
    MPI_Bcast(&prime_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) primes.resize(prime_count);
    if (prime_count > 0) {
        MPI_Bcast(primes.data(), prime_count, MPI_LONG, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        cout << "Running with " << size << " processes. Loaded " << prime_count << " primes.\n";
        create_directory("max_dim_results");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    vector<pair<long,long> > summary; // only used on rank 0

    // process primes sequentially (you can parallelize across primes later)
    for (size_t idx = 0; idx < primes.size(); ++idx) {
        long p = primes[idx];
        if (rank == 0) {
            cout << "========================================\n";
            cout << "Prime p = " << p << "\n";
            cout << "========================================\n";
        }

        if (p <= 1) {
            if (rank == 0) cerr << "Skipping invalid p=" << p << "\n";
            continue;
        }

        string prime_dir = string("max_dim_results/p") + to_string(p);
        if (rank == 0) create_directory(prime_dir);
        MPI_Barrier(MPI_COMM_WORLD);

        long max_d = find_max_dimension_for_prime((int)p, rank, size, prime_dir);

        if (rank == 0) {
            summary.push_back(make_pair(p, max_d));
            string summary_fn = prime_dir + "/summary.txt";
            ofstream out(summary_fn.c_str());
            out << "Prime: " << p << "\n";
            out << "Maximum dimension: " << max_d << "\n";
            out.close();

            cout << "RESULT: p=" << p << " -> max_d=" << max_d << "\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // final summary write by rank 0
    if (rank == 0) {
        ofstream out("max_dim_results/FINAL_SUMMARY.txt");
        out << "Prime (p) | Max d | log2(p)-2\n";
        out << "-------------------------------\n";
        for (size_t i = 0; i < summary.size(); ++i) {
            long p = summary[i].first;
            long d = summary[i].second;
            out << p << " | " << d << " | " << ( (long)floor(log2((double)p)) - 2 ) << "\n";
        }
        out.close();
        cout << "All done. Results in max_dim_results/\n";
    }

    MPI_Finalize();
    return 0;
}

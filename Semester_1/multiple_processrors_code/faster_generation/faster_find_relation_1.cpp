// relation_binary_with_sample_experimental_fs.cpp
// Build: mpicxx -O2 -std=c++17 -lntl -lgmp -o relation_binary_with_sample_experimental_fs relation_binary_with_sample_experimental_fs.cpp

#include <NTL/mat_ZZ_p.h>
#include <NTL/vector.h>
#include <NTL/ZZ.h>

#include <mpi.h>
#include <experimental/filesystem> // <-- use experimental for old servers
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <functional>
#include <cstdint>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace NTL;
namespace fs = std::experimental::filesystem;
using u64 = uint64_t;

// Row struct
struct Row {
    Vec<ZZ_p> elems;
    vector<u64> mask;
};

// Bit helpers
inline void set_bit(vector<u64>& mask, int idx) { mask[idx >> 6] |= (1ULL << (idx & 63)); }
inline bool intersects(const vector<u64>& a, const vector<u64>& b) {
    for (size_t i = 0; i < a.size(); ++i) if (a[i] & b[i]) return true;
    return false;
}
inline void or_into(vector<u64>& a, const vector<u64>& b) {
    for (size_t i = 0; i < a.size(); ++i) a[i] |= b[i];
}

// -------------------- DFS with rank-pruning (used by each rank) --------------------
// Find a non-singular packing starting with a given start_idx.
// used_mask must be pre-sized to W words; sol returned on success.
bool dfs_find_nonsingular_sample_local(const vector<Row>& rows, int n, int start_idx,
                                       vector<int>& sol, vector<u64>& used_mask,
                                       double time_limit, double tstart) {
    int R = (int)rows.size();
    if (start_idx < 0 || start_idx >= R) return false;
    sol.clear();

    // choose first row
    or_into(used_mask, rows[start_idx].mask);
    sol.push_back(start_idx);

    function<bool(int,int)> dfs = [&](int depth, int last_idx)->bool {
        // time cutoff
        if (MPI_Wtime() - tstart > time_limit) return false;

        // partial rank check
        if (depth > 0) {
            Mat<ZZ_p> small;
            small.SetDims(depth, n);
            for (int r = 0; r < depth; ++r) {
                int ridx = sol[r];
                for (int c = 0; c < n; ++c) small[r][c] = rows[ridx].elems[c];
            }
            long rk = gauss(small); // returns rank
            if (rk < depth) return false; // prune
        }

        if (depth == n) {
            // full candidate (already independent by rank checks) — check det
            Mat<ZZ_p> full;
            full.SetDims(n, n);
            for (int r = 0; r < n; ++r)
                for (int c = 0; c < n; ++c)
                    full[r][c] = rows[sol[r]].elems[c];
            ZZ_p det = determinant(full);
            return det != 0;
        }

        for (int idx = last_idx + 1; idx < R; ++idx) {
            // conflict test (fast)
            bool ok = true;
            for (size_t w = 0; w < used_mask.size(); ++w) {
                if (used_mask[w] & rows[idx].mask[w]) { ok = false; break; }
            }
            if (!ok) continue;

            // inline OR with save/restore (no heavy allocations)
            size_t W = used_mask.size();
            vector<u64> saved(W);
            for (size_t w = 0; w < W; ++w) { saved[w] = used_mask[w]; used_mask[w] |= rows[idx].mask[w]; }

            sol.push_back(idx);
            if (dfs(depth + 1, idx)) return true;
            sol.pop_back();
            for (size_t w = 0; w < W; ++w) used_mask[w] = saved[w];
        }
        return false;
    };

    return dfs(1, start_idx);
}

// -------------------- Parallel search with robust collection --------------------
// Each rank searches different start indices. If any rank finds a solution, we collect one sample to rank 0.
// This function returns true if a non-singular packing was found anywhere.
// If found, rank 0 receives the solution indices via gather and populates sol_out (flattened indices) on rank 0.
bool parallel_find_nonsingular_sample(const vector<Row>& rows, int n, MPI_Comm comm,
                                      vector<int>& sol_out, double time_limit) {
    int rank, size; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);
    int R = (int)rows.size();
    if (R < n) return false;
    int W = rows.empty() ? 0 : (int)rows[0].mask.size();
    double tstart = MPI_Wtime();

    int local_found = 0;
    vector<int> local_sol; local_sol.reserve(n);
    vector<u64> used_mask(W);

    const int CHECK_BATCH = 512;
    int iter = 0;

    // Search loop: each rank tries starts = rank, rank+size, ...
    for (int start = rank; start < R; start += size) {
        if (MPI_Wtime() - tstart > time_limit) break;

        // Occasionally check if someone else already found it
        if ((iter & (CHECK_BATCH - 1)) == 0) {
            int any = local_found ? 1 : 0, anyg = 0;
            MPI_Allreduce(&any, &anyg, 1, MPI_INT, MPI_LOR, comm);
            if (anyg) break;
        }

        fill(used_mask.begin(), used_mask.end(), 0ULL);
        if (dfs_find_nonsingular_sample_local(rows, n, start, local_sol, used_mask, time_limit, tstart)) {
            local_found = 1;
            break; // keep local_sol as the found solution on this rank
        }
        iter++;
    }

    // Now every rank participates in gather of lengths (collective)
    int local_len = local_found ? (int)local_sol.size() : 0;
    vector<int> all_lens(size);
    MPI_Gather(&local_len, 1, MPI_INT, all_lens.data(), 1, MPI_INT, 0, comm);

    // compute displacements on root
    vector<int> displs(size, 0);
    int total_len = 0;
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            displs[i] = total_len;
            total_len += all_lens[i];
        }
    }

    // Flatten local_sol into sendbuf of length local_len (may be zero)
    vector<int> sendbuf(local_len);
    for (int i = 0; i < local_len; ++i) sendbuf[i] = local_sol[i];

    // Root receives into recvbuf (size total_len); others pass nullptr (but must call)
    vector<int> recvbuf;
    if (rank == 0) recvbuf.resize(total_len);

    // Use MPI_Gatherv to collect variable-length solutions
    // Prepare arrays for counts and displacements (root only); but MPI_Gatherv requires counts and displs on root; non-root still must pass pointers.
    // Convert all_lens to int* and displs to int* for call
    MPI_Gatherv(sendbuf.data(), local_len, MPI_INT,
                (rank == 0 ? recvbuf.data() : nullptr),
                (rank == 0 ? all_lens.data() : nullptr),
                (rank == 0 ? displs.data() : nullptr),
                MPI_INT, 0, comm);

    // If root got any solution(s), pick the first complete solution of length n
    if (rank == 0) {
        sol_out.clear();
        if (total_len > 0) {
            // find first block with length == n
            int idx = 0;
            for (int r = 0; r < size; ++r) {
                int lenr = all_lens[r];
                if (lenr == n) {
                    sol_out.assign(recvbuf.begin() + displs[r], recvbuf.begin() + displs[r] + lenr);
                    break;
                }
            }
            // If none exactly length n but some found shorter (shouldn't happen), ignore.
        }
    }

    // Broadcast whether any found (so all ranks can see)
    int any_found_local = local_found ? 1 : 0;
    int any_found_global = 0;
    MPI_Allreduce(&any_found_local, &any_found_global, 1, MPI_INT, MPI_LOR, comm);

    return any_found_global != 0;
}

// -------------------- Row generation & broadcast (rank 0 generates) --------------------
void generate_and_broadcast_rows_rank0(int p, int n, MPI_Comm comm,
                                      vector<Row>& rows_out, int root,
                                      double &gen_time, int &rows_count) {
    int rank; MPI_Comm_rank(comm, &rank);
    double t0 = MPI_Wtime();
    vector<vector<int>> rows_flat;

    if (rank == root) {
        // generate combinations of size n with integer sum == p-1 (pruned)
        vector<int> pool(p - 1);
        for (int i = 0; i < p - 1; ++i) pool[i] = i + 1;

        vector<int> comb; comb.reserve(n);
        int target = p - 1;

        function<void(int,int,int)> dfs = [&](int start, int depth, int cur_sum) {
            if (depth == n) {
                if (cur_sum == target) rows_flat.push_back(comb);
                return;
            }
            int needed = n - depth;
            if ((int)pool.size() - start < needed) return;
            for (int i = start; i <= (int)pool.size() - needed; ++i) {
                int min_add = 0;
                for (int t = 0; t < needed; ++t) min_add += pool[i + t];
                int max_add = 0;
                for (int t = 0; t < needed; ++t) max_add += pool[pool.size() - 1 - t];
                int min_total = cur_sum + min_add;
                int max_total = cur_sum + max_add;
                if (target < min_total || target > max_total) continue;
                comb.push_back(pool[i]);
                dfs(i + 1, depth + 1, cur_sum + pool[i]);
                comb.pop_back();
            }
        };
        dfs(0, 0, 0);
    }

    double t1 = MPI_Wtime();
    int rcount = (rank == root) ? (int)rows_flat.size() : 0;
    // Broadcast count to all ranks
    MPI_Bcast(&rcount, 1, MPI_INT, root, comm);
    rows_count = rcount;

    if (rcount == 0) {
        gen_time = (rank == root ? (t1 - t0) : 0.0);
        rows_out.clear();
        return;
    }

    // Prepare flat buffer on root
    if (rank == root) {
        vector<int> flat; flat.reserve(rcount * n);
        for (auto &r : rows_flat) for (int v : r) flat.push_back(v);
        MPI_Bcast(flat.data(), rcount * n, MPI_INT, root, comm);

        int W = (p + 63) / 64;
        rows_out.resize(rcount);
        for (int i = 0; i < rcount; ++i) {
            rows_out[i].elems.SetLength(n);
            rows_out[i].mask.assign(W, 0ULL);
            for (int j = 0; j < n; ++j) {
                int val = flat[i * n + j];
                rows_out[i].elems[j] = conv<ZZ_p>(val);
                set_bit(rows_out[i].mask, val - 1);
            }
        }
    } else {
        vector<int> flat(rcount * n);
        MPI_Bcast(flat.data(), rcount * n, MPI_INT, root, comm);
        int W = (p + 63) / 64;
        rows_out.resize(rcount);
        for (int i = 0; i < rcount; ++i) {
            rows_out[i].elems.SetLength(n);
            rows_out[i].mask.assign(W, 0ULL);
            for (int j = 0; j < n; ++j) {
                int val = flat[i * n + j];
                rows_out[i].elems[j] = conv<ZZ_p>(val);
                set_bit(rows_out[i].mask, val - 1);
            }
        }
    }

    gen_time = (rank == root ? (t1 - t0) : 0.0);
}

// -------------------- Find max d with sample extraction --------------------
int find_max_d_with_sample(int p, MPI_Comm comm, double time_per_n,
                           int &rows_for_best, double &row_gen_time_for_best,
                           vector<long> &sample_matrix_flat) {
    int rank, size; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);

    int ub = (int)floor(sqrt((double)(p - 1)));
    int lo = 2, hi = ub, best = 1;
    rows_for_best = 0; row_gen_time_for_best = 0.0;
    sample_matrix_flat.clear();

    vector<Row> best_rows;
    int best_n = 1;

    double bs_start = MPI_Wtime();

    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (rank == 0) { cout << "[BS] Testing n=" << mid << " [" << lo << "," << hi << "]\n"; cout.flush(); }

        vector<Row> rows;
        double gen_time = 0.0; int rowcount = 0;
        generate_and_broadcast_rows_rank0(p, mid, comm, rows, 0, gen_time, rowcount);

        if (rank == 0) { cout << "  rows_generated=" << rowcount << " gen_time(s)=" << gen_time << "\n"; cout.flush(); }

        bool feasible = false;
        vector<int> sol;

        if (rowcount >= mid && !rows.empty()) {
            feasible = parallel_find_nonsingular_sample(rows, mid, comm, sol, time_per_n);
            // If feasible, root (inside parallel function) already has a sample in sol_out
            // We rely on sol (root-populated) only if rank==0 inside the gather
        }

        // All ranks see feasibility via parallel_find_nonsingular_sample's internal reduction
        if (rank == 0) { cout << "  feasible=" << (feasible ? "YES" : "NO") << "\n"; cout.flush(); }

        if (feasible) {
            best = mid;
            rows_for_best = rowcount;
            row_gen_time_for_best = gen_time;
            best_rows = rows;
            best_n = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    double bs_total = MPI_Wtime() - bs_start;
    if (rank == 0) cout << "Binary search total time(s): " << bs_total << "\n";

    // If we found best>1, request a sample using the saved best_rows (broadcasted earlier)
    if (best > 1 && !best_rows.empty()) {
        if (rank == 0) { cout << "Extracting sample for best n=" << best << "\n"; cout.flush(); }
        vector<int> sample_sol;
        bool any = parallel_find_nonsingular_sample(best_rows, best_n, comm, sample_sol, time_per_n);

        // Root will now have sample_sol (if any) because parallel function gathered to root
        if (rank == 0 && !sample_sol.empty()) {
            sample_matrix_flat.clear();
            sample_matrix_flat.reserve(best_n * best_n);
            for (int i = 0; i < best_n; ++i)
                for (int j = 0; j < best_n; ++j)
                    sample_matrix_flat.push_back(conv<long>(rep(best_rows[sample_sol[i]].elems[j])));
        }
    }

    // Broadcast sample size and data (so everyone has consistent view)
    int sample_len = (int)sample_matrix_flat.size();
    MPI_Bcast(&sample_len, 1, MPI_INT, 0, comm);
    if (sample_len > 0) {
        if (rank != 0) sample_matrix_flat.resize(sample_len);
        MPI_Bcast(sample_matrix_flat.data(), sample_len, MPI_LONG, 0, comm);
    }

    MPI_Bcast(&rows_for_best, 1, MPI_INT, 0, comm);
    MPI_Bcast(&row_gen_time_for_best, 1, MPI_DOUBLE, 0, comm);

    return best;
}

// -------------------- main --------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);

    if (argc < 2) {
        if (rank == 0) cerr << "Usage: " << argv[0] << " <primes_file_or_single_p> [--binary-search] [--time-limit secs]\n";
        MPI_Finalize(); return 1;
    }

    bool use_binary = false;
    double time_limit = 120.0;
    for (int i = 2; i < argc; ++i) {
        string s = argv[i];
        if (s == "--binary-search") use_binary = true;
        if (s == "--time-limit" && i + 1 < argc) time_limit = atof(argv[++i]);
    }
    if (!use_binary) {
        if (rank == 0) cerr << "This program expects --binary-search\n";
        MPI_Finalize(); return 1;
    }

    // Read primes on rank 0 and broadcast
    vector<int> primes;
    if (rank == 0) {
        string arg = argv[1];
        bool isnum = !arg.empty();
        for (char c : arg) if (!isdigit(c) && c != '-') { isnum = false; break; }
        if (isnum) primes.push_back(stoi(arg));
        else {
            ifstream in(arg);
            if (!in) { cerr << "Rank0: cannot open " << arg << "\n"; MPI_Abort(comm, 1); }
            int v; while (in >> v) primes.push_back(v);
            in.close();
            if (primes.empty()) { cerr << "Rank0: primes empty\n"; MPI_Abort(comm, 1); }
        }
    }
    int pcount = (int)primes.size();
    MPI_Bcast(&pcount, 1, MPI_INT, 0, comm);
    if (rank != 0) primes.resize(pcount);
    if (pcount > 0) MPI_Bcast(primes.data(), pcount, MPI_INT, 0, comm);

    for (int ip = 0; ip < pcount; ++ip) {
        int p = primes[ip];
        if (rank == 0) { cout << "\n=== START p=" << p << " (ranks=" << size << ") ===\n"; cout.flush(); }

        ZZ_p::init(ZZ(p));
        string pdir = string("p_") + to_string(p);
        if (rank == 0) fs::create_directories(pdir);
        MPI_Barrier(comm);

        int rows_for_best = 0;
        double row_gen_time = 0.0;
        vector<long> sample_flat;
        int best = find_max_d_with_sample(p, comm, time_limit, rows_for_best, row_gen_time, sample_flat);

        if (rank == 0) {
            ofstream out(pdir + "/max_dimension.txt");
            out << "p = " << p << "\n";
            out << "max_d = " << best << "\n";
            out << "log2(p) = " << log2((double)p) << "\n";
            out << "sqrt(p-1) = " << sqrt((double)(p - 1)) << "\n";
            out << "rows_generated_for_max_d = " << rows_for_best << "\n";
            out << "row_generation_time_sec = " << row_gen_time << "\n";
            out.close();

            if (!sample_flat.empty()) {
                int n = best;
                ofstream s(pdir + "/sample_matrix_max_d.txt");
                s << "# sample non-singular matrix for p=" << p << " n=" << n << "\n";
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) s << sample_flat[i * n + j] << (j + 1 == n ? "\n" : " ");
                }
                s.close();
                cout << "Wrote sample matrix to " << pdir << "/sample_matrix_max_d.txt\n";
            }
            cout << "Wrote " << pdir << "/max_dimension.txt\n";
        }

        MPI_Barrier(comm);
        if (rank == 0) { cout << "=== FINISHED p=" << p << " ===\n"; cout.flush(); }
    }

    MPI_Finalize();
    return 0;
}


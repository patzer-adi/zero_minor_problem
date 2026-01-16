#include <NTL/mat_ZZ_p.h>
#include <NTL/ZZ_p.h>
#include <mpi.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <map>
#include <set>
#include <string>
#include <cstring>

using namespace std;
using namespace NTL;

using u64 = uint64_t;
using Clock = chrono::high_resolution_clock;

/* ================= ROW STRUCTURE ================= */

struct Row {
    vector<u64> mask;
    vector<int> elements;
};

inline bool intersects(const vector<u64>& a, const vector<u64>& b) {
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] & b[i]) return true;
    return false;
}

inline void or_into(vector<u64>& a, const vector<u64>& b) {
    for (size_t i = 0; i < a.size(); ++i) a[i] |= b[i];
}

inline void xor_into(vector<u64>& a, const vector<u64>& b) {
    for (size_t i = 0; i < a.size(); ++i) a[i] ^= b[i];
}

inline int popcount(const vector<u64>& m) {
    int c = 0;
    for (u64 x : m) c += __builtin_popcountll(x);
    return c;
}

/* ================= ROW GENERATION ================= */

void generate_rows(int n, long p, vector<Row>& rows) {
    vector<int> pool(p - 1);
    for (int i = 0; i < p - 1; ++i) pool[i] = i + 1;

    int W = (p + 63) / 64;
    vector<int> cur;
    cur.reserve(n);

    vector<long> prefix(p - 1);
    prefix[0] = pool[0];
    for (int i = 1; i < p - 1; ++i)
        prefix[i] = prefix[i - 1] + pool[i];

    const long target = p - 1;

    function<void(int,int,long)> dfs = [&](int start, int depth, long sum) {
        if (sum > target) return;
        if (depth == n) {
            if (sum == target) {
                Row r;
                r.mask.assign(W, 0ULL);
                r.elements = cur;

                for (int x : cur) {
                    int v = x - 1;
                    r.mask[v >> 6] |= (1ULL << (v & 63));
                }
                rows.push_back(move(r));
            }
            return;
        }

        int need = n - depth;
        if (start + need > (int)pool.size()) return;

        long max_possible = prefix.back() - (start ? prefix[start - 1] : 0);
        if (sum + max_possible < target) return;

        for (int i = start; i <= (int)pool.size() - need; ++i) {
            cur.push_back(pool[i]);
            dfs(i + 1, depth + 1, sum + pool[i]);
            cur.pop_back();
        }
    };

    dfs(0, 0, 0);
}

/* ================= SEARCH RESULT ================= */

enum class SearchStatus { FOUND, NOT_FOUND, TIMEOUT };

struct SearchResult {
    SearchStatus status;
    long long nodes_explored;
    double time_taken;
    vector<int> solution_indices;
};

/* ================= PARALLEL SEARCH (FIXED) ================= */

SearchResult parallel_search(const vector<Row>& rows, int n, long p,
                             MPI_Comm comm, double time_limit) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    auto t0 = Clock::now();
    int W = rows[0].mask.size();

    bool local_found = false;
    long long local_nodes = 0;
    bool should_stop = false;
    vector<int> solution;

    auto time_exceeded = [&]() {
        return chrono::duration<double>(Clock::now() - t0).count() > time_limit;
    };

    for (int s = rank; s < (int)rows.size() && !should_stop; s += size) {
        vector<u64> used(W, 0ULL);
        or_into(used, rows[s].mask);
        vector<int> selected = {s};

        function<bool(int,int)> dfs = [&](int depth, int last) -> bool {
            local_nodes++;

            if ((local_nodes & 16383) == 0) {
                if (time_exceeded()) {
                    should_stop = true;
                    return false;
                }
            }

            if (depth == n) {
                // Check non-singularity
                mat_ZZ_p M;
                M.SetDims(n, n);
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        M[i][j] = conv<ZZ_p>(rows[selected[i]].elements[j]);
                    }
                }

                ZZ_p det_val = determinant(M);
                return !IsZero(det_val);
            }

            if ((int)rows.size() - last - 1 < n - depth)
                return false;

            for (int i = last + 1; i < (int)rows.size() && !should_stop; ++i) {
                if (!intersects(rows[i].mask, used)) {
                    or_into(used, rows[i].mask);
                    selected.push_back(i);
                    if (dfs(depth + 1, i)) return true;
                    selected.pop_back();
                    xor_into(used, rows[i].mask);
                }
            }
            return false;
        };

        if (dfs(1, s)) {
            local_found = true;
            solution = selected;
            should_stop = true;
            break;
        }
    }

    MPI_Barrier(comm);

    // Determine if anyone found a solution
    int lf = local_found ? 1 : 0, gf = 0;
    MPI_Allreduce(&lf, &gf, 1, MPI_INT, MPI_LOR, comm);

    // Sum up total nodes explored
    long long global_nodes = 0;
    MPI_Reduce(&local_nodes, &global_nodes, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);

    // FIXED: Properly broadcast solution from whoever found it
    if (gf) {
        // Find which rank has the solution (using max since -1 < any valid rank)
        int sender_rank = local_found ? rank : -1;
        int actual_sender = -1;
        MPI_Allreduce(&sender_rank, &actual_sender, 1, MPI_INT, MPI_MAX, comm);

        // Broadcast solution size from the finder
        int sol_size = 0;
        if (rank == actual_sender) {
            sol_size = solution.size();
        }
        MPI_Bcast(&sol_size, 1, MPI_INT, actual_sender, comm);

        // Resize solution vector for non-finders
        if (rank != actual_sender) {
            solution.resize(sol_size);
        }

        // Broadcast the actual solution data
        MPI_Bcast(solution.data(), sol_size, MPI_INT, actual_sender, comm);
    }

    double time_taken = chrono::duration<double>(Clock::now() - t0).count();

    SearchStatus status = gf ? SearchStatus::FOUND :
                         (time_taken >= time_limit * 0.95 ? SearchStatus::TIMEOUT : SearchStatus::NOT_FOUND);

    return {status, global_nodes, time_taken, solution};
}

/* ================= TIME LIMITS ================= */

double get_time_limit(int n, size_t num_rows) {
    if (num_rows > 5000000) return 5.0;
    if (num_rows > 3000000) return 10.0;
    if (num_rows > 1500000) return 15.0;
    if (num_rows > 800000) return 20.0;
    if (num_rows > 400000) return 30.0;
    if (n >= 10) return 40.0;
    if (n >= 8) return 50.0;
    return 80.0;
}

/* ================= MATRIX PRINTING ================= */

void print_matrix(ofstream& logfile,
                  const vector<Row>& solution_rows,
                  int n, long p)
{
    logfile << "Solution Matrix (each row sums to " << (p - 1) << "):\n[\n";

    for (int i = 0; i < n; ++i) {
        logfile << "  [";
        for (int j = 0; j < n; ++j) {
            logfile << setw(3) << solution_rows[i].elements[j];
            if (j < n - 1) logfile << ", ";
        }
        logfile << "]";
        if (i < n - 1) logfile << ",";

        // Verify sum
        int sum = 0;
        for (int x : solution_rows[i].elements) sum += x;
        logfile << "  // sum = " << sum << "\n";
    }
    logfile << "]\n\n";
}

/* ================= BINARY SEARCH MODE ================= */

int binary_search_max_dim(long p, int ub, MPI_Comm comm, ofstream& logfile) {
    int local_rank;
    MPI_Comm_rank(comm, &local_rank);

    int lo = 2, hi = ub, best = 1;
    map<int, vector<Row>> row_cache;

    while (lo <= hi) {
        int mid = (lo + hi) / 2;

        if (local_rank == 0)
            cout << "[p=" << p << "] Binary search: testing n=" << mid
                 << " (range [" << lo << "," << hi << "])\n" << flush;

        if (row_cache.find(mid) == row_cache.end()) {
            auto t0 = Clock::now();
            generate_rows(mid, p, row_cache[mid]);
            auto t1 = Clock::now();

            if (local_rank == 0) {
                double gen_time = chrono::duration<double>(t1 - t0).count();
                cout << "[p=" << p << "] Generated " << row_cache[mid].size()
                     << " rows in " << fixed << setprecision(2) << gen_time << "s\n" << flush;
            }

            sort(row_cache[mid].begin(), row_cache[mid].end(),
                 [](const Row& a, const Row& b) {
                     return popcount(a.mask) < popcount(b.mask);
                 });
        }

        double time_limit = get_time_limit(mid, row_cache[mid].size());
        auto result = parallel_search(row_cache[mid], mid, p, comm, time_limit);

        if (local_rank == 0) {
            string status = (result.status == SearchStatus::FOUND ? "FOUND" :
                           result.status == SearchStatus::TIMEOUT ? "TIMEOUT" : "NOT_FOUND");

            cout << "[p=" << p << "] n=" << mid << " " << status << "\n" << flush;

            logfile << mid << "," << row_cache[mid].size() << ","
                    << time_limit << "," << result.time_taken << ","
                    << result.nodes_explored << "," << status << "\n";
            logfile.flush();
        }

        if (result.status == SearchStatus::FOUND) {
            best = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    return best;
}

/* ================= LINEAR SEARCH MODE (FIXED) ================= */

int linear_search_max_dim(long p, int ub, MPI_Comm comm, ofstream& logfile,
                          vector<int>& solution_indices,
                          vector<Row>& solution_rows) {
    int local_rank;
    MPI_Comm_rank(comm, &local_rank);

    for (int n = ub; n >= 2; --n) {
        if (local_rank == 0)
            cout << "[p=" << p << "] Testing n=" << n << "...\n" << flush;

        auto t0 = Clock::now();
        vector<Row> rows;
        generate_rows(n, p, rows);
        auto t1 = Clock::now();

        if (local_rank == 0) {
            double gen_time = chrono::duration<double>(t1 - t0).count();
            cout << "[p=" << p << "] Generated " << rows.size()
                 << " rows in " << fixed << setprecision(2) << gen_time << "s\n" << flush;
        }

        // Skip if no rows generated (impossible for this n)
        if (rows.empty()) {
            if (local_rank == 0) {
                cout << "[p=" << p << "] n=" << n << " SKIPPED (no valid rows)\n" << flush;
                logfile << n << ",0,0,0,0,SKIPPED\n";
                logfile.flush();
            }
            continue;
        }

        sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
            return popcount(a.mask) < popcount(b.mask);
        });

        double time_limit = get_time_limit(n, rows.size());
        auto result = parallel_search(rows, n, p, comm, time_limit);

        if (local_rank == 0) {
            string status = (result.status == SearchStatus::FOUND ? "FOUND" :
                           result.status == SearchStatus::TIMEOUT ? "TIMEOUT" : "NOT_FOUND");

            cout << "[p=" << p << "] n=" << n << " " << status
                 << " (time: " << fixed << setprecision(2) << result.time_taken
                 << "s, nodes: " << result.nodes_explored << ")\n" << flush;

            logfile << n << "," << rows.size() << ","
                    << time_limit << "," << result.time_taken << ","
                    << result.nodes_explored << "," << status << "\n";
            logfile.flush();
        }

        // FIXED: Now all ranks have the solution due to Bcast in parallel_search
        if (result.status == SearchStatus::FOUND) {
            if (local_rank == 0) {
                // Clear previous data from earlier tests
                solution_indices.clear();
                solution_rows.clear();

                // Store solution indices (now guaranteed to be complete on rank 0)
                solution_indices = result.solution_indices;

                // Validate we have exactly n indices
                if ((int)solution_indices.size() != n) {
                    cerr << "FATAL ERROR [p=" << p << ", n=" << n
                         << "]: solution_indices has " << solution_indices.size()
                         << " elements, expected " << n << "\n";
                    MPI_Abort(comm, 1);
                }

                // Store all n rows from the current rows vector
                for (int idx : solution_indices) {
                    if (idx >= 0 && idx < (int)rows.size()) {
                        solution_rows.push_back(rows[idx]);
                    } else {
                        cerr << "FATAL ERROR [p=" << p << ", n=" << n
                             << "]: Invalid index " << idx
                             << " (rows.size=" << rows.size() << ")\n";
                        MPI_Abort(comm, 1);
                    }
                }

                // SAFETY CHECK: Verify we have exactly n rows
                if ((int)solution_rows.size() != n) {
                    cerr << "FATAL ERROR [p=" << p << ", n=" << n
                         << "]: Expected " << n << " rows, got "
                         << solution_rows.size() << "\n";
                    MPI_Abort(comm, 1);
                }

                // Additional validation: check each row has exactly n elements
                for (int i = 0; i < n; ++i) {
                    if ((int)solution_rows[i].elements.size() != n) {
                        cerr << "FATAL ERROR [p=" << p << ", n=" << n
                             << "]: Row " << i << " has "
                             << solution_rows[i].elements.size()
                             << " elements, expected " << n << "\n";
                        MPI_Abort(comm, 1);
                    }
                }
            }
            return n;
        }
    }

    return 1;
}

/* ================= PROCESS SINGLE PRIME ================= */

void process_prime(long p, MPI_Comm prime_comm, int global_rank, bool use_binary) {
    int local_rank;
    MPI_Comm_rank(prime_comm, &local_rank);

    auto prime_start = Clock::now();

    ZZ_p::init(ZZ(p));
    int ub = floor(sqrt(p - 1));

    ofstream logfile;
    if (local_rank == 0) {
        logfile.open("results/matrix_p" + to_string(p) + "_d*.txt");
        logfile << "p = " << p << "\n";
        logfile << "log2(p) = " << fixed << setprecision(5) << log2((double)p) << "\n";
        logfile << "sqrt(p-1) = " << fixed << setprecision(3) << sqrt(p - 1.0) << "\n\n\n";
        logfile << "n,rows,time_limit_s,search_time_s,nodes_explored,result\n";
    }

    vector<int> solution_indices;
    vector<Row> solution_rows;
    int max_dim;

    if (use_binary) {
        max_dim = binary_search_max_dim(p, ub, prime_comm, logfile);
    } else {
        max_dim = linear_search_max_dim(p, ub, prime_comm, logfile,
                                       solution_indices, solution_rows);
    }

    auto prime_end = Clock::now();
    double total_time = chrono::duration<double>(prime_end - prime_start).count();

    if (local_rank == 0) {
        logfile << "\nMAX_DIMENSION = " << max_dim << "\n\n";

        // Print complete matrix with verification
        if (max_dim >= 2 && !solution_rows.empty()) {
            // Double-check we have the right number of rows
            if ((int)solution_rows.size() != max_dim) {
                logfile << "ERROR: solution_rows.size()=" << solution_rows.size()
                       << " but max_dim=" << max_dim << "\n";
                logfile << "Cannot print incomplete solution.\n\n";
            } else {
                print_matrix(logfile, solution_rows, max_dim, p);

                // Comprehensive verification
                logfile << "Verification:\n";

                bool all_correct = true;

                // Check row dimensions
                for (int i = 0; i < max_dim; ++i) {
                    if ((int)solution_rows[i].elements.size() != max_dim) {
                        logfile << "  ERROR: Row " << i << " has "
                               << solution_rows[i].elements.size()
                               << " elements (expected " << max_dim << ")\n";
                        all_correct = false;
                    }
                }

                // Check row sums
                for (int i = 0; i < max_dim; ++i) {
                    long sum = 0;
                    for (int x : solution_rows[i].elements) sum += x;
                    if (sum != p - 1) {
                        logfile << "  ERROR: Row " << i << " sum = " << sum
                               << " (expected " << (p-1) << ")\n";
                        all_correct = false;
                    }
                }

                // Check pairwise disjointness
                bool all_disjoint = true;
                for (int i = 0; i < max_dim; ++i) {
                    for (int j = i + 1; j < max_dim; ++j) {
                        if (intersects(solution_rows[i].mask, solution_rows[j].mask)) {
                            logfile << "  ERROR: Rows " << i << " and " << j << " intersect!\n";

                            // Show which elements overlap
                            std::set<int> row_i_set(solution_rows[i].elements.begin(),
                                                    solution_rows[i].elements.end());
                            std::set<int> overlap;
                            for (int elem : solution_rows[j].elements) {
                                if (row_i_set.count(elem)) {
                                    overlap.insert(elem);
                                }
                            }
                            if (!overlap.empty()) {
                                logfile << "    Overlapping elements: ";
                                for (int elem : overlap) logfile << elem << " ";
                                logfile << "\n";
                            }

                            all_disjoint = false;
                            all_correct = false;
                        }
                    }
                }

                // Verify determinant
                mat_ZZ_p M;
                M.SetDims(max_dim, max_dim);
                for (int i = 0; i < max_dim; ++i) {
                    for (int j = 0; j < max_dim; ++j) {
                        M[i][j] = conv<ZZ_p>(solution_rows[i].elements[j]);
                    }
                }
                ZZ_p det_val = determinant(M);
                bool is_nonsingular = !IsZero(det_val);

                if (all_correct && all_disjoint && is_nonsingular) {
                    logfile << "  ✓ All " << max_dim << " rows have " << max_dim << " elements\n";
                    logfile << "  ✓ All rows sum to " << (p-1) << "\n";
                    logfile << "  ✓ All rows pairwise disjoint\n";
                    logfile << "  ✓ Matrix is non-singular (det ≠ 0 mod " << p << ")\n";
                } else {
                    if (!is_nonsingular) {
                        logfile << "  ERROR: Matrix is singular (det = 0 mod " << p << ")!\n";
                    }
                }
                logfile << "\n";
            }
        } else if (max_dim >= 2) {
            logfile << "WARNING: Solution not saved (binary search mode or no solution found)\n\n";
        } else {
            logfile << "No solution matrix to print (max_dim < 2)\n\n";
        }

        logfile << "Total time: " << fixed << setprecision(2) << total_time << " seconds\n";
        logfile.close();

        // Rename file with actual dimension
        string old_name = "results/matrix_p" + to_string(p) + "_d*.txt";
        string new_name = "results/matrix_p" + to_string(p) + "_d" + to_string(max_dim) + ".txt";
        rename(old_name.c_str(), new_name.c_str());

        cout << "[p=" << p << "] ✓✓✓ COMPLETE! max_d=" << max_dim
             << " (" << fixed << setprecision(1) << total_time << "s)\n\n" << flush;
    }
}

/* ================= MAIN ================= */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Parse arguments
    if (argc < 2) {
        if (world_rank == 0) {
            cerr << "Usage: mpirun -np <N> ./pattern_mat primes.txt [options]\n";
            cerr << "Options:\n";
            cerr << "  --binary-search        Use binary search (faster but may be incomplete)\n";
            cerr << "  --procs-per-prime N    Use N processes per prime (default: all)\n";
            cerr << "\nExamples:\n";
            cerr << "  mpirun -np 48 --hostfile hosts.txt ./pattern_mat primes.txt\n";
            cerr << "  mpirun -np 48 --hostfile hosts.txt ./pattern_mat primes.txt --binary-search\n";
            cerr << "  mpirun -np 48 ./pattern_mat primes.txt --procs-per-prime 4\n";
        }
        MPI_Finalize();
        return 1;
    }

    string prime_file = argv[1];
    bool use_binary = false;
    int procs_per_prime = world_size;

    // Parse options
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--binary-search") == 0) {
            use_binary = true;
        } else if (strcmp(argv[i], "--procs-per-prime") == 0 && i + 1 < argc) {
            procs_per_prime = atoi(argv[i + 1]);
            i++;
        }
    }

    // Create results directory
    if (world_rank == 0) {
        system("mkdir -p results");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Load primes
    vector<long> primes;
    if (world_rank == 0) {
        ifstream in(prime_file);
        long x;
        while (in >> x) primes.push_back(x);
        in.close();

        cout << "========================================\n";
        cout << "PATTERN MATRIX SEARCH\n";
        cout << "========================================\n";
        cout << "Total processes: " << world_size << "\n";
        cout << "Search mode: " << (use_binary ? "Binary Search" : "Linear Descending") << "\n";
        cout << "Processes per prime: " << procs_per_prime << "\n";
        if (procs_per_prime < world_size) {
            cout << "Concurrent primes: " << (world_size / procs_per_prime) << "\n";
        }
        cout << "Total primes: " << primes.size() << "\n";
        cout << "========================================\n\n" << flush;
    }

    long nprimes;
    if (world_rank == 0) nprimes = primes.size();
    MPI_Bcast(&nprimes, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    if (world_rank != 0) primes.resize(nprimes);
    MPI_Bcast(primes.data(), nprimes, MPI_LONG, 0, MPI_COMM_WORLD);

    if (procs_per_prime < world_size && world_size % procs_per_prime == 0) {
        // Multi-prime parallel mode
        int num_groups = world_size / procs_per_prime;
        int my_group = world_rank / procs_per_prime;

        MPI_Comm prime_comm;
        MPI_Comm_split(MPI_COMM_WORLD, my_group, world_rank, &prime_comm);

        for (long idx = my_group; idx < nprimes; idx += num_groups) {
            process_prime(primes[idx], prime_comm, world_rank, use_binary);
        }

        MPI_Comm_free(&prime_comm);
    } else {
        // Single prime mode
        for (long p : primes) {
            process_prime(p, MPI_COMM_WORLD, world_rank, use_binary);
        }
    }

    if (world_rank == 0) {
        cout << "\n========================================\n";
        cout << "ALL PRIMES COMPLETED!\n";
        cout << "========================================\n";
    }

    MPI_Finalize();

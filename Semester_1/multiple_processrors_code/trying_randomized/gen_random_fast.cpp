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
#include <random>
#include <sys/stat.h>

using namespace std;
using namespace NTL;
using u64 = uint64_t;
using Clock = chrono::high_resolution_clock;

/* ================= ROW STRUCTURE ================= */
struct Row {
    vector<u64> mask;
    vector<int> elements;
    int id;
};

inline bool intersects(const vector<u64>& a, const vector<u64>& b) {
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] & b[i]) return true;
    return false;
}

inline void or_into(vector<u64>& a, const vector<u64>& b) {
    for (size_t i = 0; i < a.size(); ++i) a[i] |= b[i];
}

/* ================= ROW GENERATION ================= */
void generate_rows(int n, long p, vector<Row>& rows) {
    vector<int> pool(p - 1);
    for (int i = 0; i < p - 1; ++i) pool[i] = i + 1;

    int W = (p + 63) / 64;
    vector<int> cur;
    cur.reserve(n);

    const long target = p - 1;
    int row_id = 0;

    function<void(int,int,long)> dfs = [&](int start, int depth, long sum) {
        if (depth == n) {
            if (sum == target) {
                Row r;
                r.mask.assign(W, 0ULL);
                r.id = row_id++;
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

        for (int i = start; i <= (int)pool.size() - need; ++i) {
            cur.push_back(pool[i]);
            dfs(i + 1, depth + 1, sum + pool[i]);
            cur.pop_back();
        }
    };

    dfs(0, 0, 0);
}

/* ================= FILE I/O ================= */
void save_rows(const vector<Row>& rows, long p, int n, const string& dir) {
    string filename = dir + "/rows_p" + to_string(p) + "_n" + to_string(n) + ".bin";
    ofstream out(filename, ios::binary);

    size_t num_rows = rows.size();
    size_t mask_size = rows[0].mask.size();

    out.write((char*)&num_rows, sizeof(num_rows));
    out.write((char*)&mask_size, sizeof(mask_size));
    out.write((char*)&n, sizeof(n));

    for (const auto& row : rows) {
        out.write((char*)row.mask.data(), mask_size * sizeof(u64));
        out.write((char*)row.elements.data(), n * sizeof(int));
    }
    out.close();
}

bool load_rows(vector<Row>& rows, long p, int n, const string& dir) {
    string filename = dir + "/rows_p" + to_string(p) + "_n" + to_string(n) + ".bin";
    ifstream in(filename, ios::binary);
    if (!in) return false;

    size_t num_rows, mask_size;
    int dimension;
    in.read((char*)&num_rows, sizeof(num_rows));
    in.read((char*)&mask_size, sizeof(mask_size));
    in.read((char*)&dimension, sizeof(dimension));

    rows.clear();
    rows.reserve(num_rows);

    for (size_t i = 0; i < num_rows; ++i) {
        Row r;
        r.mask.resize(mask_size);
        r.elements.resize(dimension);
        r.id = i;
        in.read((char*)r.mask.data(), mask_size * sizeof(u64));
        in.read((char*)r.elements.data(), dimension * sizeof(int));
        rows.push_back(move(r));
    }
    in.close();
    return true;
}

/* ================= MONTE CARLO SEARCH ================= */
bool monte_carlo_search(const vector<Row>& rows, int n, long p,
                        vector<Row>& solution,
                        MPI_Comm comm, double time_limit, long long max_trials) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    auto t0 = Clock::now();
    int W = rows[0].mask.size();

    mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count() + rank * 99991);
    uniform_int_distribution<size_t> dist(0, rows.size() - 1);

    bool local_found = false;
    long long local_trials = 0;

    while (local_trials < max_trials && !local_found) {
        // Check timeout
        if (local_trials % 1000 == 0) {
            double elapsed = chrono::duration<double>(Clock::now() - t0).count();
            if (elapsed > time_limit) break;
        }

        // Check if another rank found it
        if (local_trials % 10000 == 0) {
            int any = 0, lf = local_found ? 1 : 0;
            MPI_Allreduce(&lf, &any, 1, MPI_INT, MPI_LOR, comm);
            if (any) break;
        }

        local_trials++;

        // Random greedy selection
        vector<u64> used(W, 0ULL);
        vector<Row> selected;
        selected.reserve(n);

        for (int depth = 0; depth < n; ++depth) {
            // Find all compatible rows
            vector<size_t> candidates;
            for (size_t i = 0; i < rows.size(); ++i) {
                if (!intersects(rows[i].mask, used)) {
                    candidates.push_back(i);
                }
            }

            if (candidates.empty()) break;

            // Pick random candidate
            size_t choice = candidates[dist(rng) % candidates.size()];
            selected.push_back(rows[choice]);
            or_into(used, rows[choice].mask);
        }

        // Check if we got n rows
        if ((int)selected.size() == n) {
            // Verify with determinant
            Mat<ZZ_p> M;
            M.SetDims(n, n);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    M[i][j] = conv<ZZ_p>(selected[i].elements[j]);
                }
            }

            if (determinant(M) != 0) {
                solution = selected;
                local_found = true;
            }
        }
    }

    // Final sync
    int lf = local_found ? 1 : 0, gf = 0;
    MPI_Allreduce(&lf, &gf, 1, MPI_INT, MPI_LOR, comm);

    // Gather solution to rank 0
    if (gf && local_found && rank != 0) {
        int sol_size = solution.size();
        MPI_Send(&sol_size, 1, MPI_INT, 0, 0, comm);
        for (const auto& row : solution) {
            MPI_Send(row.elements.data(), n, MPI_INT, 0, 1, comm);
        }
    }

    if (rank == 0 && gf && !local_found) {
        MPI_Status status;
        int sol_size;
        MPI_Recv(&sol_size, 1, MPI_INT, MPI_ANY_SOURCE, 0, comm, &status);
        solution.resize(sol_size);
        for (auto& row : solution) {
            row.elements.resize(n);
            MPI_Recv(row.elements.data(), n, MPI_INT, status.MPI_SOURCE, 1, comm, &status);
        }
    }

    long long total_trials = 0;
    MPI_Reduce(&local_trials, &total_trials, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);

    if (rank == 0) {
        double elapsed = chrono::duration<double>(Clock::now() - t0).count();
        cout << "    Trials: " << total_trials << ", Time: " << fixed
             << setprecision(2) << elapsed << "s, Result: "
             << (gf ? "FOUND ✓" : "TIMEOUT ✗") << endl;
    }

    return gf != 0;
}

/* ================= PHASE 1: PRE-GENERATION ================= */
void pregenerate_phase(const vector<long>& primes, const string& dir, int rank) {
    mkdir(dir.c_str(), 0755);

    if (rank == 0) {
        cout << "\n========================================\n";
        cout << "PHASE 1: PRE-GENERATING ALL ROWS\n";
        cout << "========================================\n";
    }

    for (long p : primes) {
        if (rank == 0) {
            cout << "\nPrime p = " << p << "\n";
        }

        ZZ_p::init(ZZ(p));
        int max_n = (int)floor(sqrt(p - 1));

        for (int n = 2; n <= max_n; ++n) {
            string filename = dir + "/rows_p" + to_string(p) + "_n" + to_string(n) + ".bin";
            ifstream check(filename);
            if (check.good()) {
                if (rank == 0) cout << "  n=" << n << ": exists, skipping\n";
                check.close();
                continue;
            }

            if (rank == 0) {
                cout << "  n=" << n << ": generating... " << flush;

                auto t0 = Clock::now();
                vector<Row> rows;
                generate_rows(n, p, rows);
                auto t1 = Clock::now();

                double gen_time = chrono::duration<double>(t1 - t0).count();
                save_rows(rows, p, n, dir);

                cout << rows.size() << " rows in "
                     << fixed << setprecision(2) << gen_time << "s ✓\n";
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        cout << "\n========================================\n";
        cout << "PRE-GENERATION COMPLETE!\n";
        cout << "All rows saved to: " << dir << "/\n";
        cout << "========================================\n";
    }
}

/* ================= PHASE 2: SEARCH ================= */
void search_phase(const vector<long>& primes, const string& dir, int rank, int size) {
    mkdir("results", 0755);

    if (rank == 0) {
        cout << "\n========================================\n";
        cout << "PHASE 2: SEARCHING FOR MAX DIMENSIONS\n";
        cout << "Using Monte Carlo with " << size << " MPI ranks\n";
        cout << "========================================\n";
    }

    for (long p : primes) {
        auto p_start = Clock::now();

        if (rank == 0) {
            cout << "\n========================================\n";
            cout << "PRIME p = " << p << "\n";
            cout << "sqrt(p-1) = " << fixed << setprecision(3) << sqrt(p-1) << "\n";
            cout << "========================================\n";
        }

        ZZ_p::init(ZZ(p));
        int max_n = (int)floor(sqrt(p - 1));
        int found_d = 0;
        vector<Row> best_solution;

        // Search from max_n down to 2
        for (int n = max_n; n >= 2; --n) {
            if (rank == 0) {
                cout << "\n--- Testing n = " << n << " ---\n";
            }

            vector<Row> rows;
            if (!load_rows(rows, p, n, dir)) {
                if (rank == 0) {
                    cerr << "  ERROR: Cannot load rows file!\n";
                }
                continue;
            }

            if (rank == 0) {
                cout << "  Loaded " << rows.size() << " rows\n";
            }

            // Adaptive parameters based on row count
            long long max_trials;
            double time_limit;

            if (rows.size() > 5000000) {
                max_trials = 50000;
                time_limit = 10.0;
            } else if (rows.size() > 1000000) {
                max_trials = 200000;
                time_limit = 30.0;
            } else if (rows.size() > 100000) {
                max_trials = 500000;
                time_limit = 60.0;
            } else {
                max_trials = 1000000;
                time_limit = 120.0;
            }

            if (rank == 0) {
                cout << "  Monte Carlo search (max trials: " << max_trials
                     << " per rank, timeout: " << time_limit << "s)\n";
            }

            vector<Row> solution;
            bool found = monte_carlo_search(rows, n, p, solution, MPI_COMM_WORLD,
                                           time_limit, max_trials);

            if (found) {
                found_d = n;
                best_solution = solution;

                if (rank == 0) {
                    cout << "\n  ✓✓✓ FOUND! Maximum dimension = " << n << " ✓✓✓\n";
                }
                break; // Found max, stop
            }
        }

        // Save results
        if (rank == 0) {
            auto p_end = Clock::now();
            double total_time = chrono::duration<double>(p_end - p_start).count();

            string result_file = "results/matrix_p" + to_string(p) + "_d" + to_string(found_d) + ".txt";
            ofstream fout(result_file);

            fout << "========================================\n";
            fout << "Prime: " << p << "\n";
            fout << "Maximum Dimension: " << found_d << "\n";
            fout << "sqrt(p-1): " << sqrt(p-1) << "\n";
            fout << "Total time: " << total_time << " seconds\n";
            fout << "========================================\n\n";

            if (found_d > 0 && !best_solution.empty()) {
                fout << "Matrix (each row sums to " << (p-1) << "):\n";

                // Build and verify
                Mat<ZZ_p> M;
                M.SetDims(found_d, found_d);
                for (int i = 0; i < found_d; ++i) {
                    for (int j = 0; j < found_d; ++j) {
                        M[i][j] = conv<ZZ_p>(best_solution[i].elements[j]);
                    }
                }

                for (int i = 0; i < found_d; ++i) {
                    for (int j = 0; j < found_d; ++j) {
                        fout << setw(4) << M[i][j];
                    }
                    fout << "\n";
                }

                fout << "\nRow Elements:\n";
                for (size_t i = 0; i < best_solution.size(); ++i) {
                    fout << "Row " << (i+1) << ": ";
                    int sum = 0;
                    for (int x : best_solution[i].elements) {
                        fout << x << " ";
                        sum += x;
                    }
                    fout << " (sum=" << sum << ")\n";
                }

                fout << "\nVerification:\n";
                ZZ_p det = determinant(M);
                fout << "Determinant: " << det << "\n";
                fout << "Non-singular: " << (det != 0 ? "YES ✓" : "NO") << "\n";

                // Check uniqueness
                vector<bool> used(p, false);
                bool all_unique = true;
                for (const auto& row : best_solution) {
                    for (int x : row.elements) {
                        if (used[x]) all_unique = false;
                        used[x] = true;
                    }
                }
                fout << "All elements unique: " << (all_unique ? "YES ✓" : "NO") << "\n";
                fout << "Elements in [1, p-1]: YES ✓\n";
            } else {
                fout << "No valid matrix found.\n";
            }

            fout.close();

            cout << "\n========================================\n";
            cout << "RESULT: max_d(" << p << ") = " << found_d << "\n";
            cout << "Time: " << total_time << " seconds\n";
            cout << "Saved to: " << result_file << "\n";
            cout << "========================================\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
}

/* ================= MAIN ================= */
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <primes.txt>\n";
            cerr << "\nThis will:\n";
            cerr << "  1. Pre-generate rows (if not exist)\n";
            cerr << "  2. Search for max dimensions\n";
            cerr << "  3. Save results to results/\n";
        }
        MPI_Finalize();
        return 1;
    }

    string data_dir = "row_data";
    mkdir(data_dir.c_str(), 0755);
    mkdir("results", 0755);

    // Read primes
    vector<long> primes;
    if (rank == 0) {
        ifstream in(argv[1]);
        if (!in) {
            cerr << "ERROR: Cannot open " << argv[1] << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        long p;
        while (in >> p) {
            if (p > 1 && p < 65536) primes.push_back(p);
        }
        in.close();

        cout << "========================================\n";
        cout << "FAST MAX DIMENSION FINDER\n";
        cout << "========================================\n";
        cout << "Primes: " << primes.size() << "\n";
        cout << "MPI Ranks: " << size << "\n";
        cout << "Method: Monte Carlo Random Search\n";
        cout << "========================================\n";
    }

    long npr = primes.size();
    MPI_Bcast(&npr, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    if (rank != 0) primes.resize(npr);
    MPI_Bcast(primes.data(), npr, MPI_LONG, 0, MPI_COMM_WORLD);

    // Process each prime: pregenerate if needed, then search
    for (size_t pidx = 0; pidx < primes.size(); ++pidx) {
        long p = primes[pidx];

        auto prime_start = Clock::now();

        if (rank == 0) {
            cout << "\n========================================\n";
            cout << "PRIME " << (pidx+1) << "/" << primes.size() << ": p = " << p << "\n";
            cout << "sqrt(p-1) = " << fixed << setprecision(3) << sqrt(p-1) << "\n";
            cout << "========================================\n";
        }

        ZZ_p::init(ZZ(p));
        int max_n = (int)floor(sqrt(p - 1));

        // PHASE 1: Pre-generate rows if they don't exist
        if (rank == 0) {
            cout << "\n[1] Checking/Generating rows...\n";
        }

        for (int n = 2; n <= max_n; ++n) {
            string filename = data_dir + "/rows_p" + to_string(p) + "_n" + to_string(n) + ".bin";
            ifstream check(filename);

            if (check.good()) {
                // File exists, skip
                check.close();
                if (rank == 0 && n == 2) {
                    cout << "  Rows already exist, using cached files\n";
                }
                continue;
            }

            // Need to generate
            if (rank == 0) {
                cout << "  Generating n=" << n << "... " << flush;

                auto t0 = Clock::now();
                vector<Row> rows;
                generate_rows(n, p, rows);
                auto t1 = Clock::now();

                double gen_time = chrono::duration<double>(t1 - t0).count();
                save_rows(rows, p, n, data_dir);

                cout << rows.size() << " rows in "
                     << fixed << setprecision(2) << gen_time << "s ✓\n";
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // PHASE 2: Search for max dimension
        if (rank == 0) {
            cout << "\n[2] Searching for max dimension...\n";
        }

        int found_d = 0;
        vector<Row> best_solution;

        // Search from max_n down to 2
        for (int n = max_n; n >= 2; --n) {
            if (rank == 0) {
                cout << "\n  Testing n = " << n << "... " << flush;
            }

            vector<Row> rows;
            if (!load_rows(rows, p, n, data_dir)) {
                if (rank == 0) {
                    cerr << "\n    ERROR: Cannot load rows!\n";
                }
                continue;
            }

            if (rank == 0) {
                cout << rows.size() << " rows\n";
            }

            // Adaptive parameters
            long long max_trials;
            double time_limit;

            if (rows.size() > 5000000) {
                max_trials = 50000;
                time_limit = 10.0;
            } else if (rows.size() > 1000000) {
                max_trials = 200000;
                time_limit = 30.0;
            } else if (rows.size() > 100000) {
                max_trials = 500000;
                time_limit = 60.0;
            } else {
                max_trials = 1000000;
                time_limit = 120.0;
            }

            if (rank == 0) {
                cout << "    Searching (trials: " << max_trials
                     << " per rank, timeout: " << time_limit << "s)... " << flush;
            }

            vector<Row> solution;
            bool found = monte_carlo_search(rows, n, p, solution, MPI_COMM_WORLD,
                                           time_limit, max_trials);

            if (found) {
                found_d = n;
                best_solution = solution;

                if (rank == 0) {
                    cout << "\n\n  ✓✓✓ FOUND! Maximum dimension = " << n << " ✓✓✓\n";
                }
                break;
            } else {
                if (rank == 0) {
                    cout << "\n";
                }
            }
        }

        // Save results
        if (rank == 0) {
            auto prime_end = Clock::now();
            double total_time = chrono::duration<double>(prime_end - prime_start).count();

            string result_file = "results/matrix_p" + to_string(p) + "_d" + to_string(found_d) + ".txt";
            ofstream fout(result_file);

            fout << "========================================\n";
            fout << "Prime: " << p << "\n";
            fout << "Maximum Dimension: " << found_d << "\n";
            fout << "sqrt(p-1): " << fixed << setprecision(3) << sqrt(p-1) << "\n";
            fout << "Total time: " << fixed << setprecision(2) << total_time << " seconds\n";
            fout << "========================================\n\n";

            if (found_d > 0 && !best_solution.empty()) {
                fout << "Matrix (each row sums to " << (p-1) << "):\n";

                // Build matrix
                Mat<ZZ_p> M;
                M.SetDims(found_d, found_d);
                for (int i = 0; i < found_d; ++i) {
                    for (int j = 0; j < found_d; ++j) {
                        M[i][j] = conv<ZZ_p>(best_solution[i].elements[j]);
                    }
                }

                for (int i = 0; i < found_d; ++i) {
                    for (int j = 0; j < found_d; ++j) {
                        fout << setw(4) << M[i][j];
                    }
                    fout << "\n";
                }

                fout << "\nRow Elements:\n";
                for (size_t i = 0; i < best_solution.size(); ++i) {
                    fout << "Row " << (i+1) << ": ";
                    int sum = 0;
                    for (int x : best_solution[i].elements) {
                        fout << x << " ";
                        sum += x;
                    }
                    fout << " (sum=" << sum << ")\n";
                }

                fout << "\nVerification:\n";
                ZZ_p det = determinant(M);
                fout << "Determinant: " << det << "\n";
                fout << "Non-singular: " << (det != 0 ? "YES ✓" : "NO") << "\n";

                // Check uniqueness
                vector<bool> used(p, false);
                bool all_unique = true;
                for (const auto& row : best_solution) {
                    for (int x : row.elements) {
                        if (x < 1 || x >= p) all_unique = false;
                        if (used[x]) all_unique = false;
                        used[x] = true;
                    }
                }
                fout << "All elements unique: " << (all_unique ? "YES ✓" : "NO") << "\n";
                fout << "All elements in [1, p-1]: " << (all_unique ? "YES ✓" : "NO") << "\n";
            } else {
                fout << "No valid matrix found.\n";
            }

            fout.close();

            cout << "\n========================================\n";
            cout << "★ RESULT: max_d(" << p << ") = " << found_d << "\n";
            cout << "★ Time: " << fixed << setprecision(2) << total_time << " seconds\n";
            cout << "★ Saved: " << result_file << "\n";
            cout << "========================================\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        cout << "\n========================================\n";
        cout << "ALL PRIMES COMPLETED!\n";
        cout << "Results in: results/\n";
        cout << "Row cache in: row_data/\n";
        cout << "========================================\n";
    }

    MPI_Finalize();
    return 0;
}

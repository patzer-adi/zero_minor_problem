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

using namespace std;
using namespace NTL;

using u64 = uint64_t;
using Clock = chrono::high_resolution_clock;

/* ================= BITSET ================= */

struct Row {
    vector<u64> mask;
    Vec<ZZ_p> elems;
};

inline void or_into(vector<u64>& a, const vector<u64>& b) {
    for (size_t i = 0; i < a.size(); ++i) a[i] |= b[i];
}
inline void xor_into(vector<u64>& a, const vector<u64>& b) {
    for (size_t i = 0; i < a.size(); ++i) a[i] ^= b[i];
}
inline bool intersects(const vector<u64>& a, const vector<u64>& b) {
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] & b[i]) return true;
    return false;
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

    auto dfs = [&](auto&& self, int start, int depth, long sum) -> void {
        if (sum > target) return;
        if (depth == n) {
            if (sum == target) {
                Row r;
                r.mask.assign(W, 0ULL);
                r.elems.SetLength(n);
                for (int i = 0; i < n; ++i) {
                    r.elems[i] = conv<ZZ_p>(cur[i]);
                    int v = cur[i] - 1;
                    r.mask[v >> 6] |= (1ULL << (v & 63));
                }
                rows.push_back(move(r));
            }
            return;
        }

        int need = n - depth;
        if (start + need > (int)pool.size()) return;

        long max_possible =
            prefix.back() - (start ? prefix[start - 1] : 0);
        if (sum + max_possible < target) return;

        for (int i = start; i <= (int)pool.size() - need; ++i) {
            cur.push_back(pool[i]);
            self(self, i + 1, depth + 1, sum + pool[i]);
            cur.pop_back();
        }
    };

    dfs(dfs, 0, 0, 0);
}

/* ================= MATRIX UTILITIES ================= */

void save_matrix(const mat_ZZ_p& M, const string& filename,
                 long p, int n, bool is_singular) {
    ofstream out(filename);
    out << "Prime p = " << p << "\n";
    out << "Dimension n = " << n << "\n";
    out << "Matrix type: " << (is_singular ? "SINGULAR" : "NON-SINGULAR") << "\n";
    out << "Determinant: " << determinant(M) << "\n\n";
    out << "Matrix:\n";
    out << M << "\n";
    out.close();
}

/* ================= PARALLEL SEARCH ================= */

struct SearchResult {
    bool found;
    vector<int> row_indices;
    bool is_singular;
};

SearchResult parallel_search(const vector<Row>& rows, int n,
                             MPI_Comm comm,
                             double time_limit,
                             long long& explored_nodes) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    auto t0 = Clock::now();
    int W = rows[0].mask.size();

    SearchResult local_result;
    local_result.found = false;
    explored_nodes = 0;

    auto time_exceeded = [&]() {
        double elapsed = chrono::duration<double>(Clock::now() - t0).count();
        return elapsed > time_limit;
    };

    for (int s = rank; s < (int)rows.size(); s += size) {
        if (time_exceeded()) break;

        vector<u64> used(W, 0ULL);
        or_into(used, rows[s].mask);

        vector<int> selected_rows;
        selected_rows.push_back(s);

        function<bool(int,int)> dfs = [&](int depth, int last) -> bool {
            explored_nodes++;

            if ((explored_nodes & 4095) == 0 && time_exceeded())
                return false;

            if (depth == n) {
                // Build matrix and check determinant
                mat_ZZ_p M;
                M.SetDims(n, n);
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        M[i][j] = rows[selected_rows[i]].elems[j];
                    }
                }

                ZZ_p det = determinant(M);

                // Save the first matrix found (singular or non-singular)
                if (!local_result.found) {
                    local_result.found = true;
                    local_result.row_indices = selected_rows;
                    local_result.is_singular = IsZero(det);
                }

                return true;
            }

            for (int i = last + 1; i < (int)rows.size(); ++i) {
                if ((int)rows.size() - i < n - depth) break;
                if (!intersects(rows[i].mask, used)) {
                    or_into(used, rows[i].mask);
                    selected_rows.push_back(i);

                    if (dfs(depth + 1, i)) return true;

                    selected_rows.pop_back();
                    xor_into(used, rows[i].mask);
                }
            }
            return false;
        };

        if (dfs(1, s)) {
            break;
        }

        selected_rows.clear();
    }

    // Gather results from all processes
    int local_found = local_result.found ? 1 : 0;
    vector<int> all_found(size);
    MPI_Gather(&local_found, 1, MPI_INT,
               all_found.data(), 1, MPI_INT, 0, comm);

    // Process 0 collects the first result found
    SearchResult global_result;
    global_result.found = false;

    if (rank == 0) {
        // Find first rank that found a result
        for (int r = 0; r < size; ++r) {
            if (all_found[r]) {
                if (r == 0) {
                    global_result = local_result;
                } else {
                    // Receive from that rank
                    int num_indices;
                    MPI_Recv(&num_indices, 1, MPI_INT, r, 0, comm, MPI_STATUS_IGNORE);
                    global_result.row_indices.resize(num_indices);
                    MPI_Recv(global_result.row_indices.data(), num_indices,
                            MPI_INT, r, 1, comm, MPI_STATUS_IGNORE);
                    int is_sing;
                    MPI_Recv(&is_sing, 1, MPI_INT, r, 2, comm, MPI_STATUS_IGNORE);
                    global_result.is_singular = (is_sing == 1);
                    global_result.found = true;
                }
                break;
            }
        }
    } else if (local_result.found) {
        // Send to rank 0 if requested
        int is_first = 0;
        for (int r = 0; r < rank; ++r) {
            if (all_found[r]) {
                is_first = 0;
                break;
            }
        }
        if (all_found[rank] && is_first == 0) {
            // Check if we're the first by comparing with rank 0
            MPI_Request req;
            int dummy;
            MPI_Irecv(&dummy, 1, MPI_INT, 0, 99, comm, &req);
            MPI_Test(&req, &is_first, MPI_STATUS_IGNORE);

            // Only send if there's no one before us
            bool should_send = true;
            for (int r = 0; r < rank; ++r) {
                if (all_found[r]) {
                    should_send = false;
                    break;
                }
            }

            if (should_send) {
                int num_indices = local_result.row_indices.size();
                MPI_Send(&num_indices, 1, MPI_INT, 0, 0, comm);
                MPI_Send(local_result.row_indices.data(), num_indices,
                        MPI_INT, 0, 1, comm);
                int is_sing = local_result.is_singular ? 1 : 0;
                MPI_Send(&is_sing, 1, MPI_INT, 0, 2, comm);
            }
        }
    }

    long long global_nodes = 0;
    MPI_Reduce(&explored_nodes, &global_nodes,
               1, MPI_LONG_LONG, MPI_SUM, 0, comm);

    int gf = 0;
    MPI_Allreduce(&local_found, &gf, 1, MPI_INT, MPI_LOR, comm);

    explored_nodes = global_nodes;

    // Broadcast result to all ranks
    int result_found = global_result.found ? 1 : 0;
    MPI_Bcast(&result_found, 1, MPI_INT, 0, comm);
    global_result.found = (result_found == 1);

    if (global_result.found) {
        int num_indices = global_result.row_indices.size();
        MPI_Bcast(&num_indices, 1, MPI_INT, 0, comm);
        if (rank != 0) global_result.row_indices.resize(num_indices);
        MPI_Bcast(global_result.row_indices.data(), num_indices, MPI_INT, 0, comm);
        int is_sing = global_result.is_singular ? 1 : 0;
        MPI_Bcast(&is_sing, 1, MPI_INT, 0, comm);
        global_result.is_singular = (is_sing == 1);
    }

    return global_result;
}

/* ================= MAIN ================= */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == 0)
            cerr << "Usage: mpirun -np k ./max_d primes.txt\n";
        MPI_Finalize();
        return 1;
    }

    vector<long> primes;
    if (rank == 0) {
        ifstream in(argv[1]);
        long x;
        while (in >> x) primes.push_back(x);
    }

    long npr = primes.size();
    MPI_Bcast(&npr, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    if (rank != 0) primes.resize(npr);
    MPI_Bcast(primes.data(), npr, MPI_LONG, 0, MPI_COMM_WORLD);

    for (long p : primes) {
        if (rank == 0)
            cout << "\n=== PRIME p=" << p << " ===\n";

        ZZ_p::init(ZZ(p));
        int ub = floor(sqrt(p - 1));

        ofstream logfile;
        if (rank == 0) {
            logfile.open("results_p_" + to_string(p) + ".txt");
            logfile << "Prime p = " << p << "\n";
            logfile << "n,rows,search_time_sec,result,matrix_type\n";
        }

        for (int n = ub; n >= 2; --n) {
            if (rank == 0)
                cout << "n=" << n << ": generating rows...\n";

            auto gen_start = Clock::now();
            vector<Row> rows;
            generate_rows(n, p, rows);
            auto gen_end = Clock::now();

            if (rank == 0)
                cout << "  rows = " << rows.size() << "\n";

            if (rows.size() > 400000) {
                if (rank == 0)
                    logfile << n << "," << rows.size()
                            << ",0,SKIPPED,N/A\n";
                continue;
            }

            sort(rows.begin(), rows.end(),
                 [](const Row& a, const Row& b) {
                     return popcount(a.mask) < popcount(b.mask);
                 });

            double time_limit =
                (n >= 9 ? 15.0 :
                 n >= 7 ? 30.0 : 60.0);

            if (rank == 0)
                cout << "n=" << n << ": searching...\n";

            auto search_start = Clock::now();
            long long nodes = 0;
            SearchResult result = parallel_search(
                rows, n, MPI_COMM_WORLD, time_limit, nodes);
            auto search_end = Clock::now();

            double search_time =
                chrono::duration<double>(search_end - search_start).count();

            if (rank == 0) {
                string status = result.found ? "FOUND" : "TIMEOUT";
                string mat_type = result.found ?
                    (result.is_singular ? "SINGULAR" : "NON-SINGULAR") : "N/A";

                logfile << n << "," << rows.size() << ","
                        << fixed << setprecision(2)
                        << search_time << ","
                        << status << "," << mat_type << "\n";

                // Save the matrix to file if found
                if (result.found) {
                    mat_ZZ_p M;
                    M.SetDims(n, n);
                    for (int i = 0; i < n; ++i) {
                        for (int j = 0; j < n; ++j) {
                            M[i][j] = rows[result.row_indices[i]].elems[j];
                        }
                    }

                    string matrix_filename = "matrix_p" + to_string(p) +
                                           "_n" + to_string(n) + ".txt";
                    save_matrix(M, matrix_filename, p, n, result.is_singular);

                    cout << "  Saved matrix to " << matrix_filename << "\n";
                    cout << "  Matrix is " << mat_type << "\n";
                }
            }

            if (result.found) {
                if (rank == 0) {
                    cout << "✓ MAX DIMENSION = " << n << "\n";
                    logfile << "\nMAX_DIMENSION = " << n << "\n";
                    logfile.close();
                }
                break;
            }
        }
    }

    MPI_Finalize();
    return 0;
}

// max_d_mpi_safe.cpp
#include <mpi.h>
#include <NTL/mat_ZZ_p.h>
#include <NTL/ZZ.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <sys/stat.h>

using namespace std;
using namespace NTL;

// ---------------- Helpers ----------------
void create_dir(const string &path) {
    mkdir(path.c_str(), 0755);
}

struct Row {
    vector<int> vals;
    vector<bool> used; // track which numbers are used in this row
};

// Recursive function to generate rows summing to p-1
void generate_rows_recursive(long d, long p, int pos, int sum,
                             vector<int> &current, vector<Row> &rows,
                             vector<bool> &global_used, int last_val) {
    if(pos == d) {
        if(sum == p-1) {
            Row r;
            r.vals = current;
            r.used.assign(p, false);
            for(int v : current) r.used[v] = true;
            rows.push_back(r);
        }
        return;
    }
    int remaining = d - pos;
    for(int v = last_val + 1; v < p; ++v) {
        if(global_used[v]) continue; // avoid duplicates globally
        int min_sum = sum + v + (remaining-1)*(v+1);
        int max_sum = sum + v + (remaining-1)*(p-1);
        if(min_sum > p-1) break;
        if(max_sum < p-1) continue;

        current[pos] = v;
        global_used[v] = true;
        generate_rows_recursive(d, p, pos+1, sum+v, current, rows, global_used, v);
        global_used[v] = false;
    }
}

// Wrapper
vector<Row> generate_rows(long d, long p) {
    vector<Row> rows;
    vector<int> current(d, 0);
    vector<bool> global_used(p, false);
    generate_rows_recursive(d, p, 0, 0, current, rows, global_used, 0);
    return rows;
}

// Check if selected rows have unique numbers
bool rows_disjoint(const vector<Row> &rows, const vector<int> &indices) {
    vector<bool> used(rows[0].used.size(), false);
    for(int idx : indices) {
        for(int i = 1; i < rows[idx].used.size(); ++i) {
            if(rows[idx].used[i]) {
                if(used[i]) return false;
                used[i] = true;
            }
        }
    }
    return true;
}

// Build matrix from selected rows
Mat<ZZ_p> build_matrix(const vector<Row> &rows, const vector<int> &indices, long d) {
    Mat<ZZ_p> A; A.SetDims(d, d);
    for(int i = 0; i < d; ++i) {
        for(int j = 0; j < d; ++j) {
            A[i][j] = conv<ZZ_p>(rows[indices[i]].vals[j]);
        }
    }
    return A;
}

// ---------------- Main ----------------
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc < 2) {
        if(rank==0) cerr << "Usage: " << argv[0] << " <prime_file>\n";
        MPI_Finalize();
        return 1;
    }

    // --------------- Read primes from file ----------------
    vector<long> primes;
    if(rank == 0) {
        ifstream fin(argv[1]);
        long val;
        while(fin >> val) {
            if(val > 1) primes.push_back(val);
        }
        fin.close();
        if(primes.empty()) {
            cerr << "No valid primes found in file.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast number of primes
    uint64_t nprimes = primes.size();
    MPI_Bcast(&nprimes, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if(rank != 0) primes.resize(nprimes);
    MPI_Bcast(primes.data(), nprimes, MPI_LONG, 0, MPI_COMM_WORLD);

    // ---------------- Loop over primes ----------------
    for(long p : primes) {
        if(rank==0) cout << "=== PRIME " << p << " ===\n";

        // Initialize ZZ_p context
        ZZ_p::init(ZZ(p));

        long max_d = min<long>(p-1, (long)sqrt(p-1));
        bool found = false;
        long final_d = 0;
        vector<int> solution;

        for(long d = max_d; d >= 2 && !found; --d) {
            if(rank==0) cout << "Generating candidate rows d=" << d << endl;
            auto start = chrono::high_resolution_clock::now();

            vector<Row> rows;
            if(rank==0) rows = generate_rows(d, p);

            // Broadcast number of rows
            uint64_t R = rows.size();
            MPI_Bcast(&R, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
            if(rank != 0) rows.resize(R);
            int row_vals_size = d;
            for(uint64_t i = 0; i < R; ++i) {
                if(rank != 0) rows[i].vals.resize(d);
                MPI_Bcast(rows[i].vals.data(), d, MPI_INT, 0, MPI_COMM_WORLD);

                if(rank != 0) rows[i].used.assign(p, false);
                for(int v : rows[i].vals) rows[i].used[v] = true;
            }

            // Simple parallel combination search
            vector<int> indices(d);
            bool local_found = false;

            for(uint64_t idx = rank; idx < R; idx += size) {
                if(local_found) break;

                // naive: try all combinations with idx as first row
                indices[0] = idx;
                for(uint64_t j1 = 0; j1 < R && !local_found; ++j1) if(j1 != idx) {
                    if(d == 2) { indices[1] = j1; }
                    else {
                        for(uint64_t j2 = 0; j2 < R && !local_found; ++j2) if(j2!=idx && j2!=j1) {
                            if(d==3) indices[1]=j1, indices[2]=j2;
                            else continue;
                            if(!rows_disjoint(rows, indices)) continue;
                            Mat<ZZ_p> A = build_matrix(rows, indices, d);
                            if(determinant(A) != 0) {
                                solution = indices;
                                local_found = true;
                            }
                        }
                        if(d==2) {
                            if(!rows_disjoint(rows, indices)) continue;
                            Mat<ZZ_p> A = build_matrix(rows, indices, d);
                            if(determinant(A) != 0) {
                                solution = indices;
                                local_found = true;
                            }
                        }
                    }
                }
            }

            // Reduce to see if anyone found a solution
            int global_found_int = local_found ? 1 : 0;
            MPI_Allreduce(MPI_IN_PLACE, &global_found_int, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

            if(global_found_int) {
                found = true;
                final_d = d;
                if(rank==0) {
                    cout << "Found valid matrix of dimension " << d << endl;
                    string out_dir = "max_dim_results";
                    create_dir(out_dir);
                    string mat_file = out_dir + "/matrix_p" + to_string(p) + "_d" + to_string(d) + ".txt";
                    ofstream f(mat_file);
                    f << "Prime: " << p << "\nDimension: " << d << "\nMatrix:\n";
                    for(int r: solution) {
                        for(int v: rows[r].vals) f << v << " ";
                        f << "\n";
                    }
                    f.close();
                    cout << "Matrix written to " << mat_file << endl;
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        if(!found && rank==0) cout << "No valid matrix found for prime " << p << endl;
    }

    MPI_Finalize();
    return 0;
}


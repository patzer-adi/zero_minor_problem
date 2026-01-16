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

bool has_conflict(const vector<Vec<ZZ_p>>& current_matrix, const Vec<ZZ_p>& new_row) {
    for (auto& row : current_matrix)
        for (long i = 0; i < row.length(); i++)
            for (long j = 0; j < new_row.length(); j++)
                if (row[i] == new_row[j]) return true;
    return false;
}

bool all_elements_unique(const Mat<ZZ_p>& mat) {
    vector<ZZ_p> elements;
    long n = mat.NumRows();
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            elements.push_back(mat[i][j]);
    for (size_t i = 0; i < elements.size(); i++)
        for (size_t j = i + 1; j < elements.size(); j++)
            if (elements[i] == elements[j]) return false;
    return true;
}

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

void generate_matrices(long n, const vector<Vec<ZZ_p>>& rows, vector<Vec<ZZ_p>>& current_matrix,
                       vector<bool>& used, ofstream& matrix_file,
                       long& local_singular, long& local_non_singular) {
    if (current_matrix.size() == n) {
        Mat<ZZ_p> mat;
        mat.SetDims(n, n);
        for (long i = 0; i < n; i++)
            for (long j = 0; j < n; j++)
                mat[i][j] = current_matrix[i][j];
/* Dont write this: no matrix to file
        if (all_elements_unique(mat)) {
            ZZ_p det = determinant(mat);
            matrix_file << "n=" << n << " | Determinant=" << det
                        << " | " << (det == 0 ? "Singular" : "Non-Singular") << "\n";
            for (long i = 0; i < mat.NumRows(); i++) {
                for (long j = 0; j < mat.NumCols(); j++)
                    matrix_file << mat[i][j] << " ";
                matrix_file << "\n";
            }
            matrix_file << "\n"; */

            if (det == 0) local_singular++;
            else local_non_singular++;
        }
        return;
    }

    for (long i = 0; i < (long)rows.size(); i++) {
        if (!used[i] && !has_conflict(current_matrix, rows[i])) {
            used[i] = true;
            current_matrix.push_back(rows[i]);
            generate_matrices(n, rows, current_matrix, used, matrix_file, local_singular, local_non_singular);
            current_matrix.pop_back();
            used[i] = false;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        if (argc >= 1) cerr << "Usage: " << argv[0] << " <primes_file.txt>\n";
        return 1;
    }

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<long> primes;

    // Rank 0 reads the primes file and broadcasts to others
    if (rank == 0) {
        string primes_file = argv[1];
        ifstream in(primes_file);
        if (!in) {
            cerr << "Failed to open primes file: " << primes_file << "\n";
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

    // Broadcast number of primes
    long prime_count = primes.size();
    MPI_Bcast(&prime_count, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    if (rank != 0) primes.resize(prime_count);

    // Broadcast prime values (as array)
    if (prime_count > 0) {
        MPI_Bcast(primes.data(), prime_count, MPI_LONG, 0, MPI_COMM_WORLD);
    }

    // Each prime is processed sequentially, but for each prime all ranks work together.
    for (long pi = 0; pi < prime_count; ++pi) {
        long p = primes[pi];
        if (rank == 0) cout << "\n=== STARTING prime p=" << p << " (index " << pi << ") ===\n";

        ZZ_p::init(ZZ(p));

        // Create directory for this prime
        string pdir = "p_" + to_string(p);
        if (rank == 0) {
            fs::create_directories(pdir);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // For increasing n
        for (long n = 2; n <= p - 1; n++) {
            auto start_time = chrono::high_resolution_clock::now();

            vector<Vec<ZZ_p>> rows;
            vector<long> current;
            generate_rows_recursive(n, p, p - 1, 1, current, rows);

            if (rows.empty()) break; // no more rows possible for larger n

            if (rank == 0) {
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

            // prepare tasks (indices)
            vector<long> tasks;
            for (long i = 0; i < (long)rows.size(); i++) tasks.push_back(i);

            string filename = pdir + "/matrices_n_" + to_string(n) + "_rank_" + to_string(rank) + ".txt";
            ofstream matrix_file(filename, ios::out);

            long local_singular = 0, local_non_singular = 0;
            long local_processed = 0;

            if (rank == 0) cout << "p=" << p << " n=" << n << " -> rows=" << rows.size() << " tasks=" << tasks.size() << " (ranks=" << size << ")\n";

            // distribute tasks in round-robin manner
            for (long t = rank; t < (long)tasks.size(); t += size) {
                vector<Vec<ZZ_p>> current_matrix;
                vector<bool> used(rows.size(), false);

                long idx = tasks[t];
                current_matrix.push_back(rows[idx]);
                used[idx] = true;

                generate_matrices(n, rows, current_matrix, used, matrix_file, local_singular, local_non_singular);

                local_processed++;

                // periodic progress update: every 50 local tasks, reduce and root prints
                if (local_processed % 50 == 0) {
                    long global_processed = 0;
                    MPI_Reduce(&local_processed, &global_processed, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
                    // root prints approximate progress
                    if (rank == 0) {
                        cout << "  [progress] p=" << p << " n=" << n << " approx tasks processed=" << global_processed << "/" << tasks.size() << "\n";
                    }
                    // reset local counter after reporting to avoid double counting
                    local_processed = 0;
                }
            }

            // final small reduction for remaining local_processed
            long rem_local = local_processed;
            long global_rem = 0;
            MPI_Reduce(&rem_local, &global_rem, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0 && global_rem > 0) {
                cout << "  [progress] p=" << p << " n=" << n << " approx tasks processed (final chunk)=" << global_rem << "\n";
            }

            matrix_file.close();

            long total_singular = 0, total_non_singular = 0;
            MPI_Reduce(&local_singular, &total_singular, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_non_singular, &total_non_singular, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                // merge per-rank files into single matrices file inside pdir
                string final_path = pdir + "/matrices_n_" + to_string(n) + ".txt";
                ofstream final_file(final_path, ios::out);
                for (int r = 0; r < size; r++) {
                    string temp_file = pdir + "/matrices_n_" + to_string(n) + "_rank_" + to_string(r) + ".txt";
                    ifstream in(temp_file);
                    if (in) {
                        final_file << in.rdbuf();
                        in.close();
                        // optionally remove temp_file
                        fs::remove(temp_file);
                    }
                }

                auto end_time = chrono::high_resolution_clock::now();
                auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

                // write summary for this n (per requirement)
                string summary_path = pdir + "/summary_n_" + to_string(n) + ".txt";
                ofstream summary_file(summary_path, ios::out);
                summary_file << "=== SUMMARY FOR p=" << p << " n=" << n << " ===\n";
                summary_file << "Total matrices: " << (total_singular + total_non_singular) << "\n";
                summary_file << "Singular: " << total_singular << "\n";
                summary_file << "Non-Singular: " << total_non_singular << "\n";
                summary_file << "Computation time (ms): " << duration.count() << "\n";
                summary_file << "============================\n";
                summary_file.close();

                // also append summary to final matrices file
                final_file << "=== SUMMARY FOR n=" << n << " ===\n";
                final_file << "Total matrices: " << (total_singular + total_non_singular) << "\n";
                final_file << "Singular: " << total_singular << "\n";
                final_file << "Non-Singular: " << total_non_singular << "\n";
                final_file << "Computation time (ms): " << duration.count() << "\n";
                final_file << "============================\n\n";
                final_file.close();

                cout << "p=" << p << " n=" << n << " done. singular=" << total_singular << " non-singular=" << total_non_singular << " time(ms)=" << duration.count() << "\n";
            }

            MPI_Barrier(MPI_COMM_WORLD);
        } // end for n

        if (rank == 0) cout << "=== FINISHED prime p=" << p << " ===\n";
        MPI_Barrier(MPI_COMM_WORLD);
    } // end for primes

    MPI_Finalize();
    return 0;
}

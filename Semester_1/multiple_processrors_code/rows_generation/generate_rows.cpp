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
        string primes_file = argv[1];
        ifstream in(primes_file);
        if (!in) {
            cerr << "Failed to open primes file: " << primes_file << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        long val;
        while (in >> val) primes.push_back(val);
        in.close();
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

        // Loop for n values
        for (long n = 2; n <= p - 1; n++) {
            auto start_time = chrono::high_resolution_clock::now();

            vector<Vec<ZZ_p>> rows;
            vector<long> current;
            generate_rows_recursive(n, p, p - 1, 1, current, rows);

            if (rows.empty()) {
                if (rank == 0)
                    cout << "No rows generated for n=" << n
                         << ", stopping further n values for p=" << p << ".\n";
                break;
            }

            if (rank == 0) {
                auto end_time = chrono::high_resolution_clock::now();
                auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

                string rows_path = pdir + "/rows_n_" + to_string(n) + ".txt";
                ofstream row_file(rows_path);
                row_file << "=== ROWS FOR p=" << p << " n=" << n << " ===\n";
                row_file << "Total rows generated: " << rows.size() << "\n";
                row_file << "Computation time (ms): " << duration.count() << "\n";
                row_file << "============================\n\n";

                for (size_t i = 0; i < rows.size(); i++) {
                    row_file << "Row " << i + 1 << ": ";
                    for (long j = 0; j < rows[i].length(); j++)
                        row_file << rows[i][j] << " ";
                    row_file << "\n";
                }
                row_file.close();

                cout << "p=" << p << " n=" << n << " done. rows=" << rows.size()
                     << " time(ms)=" << duration.count() << "\n";
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (rank == 0) cout << "=== FINISHED prime p=" << p << " ===\n";
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

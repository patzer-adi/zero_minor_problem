#include <iostream>
#include <fstream>
#include <unordered_set>
#include <vector>
#include <random>
#include <string>
#include <chrono>
#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>

using namespace std;
using namespace NTL;
using namespace chrono;

// Hash function for matrix uniqueness
struct MatrixHash {
    size_t operator()(const vector<long>& v) const {
        size_t h = 0;
        for (auto x : v) {
            h ^= std::hash<long>()(x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
};

// Generate one random row of length 4 with elements in [1..P], sum = P-1
vector<long> generate_row(long P, mt19937 &gen) {
    const int n = 4;
    vector<long> row(n, 1); // start with 1 in each position
    long remaining = (P - 1) - n; // distribute remaining sum

    if (remaining < 0) {
        throw runtime_error("P too small to form valid rows with sum = P-1 and values >=1");
    }

    uniform_int_distribution<long> dist(0, remaining);

    // Distribute "remaining" among 4 slots
    for (int i = 0; i < n - 1; i++) {
        long add = dist(gen);
        row[i] += add;
        remaining -= add;
    }
    row[n - 1] += remaining;

    // Shuffle to randomize order
    shuffle(row.begin(), row.end(), gen);

    return row;
}

int main() {
    long P;
    cout << "Enter upper bound P for matrix entries: ";
    cin >> P;

    const int n = 4;
    const int NUM_MATRICES = 1000;

    unordered_set<vector<long>, MatrixHash> seen;  // store unique flattened matrices
    vector<Mat<ZZ>> matrices;

    random_device rd;
    mt19937 gen(rd());

    // Start timing generation
    auto gen_start = high_resolution_clock::now();

    while ((int)matrices.size() < NUM_MATRICES) {
        Mat<ZZ> A;
        A.SetDims(n, n);

        vector<long> flat;
        for (int i = 0; i < n; i++) {
            vector<long> row = generate_row(P, gen);
            for (int j = 0; j < n; j++) {
                A[i][j] = row[j];
                flat.push_back(row[j]);
            }
        }

        if (seen.insert(flat).second) {
            matrices.push_back(A);
        }
    }

    auto gen_end = high_resolution_clock::now();
    auto gen_duration = duration_cast<milliseconds>(gen_end - gen_start);

    cout << "Generated " << matrices.size() << " unique 4x4 matrices.\n";

    long singular_count = 0, nonsingular_count = 0;

    // Start timing determinant computations
    auto det_start = high_resolution_clock::now();

    ofstream outfile("nonsingular_matrices.txt");

    for (const auto& M : matrices) {
        ZZ det;
        determinant(det, M);
        if (det == 0) {
            singular_count++;
        } else {
            nonsingular_count++;
            // Store the matrix
            outfile << "Matrix (det=" << det << "):\n";
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    outfile << M[i][j] << "\t";
                }
                outfile << "\n";
            }
            outfile << "\n";
        }
    }

    auto det_end = high_resolution_clock::now();
    auto det_duration = duration_cast<milliseconds>(det_end - det_start);

    outfile.close();

    ofstream results("matrix_determinants_summary.txt");
    results << "Summary for " << NUM_MATRICES << " random 4x4 matrices (rows sum = P-1, entries in 1..P):\n\n";
    results << "Non-singular (det != 0): " << nonsingular_count << "\n";
    results << "Singular (det == 0): " << singular_count << "\n\n";
    results << "Time taken for generation: " << gen_duration.count() << " ms\n";
    results << "Time taken for determinant computations: " << det_duration.count() << " ms\n";
    results << "Total time: " << (gen_duration + det_duration).count() << " ms\n";
    results.close();

    cout << "Done! Summary written to matrix_determinants_summary.txt\n";
    cout << "All non-singular matrices stored in nonsingular_matrices.txt\n";

    return 0;
}

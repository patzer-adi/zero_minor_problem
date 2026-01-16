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

// Hash function to store matrices uniquely
struct MatrixHash {
    size_t operator()(const vector<long>& v) const {
        size_t h = 0;
        for (auto x : v) {
            h ^= std::hash<long>()(x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
};

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
    uniform_int_distribution<long> dist(1, P);

    // Start timer for generation
    auto gen_start = high_resolution_clock::now();

    // Generate unique matrices
    while ((int)matrices.size() < NUM_MATRICES) {
        Mat<ZZ> A;
        A.SetDims(n, n);

        vector<long> flat;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                long val = dist(gen);
                A[i][j] = val;
                flat.push_back(val);
            }
        }

        if (seen.insert(flat).second) {
            matrices.push_back(A); // only add if unique
        }
    }

    auto gen_end = high_resolution_clock::now();
    auto gen_duration = duration_cast<milliseconds>(gen_end - gen_start);

    cout << "Generated " << matrices.size() << " unique 4x4 matrices.\n";

    long singular_count = 0, nonsingular_count = 0;

    // Start timer for determinant computations
    auto det_start = high_resolution_clock::now();

    // Check determinants
    for (const auto& M : matrices) {
        ZZ det;
        determinant(det, M);
        if (det == 0) {
            singular_count++;
        } else {
            nonsingular_count++;
        }
    }

    auto det_end = high_resolution_clock::now();
    auto det_duration = duration_cast<milliseconds>(det_end - det_start);

    // Write results
    ofstream outfile("matrix_determinants_results.txt");
    outfile << "Results for " << NUM_MATRICES << " random 4x4 matrices (entries 1..P):\n\n";
    outfile << "Non-singular (det != 0): " << nonsingular_count << "\n";
    outfile << "Singular (det == 0): " << singular_count << "\n\n";
    outfile << "Time taken for generation: " << gen_duration.count() << " ms\n";
    outfile << "Time taken for determinant computations: " << det_duration.count() << " ms\n";
    outfile << "Total time: " << (gen_duration + det_duration).count() << " ms\n";
    outfile.close();

    cout << "Done! Results written to matrix_determinants_results.txt\n";

    return 0;
}

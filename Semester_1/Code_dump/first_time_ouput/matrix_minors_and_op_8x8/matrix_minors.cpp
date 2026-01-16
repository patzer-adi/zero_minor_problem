#include <iostream>
#include <fstream>
#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>
#include <NTL/vector.h>
#include <chrono>
#include <vector>  // ❗This is missing in your code

using namespace std;
using namespace NTL;
using namespace chrono;

// Generate all combinations of size k from {0, 1, ..., n-1}
void generate_combinations(long n, long k, vector<vector<long>> &combs, vector<long> current = {}, long start = 0) {
    if (current.size() == k) {
        combs.push_back(current);
        return;
    }
    for (long i = start; i < n; i++) {
        current.push_back(i);
        generate_combinations(n, k, combs, current, i + 1);
        current.pop_back();
    }
}

// Extract k×k submatrix using given row and column indices
Mat<ZZ> extract_submatrix(const Mat<ZZ>& A, const vector<long>& row_idx, const vector<long>& col_idx) {
    long k = row_idx.size();
    Mat<ZZ> sub;
    sub.SetDims(k, k);
    for (long i = 0; i < k; i++) {
        for (long j = 0; j < k; j++) {
            sub[i][j] = A[row_idx[i]][col_idx[j]];
        }
    }
    return sub;
}

void process_minors(const Mat<ZZ>& A, long k, const string& out_path) {
    long n = A.NumRows();
    vector<vector<long>> row_combs, col_combs;
    generate_combinations(n, k, row_combs);
    generate_combinations(n, k, col_combs);

    ofstream outfile(out_path);
    outfile << "Processing " << k << "x" << k << " minors...\n";
    outfile << "Matrix size: " << n << "x" << n << "\n";
    outfile << "Total minors: " << row_combs.size() * col_combs.size() << "\n";

    long singular_count = 0;
    auto start = high_resolution_clock::now();

    for (const auto& row_idx : row_combs) {
        for (const auto& col_idx : col_combs) {
            Mat<ZZ> minor = extract_submatrix(A, row_idx, col_idx);
            ZZ det;
            determinant(det, minor);
            outfile << "Rows: ";
            for (long i : row_idx) outfile << i << " ";
            outfile << "| Cols: ";
            for (long j : col_idx) outfile << j << " ";
            outfile << "| Det: " << det << "\n";

            if (det == 0) singular_count++;
        }
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(end - start);

    outfile << "\nSingular minors: " << singular_count << "\n";
    outfile << "Total minors: " << row_combs.size() * col_combs.size() << "\n";
    outfile << "Percentage singular: " << (100.0 * singular_count / (row_combs.size() * col_combs.size())) << "%\n";
    outfile << "Time taken: " << duration.count() << " seconds\n";
    outfile.close();
}

int main() {
    long n = 8;  // Size of the matrix (adjust as needed)
    Mat<ZZ> A;
    A.SetDims(n, n);

    // Optional: Set fixed seed for reproducibility
    // SetSeed(conv<ZZ>(12345));

    // Fill matrix with random values
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            A[i][j] = RandomBnd(100);
        }
    }

    // Save the random matrix for reference
    ofstream matrix_file("output_matrix.txt");
    matrix_file << "Random " << n << "x" << n << " Matrix:\n";
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            matrix_file << A[i][j] << "\t";
        }
        matrix_file << "\n";
    }
    matrix_file.close();

    // Loop over minor sizes k = 2 to n
    for (long k = 2; k <= 5; k++) {  // Be cautious with large k!
        string filename = "output_" + to_string(k) + "x" + to_string(k) + "_minors.txt";
        process_minors(A, k, filename);
    }

    return 0;
}

// rule_2_fixed.cpp
#include <NTL/mat_ZZ_p.h>
#include <NTL/ZZ.h>
#include <iostream>
#include <vector>
#include <set>
#include <fstream>
#include <chrono>
#include <algorithm>

using NTL::Mat;
using NTL::ZZ_p;
using NTL::ZZ;
using NTL::conv;
using NTL::determinant;

// store results (as NTL matrices)
std::vector<Mat<ZZ_p>> singular_matrices;
std::vector<Mat<ZZ_p>> nonsingular_matrices;

// Generate one row of length n (integers 1..p-1) that sums to target_sum,
// using only numbers NOT present in used_global. Uses combination generation
// (increasing order) and prunes if sum would exceed target_sum.
void generate_row(int n, long p_minus_1, long target_sum,
                  std::vector<long>& current_row,
                  std::set<long>& used_global,
                  std::set<long>& used_in_row,
                  std::vector<std::vector<long>>& valid_rows,
                  long start = 1, long current_sum = 0) {
    if ((int)current_row.size() == n) {
        if (current_sum == target_sum) valid_rows.push_back(current_row);
        return;
    }

    // Candidates are 1..p-1
    for (long i = start; i <= p_minus_1; ++i) {
        if (used_global.count(i) || used_in_row.count(i)) continue;
        // pruning
        if (current_sum + i > target_sum) break;
        // minimal remaining sum check: if even choosing the smallest remaining numbers cannot fill the row to target, skip
        // compute how many more numbers needed:
        int need = n - (int)current_row.size() - 1;
        if (need > 0) {
            // compute minimal possible sum using next `need` smallest available numbers > i
            long min_possible = 0;
            long cnt = 0;
            for (long j = i + 1; j <= p_minus_1 && cnt < need; ++j) {
                if (used_global.count(j)) continue;
                min_possible += j;
                ++cnt;
            }
            if (cnt < need) break; // not enough numbers remain
            if (current_sum + i + min_possible > target_sum) {
                // It's still possible numbers could be larger -> but if already exceed, then skip trying this i
                // however this check is conservative; we continue normally (no strict skip)
            }
        }

        current_row.push_back(i);
        used_in_row.insert(i);

        generate_row(n, p_minus_1, target_sum, current_row, used_global, used_in_row, valid_rows, i + 1, current_sum + i);

        current_row.pop_back();
        used_in_row.erase(i);
    }
}

// Recursive matrix generator: build matrix row-by-row by choosing disjoint rows
// (each row is a vector<long> of size n that sums to target_sum).
// current_matrix collects rows as integer vectors. When matrix is complete, convert
// to NTL::Mat<ZZ_p> and test determinant.
void generate_matrix(int n, long p_minus_1, long target_sum,
                     std::vector<std::vector<long>>& current_matrix,
                     std::set<long>& used_global, int& matrices_generated,
                     int max_matrices, std::ofstream& matrix_file, int &print_counter) {

    if (matrices_generated >= max_matrices) return;

    if ((int)current_matrix.size() == n) {
        // Build NTL matrix
        Mat<ZZ_p> mat;
        mat.SetDims(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                mat[i][j] = ZZ_p(current_matrix[i][j]);

        matrices_generated++;
        bool singular = (determinant(mat) == 0);
        if (singular) singular_matrices.push_back(mat);
        else nonsingular_matrices.push_back(mat);

        matrix_file << "Matrix #" << matrices_generated << ": " << (singular ? "Singular" : "Non-singular") << "\n";
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) matrix_file << current_matrix[i][j] << " ";
            matrix_file << "\n";
        }
        matrix_file << "\n";

        // occasional progress print to stdout
        if (++print_counter % 50 == 0) {
            std::cout << "Generated " << matrices_generated << " matrices so far...\n";
        }

        return;
    }

    // generate all valid candidate rows drawn from currently unused elements
    std::vector<std::vector<long>> valid_rows;
    std::vector<long> row;
    std::set<long> used_in_row;
    generate_row(n, p_minus_1, target_sum, row, used_global, used_in_row, valid_rows, 1, 0);

    // iterate candidate rows; to avoid duplicates due to row-order, enforce lexicographic ordering:
    // ensure the first element of this chosen row is >= first element of previous row (if any).
    long min_first = 1;
    if (!current_matrix.empty()) min_first = current_matrix.back().front();

    for (auto &r : valid_rows) {
        if ((int)r.size() != n) continue;
        if (r.front() < min_first) continue; // enforce non-decreasing first elements to reduce symmetric duplicates

        // mark as used globally
        for (auto x : r) used_global.insert(x);

        current_matrix.push_back(r);
        generate_matrix(n, p_minus_1, target_sum, current_matrix, used_global, matrices_generated, max_matrices, matrix_file, print_counter);
        current_matrix.pop_back();

        // unmark
        for (auto x : r) used_global.erase(x);

        if (matrices_generated >= max_matrices) return;
    }
}

int main() {
    long p, n, max_matrices;
    std::cout << "Enter prime number p: ";
    if (!(std::cin >> p)) return 1;
    std::cout << "Enter matrix size n: ";
    if (!(std::cin >> n)) return 1;
    std::cout << "Enter maximum number of matrices to generate: ";
    if (!(std::cin >> max_matrices)) return 1;

    if (n <= 0 || p <= 0 || max_matrices <= 0) {
        std::cout << "All inputs must be positive integers.\n";
        return 1;
    }
    long p_minus_1 = p - 1;
    if (n > p_minus_1) {
        std::cout << "Matrix size cannot be larger than p-1.\n";
        return 1;
    }
    // For a full partition of 1..p-1 into n rows of length n we need n*n == p-1
    if (n * n != p_minus_1) {
        std::cout << "To partition 1.." << p_minus_1 << " into " << n << " rows of size " << n
                  << " you need n*n == p-1. Currently n*n = " << (n*n)
                  << " and p-1 = " << p_minus_1 << ". Exiting.\n";
        return 1;
    }

    ZZ_p::init(ZZ(p));
    long target_sum = p_minus_1; // integer target for row sums

    auto start_time = std::chrono::high_resolution_clock::now();

    std::ofstream matrix_file("matrices.txt");
    std::vector<std::vector<long>> current_matrix;
    std::set<long> used_global;
    int matrices_generated = 0;
    int print_counter = 0;

    generate_matrix(n, p_minus_1, target_sum, current_matrix, used_global, matrices_generated, max_matrices, matrix_file, print_counter);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    matrix_file << "=== SUMMARY ===\n";
    matrix_file << "Total matrices generated: " << matrices_generated << "\n";
    matrix_file << "Singular matrices: " << singular_matrices.size() << "\n";
    matrix_file << "Non-singular matrices: " << nonsingular_matrices.size() << "\n";
    matrix_file << "Computation time (ms): " << duration.count() << "\n";
    matrix_file.close();

    std::cout << "Total matrices generated: " << matrices_generated << "\n";
    std::cout << "Singular matrices: " << singular_matrices.size() << "\n";
    std::cout << "Non-singular matrices: " << nonsingular_matrices.size() << "\n";
    std::cout << "Computation time: " << duration.count() << " ms\n";

    return 0;
}


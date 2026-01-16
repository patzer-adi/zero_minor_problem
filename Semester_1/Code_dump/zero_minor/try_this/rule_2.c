#include <NTL/mat_ZZ_p.h>
#include <iostream>
#include <vector>
#include <set>
#include <fstream>
#include <chrono>

using namespace std;
using namespace NTL;

vector<Mat<ZZ_p>> singular_matrices;
vector<Mat<ZZ_p>> nonsingular_matrices;

// Recursive function to generate a row of length n with sum = target_sum, avoiding used elements
void generate_row(int n, ZZ_p target_sum, vector<ZZ_p>& current_row, std::set<long>& used_elements,
                  vector<vector<ZZ_p>>& valid_rows, long start = 1) {
    if (current_row.size() == n) {
        ZZ_p sum = ZZ_p(0);
        for (auto x : current_row) sum += x;
        if (sum == target_sum) valid_rows.push_back(current_row);
        return;
    }

    for (long i = start; i <= conv<long>(target_sum); i++) {
        if (used_elements.count(i)) continue;
        current_row.push_back(ZZ_p(i));
        used_elements.insert(i);
        generate_row(n, target_sum, current_row, used_elements, valid_rows, 1);
        current_row.pop_back();
        used_elements.erase(i);
    }
}

// Recursive matrix generator
void generate_matrix(int n, ZZ_p target_sum, vector<vector<ZZ_p>>& current_matrix, std::set<long>& used_elements,
                     int& matrices_generated, int max_matrices, ofstream& matrix_file) {

    if (matrices_generated >= max_matrices) return;

    if (current_matrix.size() == n) {
        Mat<ZZ_p> mat;
        mat.SetDims(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                mat[i][j] = current_matrix[i][j];

        matrices_generated++;
        bool singular = (determinant(mat) == 0);
        if (singular) singular_matrices.push_back(mat);
        else nonsingular_matrices.push_back(mat);

        matrix_file << "Matrix #" << matrices_generated << ": " << (singular ? "Singular" : "Non-singular") << "\n";
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) matrix_file << mat[i][j] << " ";
            matrix_file << "\n";
        }
        matrix_file << "\n";
        return;
    }

    vector<vector<ZZ_p>> valid_rows;
    vector<ZZ_p> row;
    generate_row(n, target_sum, row, used_elements, valid_rows);

    for (auto& r : valid_rows) {
        for (auto x : r) used_elements.insert(conv<long>(x));
        current_matrix.push_back(r);

        generate_matrix(n, target_sum, current_matrix, used_elements, matrices_generated, max_matrices, matrix_file);

        current_matrix.pop_back();
        for (auto x : r) used_elements.erase(conv<long>(x));

        if (matrices_generated >= max_matrices) return;
    }
}

int main() {
    long p, n, max_matrices;
    cout << "Enter prime number p: ";
    cin >> p;
    cout << "Enter matrix size n: ";
    cin >> n;
    cout << "Enter maximum number of matrices to generate: ";
    cin >> max_matrices;

    if (n <= 0 || p <= 0 || max_matrices <= 0) {
        cout << "All inputs must be positive integers.\n";
        return 1;
    }
    if (n > p - 1) {
        cout << "Matrix size cannot be larger than p-1.\n";
        return 1;
    }

    ZZ_p::init(ZZ(p));
    ZZ_p target = ZZ_p(p - 1);

    auto start_time = chrono::high_resolution_clock::now();

    ofstream matrix_file("matrices.txt");
    vector<vector<ZZ_p>> current_matrix;
    std::set<long> used_elements;
    int matrices_generated = 0;

    generate_matrix(n, target, current_matrix, used_elements, matrices_generated, max_matrices, matrix_file);

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    matrix_file << "=== SUMMARY ===\n";
    matrix_file << "Total matrices generated: " << matrices_generated << "\n";
    matrix_file << "Singular matrices: " << singular_matrices.size() << "\n";
    matrix_file << "Non-singular matrices: " << nonsingular_matrices.size() << "\n";
    matrix_file << "Computation time (ms): " << duration.count() << "\n";
    matrix_file.close();

    cout << "Total matrices generated: " << matrices_generated << "\n";
    cout << "Computation time: " << duration.count() << " ms\n";

    return 0;
}


#include <NTL/mat_ZZ_p.h>
#include <NTL/vector.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

using namespace std;
using namespace NTL;

vector<Mat<ZZ_p>> singular_matrices;
vector<Mat<ZZ_p>> nonsingular_matrices;
vector<Vec<ZZ_p>> generated_rows;

// Check if matrix is singular
bool is_singular_det(const Mat<ZZ_p>& mat) {
    ZZ_p d = determinant(mat);
    return d == 0;
}

// Check if all elements in matrix are unique
bool all_elements_unique(const Mat<ZZ_p>& mat) {
    vector<ZZ_p> elements;
    long n = mat.NumRows();
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            elements.push_back(mat[i][j]);
    for (long i = 0; i < elements.size(); i++)
        for (long j = i+1; j < elements.size(); j++)
            if (elements[i]==elements[j]) return false;
    return true;
}

// Check if a new row conflicts with current matrix elements
bool has_conflict(const vector<Vec<ZZ_p>>& current_matrix, const Vec<ZZ_p>& new_row) {
    for (auto& row : current_matrix)
        for (long i = 0; i < row.length(); i++)
            for (long j = 0; j < new_row.length(); j++)
                if (row[i] == new_row[j]) return true;
    return false;
}

// Recursive function to generate rows of length n, distinct elements 1..p-1, sum = target_sum
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

    for (long i = start; i <= p-1; i++) {
        // Skip if already in current row
        bool used = false;
        for (auto x : current) if (x==i) { used=true; break; }
        if (used) continue;

        current.push_back(i);
        generate_rows_recursive(n, p, target_sum, i+1, current, rows);
        current.pop_back();
    }
}

// Recursive function to generate matrices from rows
void generate_matrices(long n, const vector<Vec<ZZ_p>>& rows, vector<Vec<ZZ_p>>& current_matrix,
                       vector<bool>& used, long max_matrices, long& generated,
                       bool& stop_generation, ofstream& matrix_file) {
    if (stop_generation || generated >= max_matrices) return;

    if (current_matrix.size() == n) {
        Mat<ZZ_p> mat;
        mat.SetDims(n, n);
        for (long i=0;i<n;i++)
            for (long j=0;j<n;j++)
                mat[i][j] = current_matrix[i][j];

        if (all_elements_unique(mat)) {
            generated++;
            bool singular = is_singular_det(mat);
            if (singular) singular_matrices.push_back(mat);
            else nonsingular_matrices.push_back(mat);

            // Write matrix to file
            matrix_file << "Matrix #" << generated << ": " << (singular ? "Singular" : "Non-singular") << "\n";
            for (long i=0;i<n;i++){
                for (long j=0;j<n;j++) matrix_file << mat[i][j] << " ";
                matrix_file << "\n";
            }
            matrix_file << "All elements unique: YES | Singular: " << (singular?"YES":"NO") << "\n\n";

            if (generated >= max_matrices) { stop_generation = true; return; }
        }
        return;
    }

    for (long i = 0; i < rows.size() && !stop_generation; i++) {
        if (!used[i] && !has_conflict(current_matrix, rows[i])) {
            used[i] = true;
            current_matrix.push_back(rows[i]);
            generate_matrices(n, rows, current_matrix, used, max_matrices, generated, stop_generation, matrix_file);
            current_matrix.pop_back();
            used[i] = false;
        }
    }
}

int main() {
    long p, n, max_matrices;
    cout << "Enter prime p: "; cin >> p;
    cout << "Enter matrix size n: "; cin >> n;
    cout << "Enter max number of matrices to generate: "; cin >> max_matrices;

    if (n <= 0 || p <= 1 || max_matrices <= 0) { cout << "Invalid input\n"; return 1; }
    if (n > p-1) { cout << "Matrix size cannot exceed p-1\n"; return 1; }

    ZZ_p::init(ZZ(p));
    long target_sum = p-1;

    auto start = chrono::high_resolution_clock::now();

    cout << "Generating rows with sum=" << target_sum << "...\n";
    vector<Vec<ZZ_p>> rows;
    vector<long> current;
    generate_rows_recursive(n, p, target_sum, 1, current, rows);
    generated_rows = rows;
    cout << "Total rows generated: " << rows.size() << "\n";

    // Save rows
    ofstream row_file("rows.txt");
    for (long i=0;i<rows.size();i++){
        row_file << "Row " << i+1 << ": ";
        ZZ_p s = conv<ZZ_p>(0);
        for (long j=0;j<n;j++){ row_file << rows[i][j] << " "; s += rows[i][j]; }
        row_file << "Sum=" << s << "\n";
    }
    row_file.close();

    // Generate matrices
    cout << "Generating matrices...\n";
    ofstream matrix_file("matrices.txt");
    vector<Vec<ZZ_p>> current_matrix;
    vector<bool> used(rows.size(), false);
    long generated = 0;
    bool stop = false;
    generate_matrices(n, rows, current_matrix, used, max_matrices, generated, stop, matrix_file);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    // Write summary
    matrix_file << "=== SUMMARY ===\n";
    matrix_file << "Total matrices generated: " << generated << "\n";
    matrix_file << "Singular matrices: " << singular_matrices.size() << "\n";
    matrix_file << "Non-singular matrices: " << nonsingular_matrices.size() << "\n";
    matrix_file << "Total rows generated: " << rows.size() << "\n";
    matrix_file << "Computation time (ms): " << duration.count() << "\n";
    matrix_file.close();

    cout << "Matrices generated: " << generated << "\n";
    cout << "Computation time: " << duration.count() << " ms\n";

    return 0;
}


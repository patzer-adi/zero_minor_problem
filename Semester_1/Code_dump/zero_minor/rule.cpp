#include <NTL/mat_ZZ_p.h>  // For Mat<ZZ_p>, gauss(), determinant()
#include <NTL/vector.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

using namespace std;
using namespace NTL;

// Global variables to store results
vector<Mat<ZZ_p>> singular_matrices;
vector<Mat<ZZ_p>> nonsingular_matrices;
vector<Vec<ZZ_p>> generated_rows;

// Custom comparison function for ZZ_p elements
bool zz_p_less(const ZZ_p& a, const ZZ_p& b) {
    return rep(a) < rep(b);
}

// Function to generate all possible rows with sum = p-1 and unique elements
vector<Vec<ZZ_p>> generate_rows(long n, ZZ_p p_minus_one) {
    vector<Vec<ZZ_p>> rows;
    long p = conv<long>(ZZ_p::modulus());
    
    // Generate all combinations of n distinct elements from 1 to p-1
    vector<long> indices(n);
    for (long i = 0; i < n; i++) {
        indices[i] = i + 1;  // start from 1
    }
    
    while (true) {
        Vec<ZZ_p> row;
        row.SetLength(n);
        ZZ_p sum = conv<ZZ_p>(0);
        
        for (long i = 0; i < n; i++) {
            row[i] = conv<ZZ_p>(indices[i]);
            sum += row[i];
        }
        
        if (sum == p_minus_one) {
            rows.push_back(row);
        }
        
        // Next combination
        long i = n - 1;
        while (i >= 0 && indices[i] == p - n + i) i--;
        if (i < 0) break;
        indices[i]++;
        for (long j = i + 1; j < n; j++) indices[j] = indices[j - 1] + 1;
    }
    
    return rows;
}

// Function to check if all elements in matrix are unique
bool all_elements_unique(const Mat<ZZ_p>& mat) {
    long n = mat.NumRows();
    vector<ZZ_p> elements;
    
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            elements.push_back(mat[i][j]);
    
    sort(elements.begin(), elements.end(), zz_p_less);
    
    for (long i = 1; i < elements.size(); i++)
        if (elements[i] == elements[i - 1]) return false;
    
    return true;
}

// Function to check if matrix is singular using determinant
bool is_singular_det(const Mat<ZZ_p>& mat) {
    ZZ_p d = determinant(mat);
    return d == 0;
}

// Function to check if a new row conflicts with existing matrix elements
bool has_conflict(const vector<Vec<ZZ_p>>& current_matrix, const Vec<ZZ_p>& new_row) {
    if (current_matrix.empty()) return false;
    
    vector<ZZ_p> all_elements;
    for (const auto& row : current_matrix)
        for (long j = 0; j < row.length(); j++)
            all_elements.push_back(row[j]);
    
    for (long j = 0; j < new_row.length(); j++)
        for (const auto& elem : all_elements)
            if (elem == new_row[j]) return true;
    
    return false;
}

// Recursive function to generate matrices with unique rows and elements
void generate_matrices(long n, const vector<Vec<ZZ_p>>& rows, 
                      vector<Vec<ZZ_p>>& current_matrix, 
                      vector<bool>& used_row_indices,
                      long max_matrices, long& matrices_generated,
                      bool& stop_generation) {
    
    if (stop_generation || matrices_generated >= max_matrices) return;
    
    if (current_matrix.size() == n) {
        Mat<ZZ_p> mat;
        mat.SetDims(n, n);
        
        for (long i = 0; i < n; i++)
            for (long j = 0; j < n; j++)
                mat[i][j] = current_matrix[i][j];
        
        if (all_elements_unique(mat)) {
            matrices_generated++;
            
            if (is_singular_det(mat)) {
                singular_matrices.push_back(mat);
                cout << "Found singular matrix #" << singular_matrices.size() << endl;
            } else {
                nonsingular_matrices.push_back(mat);
                cout << "Found non-singular matrix #" << nonsingular_matrices.size() << endl;
            }
            
            if (matrices_generated >= max_matrices) {
                stop_generation = true;
                cout << "Reached maximum number of matrices: " << max_matrices << endl;
            }
        }
        return;
    }
    
    for (long i = 0; i < rows.size() && !stop_generation; i++) {
        if (!used_row_indices[i] && !has_conflict(current_matrix, rows[i])) {
            used_row_indices[i] = true;
            current_matrix.push_back(rows[i]);
            
            generate_matrices(n, rows, current_matrix, used_row_indices, 
                              max_matrices, matrices_generated, stop_generation);
            
            current_matrix.pop_back();
            used_row_indices[i] = false;
        }
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
        cout << "Error: All inputs must be positive integers." << endl;
        return 1;
    }
    
    if (n > p) {
        cout << "Error: Matrix size n cannot be larger than prime p." << endl;
        return 1;
    }
    
    ZZ_p::init(ZZ(p));
    ZZ_p p_minus_one = conv<ZZ_p>(p - 1);
    
    cout << "Initialized ZZ_p with modulus " << p << endl;
    cout << "Target row sum: " << p_minus_one << endl;
    
    auto start_time = chrono::high_resolution_clock::now();
    
    cout << "Generating rows with sum = " << p_minus_one << "..." << endl;
    vector<Vec<ZZ_p>> rows = generate_rows(n, p_minus_one);
    generated_rows = rows;
    
    cout << "Generated " << rows.size() << " possible rows." << endl;
    
    if (rows.size() < n) {
        cout << "Error: Not enough rows (" << rows.size() << ") to form a " 
             << n << "x" << n << " matrix." << endl;
        return 1;
    }
    
    cout << "Generating up to " << max_matrices << " matrices..." << endl;
    
    vector<Vec<ZZ_p>> current_matrix;
    vector<bool> used_row_indices(rows.size(), false);
    long matrices_generated = 0;
    bool stop_generation = false;
    
    generate_matrices(n, rows, current_matrix, used_row_indices, 
                      max_matrices, matrices_generated, stop_generation);
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    
    cout << "\n=== FINAL RESULTS ===" << endl;
    cout << "Prime modulus: " << p << endl;
    cout << "Matrix size: " << n << "x" << n << endl;
    cout << "Possible rows generated: " << rows.size() << endl;
    cout << "Total matrices generated: " << matrices_generated << endl;
    cout << "Singular matrices: " << singular_matrices.size() << endl;
    cout << "Non-singular matrices: " << nonsingular_matrices.size() << endl;
    cout << "Computation time: " << duration.count() << " ms" << endl;
    
    if (!singular_matrices.empty()) {
        cout << "\nSample singular matrix:" << endl;
        cout << singular_matrices[0] << endl;
        cout << "Verification: This matrix is " 
             << (is_singular_det(singular_matrices[0]) ? "singular" : "non-singular") << endl;
    }
    
    if (!nonsingular_matrices.empty()) {
        cout << "\nSample non-singular matrix:" << endl;
        cout << nonsingular_matrices[0] << endl;
        cout << "Verification: This matrix is " 
             << (is_singular_det(nonsingular_matrices[0]) ? "singular" : "non-singular") << endl;
    }
    
    cout << "\nFirst 5 generated rows (showing elements and sums):" << endl;
    for (long i = 0; i < min(5L, (long)generated_rows.size()); i++) {
        cout << "Row " << i + 1 << ": [";
        ZZ_p row_sum = conv<ZZ_p>(0);
        for (long j = 0; j < n; j++) {
            cout << generated_rows[i][j];
            row_sum += generated_rows[i][j];
            if (j < n - 1) cout << ", ";
        }
        cout << "] Sum = " << row_sum;
        if (row_sum == p_minus_one) cout << " ✓ (correct)";
        else cout << " ✗ (should be " << p_minus_one << ")";
        cout << endl;
    }
    
    if (matrices_generated > 0) {
        double singular_ratio = (double)singular_matrices.size() / matrices_generated * 100;
        double nonsingular_ratio = (double)nonsingular_matrices.size() / matrices_generated * 100;
        cout << "\nRatios: Singular = " << singular_ratio << "%, Non-singular = " 
             << nonsingular_ratio << "%" << endl;
    }
    
    return 0;
}


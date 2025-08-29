#include <iostream>
#include <fstream>
#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>
#include <chrono>
#include <vector>
#include <string>

using namespace std;
using namespace NTL;
using namespace chrono;

// Generate all combinations of size k from {0, 1, ..., n-1}
void generate_combinations(long n, long k, vector<vector<long>>& combs) {
    vector<long> current(k);
    
    // Initialize first combination
    for (long i = 0; i < k; i++) {
        current[i] = i;
    }
    
    while (true) {
        combs.push_back(current);
        
        // Find rightmost element that can be incremented
        long i = k - 1;
        while (i >= 0 && current[i] == n - k + i) {
            i--;
        }
        
        if (i < 0) break; // No more combinations
        
        // Increment current[i] and reset elements to its right
        current[i]++;
        for (long j = i + 1; j < k; j++) {
            current[j] = current[j-1] + 1;
        }
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

// Process all k×k minors and write results to file
void process_minors(const Mat<ZZ>& A, long k, const string& filename) {
    long n = A.NumRows();
    
    // Generate all row and column combinations
    vector<vector<long>> row_combs, col_combs;
    generate_combinations(n, k, row_combs);
    generate_combinations(n, k, col_combs);
    
    ofstream outfile(filename);
    if (!outfile) {
        cout << "Error: Could not open file " << filename << endl;
        return;
    }
    
    auto start = high_resolution_clock::now();
    
    outfile << "Processing " << k << "×" << k << " minors of " << n << "×" << n << " matrix\n";
    outfile << "Total minors to compute: " << row_combs.size() * col_combs.size() << "\n\n";
    
    long singular_count = 0;
    long total_minors = 0;
    
    // Process each minor
    for (const auto& row_idx : row_combs) {
        for (const auto& col_idx : col_combs) {
            Mat<ZZ> minor = extract_submatrix(A, row_idx, col_idx);
            ZZ det;
            determinant(det, minor);
            
            // Write minor information
            outfile << "Minor " << (total_minors + 1) << ": ";
            outfile << "Rows[";
            for (size_t i = 0; i < row_idx.size(); i++) {
                outfile << row_idx[i];
                if (i < row_idx.size() - 1) outfile << ",";
            }
            outfile << "] Cols[";
            for (size_t i = 0; i < col_idx.size(); i++) {
                outfile << col_idx[i];
                if (i < col_idx.size() - 1) outfile << ",";
            }
            outfile << "] Det: " << det << "\n";
            
            if (det == 0) {
                singular_count++;
            }
            total_minors++;
        }
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    
    // Write summary statistics
    outfile << "\n=== SUMMARY ===\n";
    outfile << "Total " << k << "×" << k << " minors: " << total_minors << "\n";
    outfile << "Singular minors: " << singular_count << "\n";
    outfile << "Non-singular minors: " << (total_minors - singular_count) << "\n";
    outfile << "Percentage singular: " << (100.0 * singular_count / total_minors) << "%\n";
    outfile << "Computation time: " << duration.count() << " milliseconds\n";
    
    outfile.close();
    
    cout << "Completed " << k << "×" << k << " minors analysis. Results saved to " << filename << endl;
    cout << "Found " << singular_count << "/" << total_minors << " singular minors (" 
         << (100.0 * singular_count / total_minors) << "%)\n";
}

// Save matrix to file for reference
void save_matrix(const Mat<ZZ>& A, const string& filename) {
    ofstream file(filename);
    long n = A.NumRows();
    
    file << "Matrix (" << n << "×" << n << "):\n";
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            file << A[i][j];
            if (j < n - 1) file << "\t";
        }
        file << "\n";
    }
    file.close();
}

int main() {
    // Configuration
    const long matrix_size = 25;  // Size of the matrix
    const long max_minor_size = 24;  // Maximum minor size to analyze
    const long random_range = 50;   // Range for random values [0, random_range)
    
    cout << "Matrix Minor Analysis Program\n";
    cout << "============================\n";
    cout << "Matrix size: " << matrix_size << "×" << matrix_size << "\n";
    cout << "Random range: [0, " << random_range << ")\n";
    cout << "Analyzing minors from 2×2 to " << max_minor_size << "×" << max_minor_size << "\n\n";
    
    // Generate random matrix
    Mat<ZZ> A;
    A.SetDims(matrix_size, matrix_size);
    
    // Optional: Set seed for reproducible results
    // SetSeed(conv<ZZ>(12345));
    
    for (long i = 0; i < matrix_size; i++) {
        for (long j = 0; j < matrix_size; j++) {
            A[i][j] = RandomBnd(random_range);
        }
    }
    
    // Save the matrix
    save_matrix(A, "matrix.txt");
    cout << "Generated matrix saved to matrix.txt\n";
    
    // Process minors for different sizes
    for (long k = 2; k <= min(max_minor_size, matrix_size); k++) {
        string filename = "minors_" + to_string(k) + "x" + to_string(k) + ".txt";
        process_minors(A, k, filename);
    }
    
    cout << "\nAll analyses completed successfully!\n";
    return 0;
}
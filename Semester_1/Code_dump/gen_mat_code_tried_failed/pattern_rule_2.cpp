#include <iostream>
#include <fstream>
#include <unordered_set>
#include <vector>
#include <random>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <set>
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

// Efficient matrix generator using backtracking
class MatrixGenerator {
private:
    long P;
    int n;
    long target_sum;
    vector<vector<long>> current_matrix;
    set<long> used_elements;
    vector<set<vector<long>>> used_rows_per_matrix;
    mt19937& gen;
    
public:
    MatrixGenerator(long P, int n, long target_sum, mt19937& gen) 
        : P(P), n(n), target_sum(target_sum), gen(gen) {
        current_matrix.resize(n, vector<long>(n));
    }
    
    // Check if we can complete current row with remaining sum using unused elements
    bool canCompleteRow(int row, int col, long remaining_sum, const set<long>& temp_used) {
        if (col == n) {
            return remaining_sum == 0;
        }
        
        // How many positions left in this row?
        int positions_left = n - col;
        
        // Find available numbers
        vector<long> available;
        for (long val = 1; val <= P; val++) {
            if (used_elements.find(val) == used_elements.end() && 
                temp_used.find(val) == temp_used.end()) {
                available.push_back(val);
            }
        }
        
        if ((int)available.size() < positions_left) {
            return false; // Not enough unique numbers left
        }
        
        // Check if we can make the remaining sum with available numbers
        sort(available.begin(), available.end());
        
        // Minimum possible sum with smallest available numbers
        long min_sum = 0;
        for (int i = 0; i < positions_left; i++) {
            min_sum += available[i];
        }
        
        // Maximum possible sum with largest available numbers  
        long max_sum = 0;
        for (int i = available.size() - positions_left; i < (int)available.size(); i++) {
            max_sum += available[i];
        }
        
        return remaining_sum >= min_sum && remaining_sum <= max_sum;
    }
    
    // Check if we can complete the entire matrix
    bool canCompleteMatrix(int current_row, int total_elements_used) {
        int remaining_rows = n - current_row;
        int remaining_elements = n * n - total_elements_used;
        long remaining_total_sum = remaining_rows * target_sum;
        
        if (remaining_elements == 0) {
            return remaining_total_sum == 0;
        }
        
        // Find available numbers
        vector<long> available;
        for (long val = 1; val <= P; val++) {
            if (used_elements.find(val) == used_elements.end()) {
                available.push_back(val);
            }
        }
        
        if ((int)available.size() < remaining_elements) {
            return false;
        }
        
        // Check if remaining sum is achievable
        sort(available.begin(), available.end());
        
        long min_sum = 0, max_sum = 0;
        for (int i = 0; i < remaining_elements; i++) {
            min_sum += available[i];
        }
        for (int i = available.size() - remaining_elements; i < (int)available.size(); i++) {
            max_sum += available[i];
        }
        
        return remaining_total_sum >= min_sum && remaining_total_sum <= max_sum;
    }
    
    // Backtracking function to generate a valid matrix
    bool generateMatrix(int row, int col, long current_row_sum, set<vector<long>>& used_rows) {
        if (row == n) {
            // Check if all rows are unique
            vector<long> matrix_flat;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    matrix_flat.push_back(current_matrix[i][j]);
                }
            }
            return true;
        }
        
        if (col == n) {
            // Completed a row, check if sum is correct and row is unique
            if (current_row_sum != target_sum) {
                return false;
            }
            
            vector<long> current_row_vec = current_matrix[row];
            if (used_rows.find(current_row_vec) != used_rows.end()) {
                return false; // Row already used
            }
            
            used_rows.insert(current_row_vec);
            bool result = generateMatrix(row + 1, 0, 0, used_rows);
            if (!result) {
                used_rows.erase(current_row_vec); // Backtrack
            }
            return result;
        }
        
        // Early pruning checks
        if (!canCompleteMatrix(row, row * n + col)) {
            return false;
        }
        
        // Create a temporary set to track elements used in current row
        set<long> temp_row_used;
        for (int j = 0; j < col; j++) {
            temp_row_used.insert(current_matrix[row][j]);
        }
        
        if (!canCompleteRow(row, col, target_sum - current_row_sum, temp_row_used)) {
            return false;
        }
        
        // Try different values for current position
        vector<long> candidates;
        for (long val = 1; val <= P; val++) {
            if (used_elements.find(val) == used_elements.end()) {
                candidates.push_back(val);
            }
        }
        
        // Randomize order to get different matrices
        shuffle(candidates.begin(), candidates.end(), gen);
        
        for (long val : candidates) {
            // Check if this value can lead to a valid completion
            long new_row_sum = current_row_sum + val;
            if (new_row_sum > target_sum) {
                continue;
            }
            
            current_matrix[row][col] = val;
            used_elements.insert(val);
            
            if (generateMatrix(row, col + 1, new_row_sum, used_rows)) {
                return true;
            }
            
            // Backtrack
            used_elements.erase(val);
        }
        
        return false;
    }
    
    // Public interface to generate a matrix
    bool generate(Mat<ZZ>& result) {
        // Reset state
        used_elements.clear();
        for (int i = 0; i < n; i++) {
            fill(current_matrix[i].begin(), current_matrix[i].end(), 0);
        }
        
        set<vector<long>> used_rows;
        
        if (generateMatrix(0, 0, 0, used_rows)) {
            // Copy to NTL matrix
            result.SetDims(n, n);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    result[i][j] = current_matrix[i][j];
                }
            }
            return true;
        }
        
        return false;
    }
};

// Verify matrix constraints
bool verifyMatrix(const Mat<ZZ>& matrix, int n, long P, long target_sum) {
    set<long> all_elements;
    
    // Check all constraints
    for (int i = 0; i < n; i++) {
        long row_sum = 0;
        for (int j = 0; j < n; j++) {
            long val = conv<long>(matrix[i][j]);
            
            // Check range
            if (val < 1 || val > P) {
                return false;
            }
            
            // Check uniqueness
            if (all_elements.find(val) != all_elements.end()) {
                return false; // Duplicate element found
            }
            all_elements.insert(val);
            
            row_sum += val;
        }
        
        // Check row sum
        if (row_sum != target_sum) {
            return false;
        }
    }
    
    // Check if all rows are unique
    set<vector<long>> row_set;
    for (int i = 0; i < n; i++) {
        vector<long> row;
        for (int j = 0; j < n; j++) {
            row.push_back(conv<long>(matrix[i][j]));
        }
        if (row_set.find(row) != row_set.end()) {
            return false;
        }
        row_set.insert(row);
    }
    
    return all_elements.size() == n * n; // All elements must be unique
}

int main() {
    long P;
    cout << "Enter upper bound P for matrix entries (1 to P): ";
    cin >> P;
    
    if (P <= 0) {
        cout << "P must be positive!" << endl;
        return 1;
    }

    const int n = 4;
    const int NUM_MATRICES = 1000;
    const long target_sum = P - 1;

    cout << "Matrix constraints:" << endl;
    cout << "- Entries: 1 to " << P << endl;
    cout << "- Each row sum: " << target_sum << endl;
    cout << "- ALL elements in the entire matrix must be unique" << endl;
    cout << "- All rows must be unique within each matrix" << endl;
    cout << "- All matrices must be unique" << endl << endl;

    // Check basic feasibility
    if (P < 16) {
        cout << "Error: P must be at least 16 to have 16 unique elements in a 4x4 matrix!" << endl;
        return 1;
    }
    
    // Calculate theoretical bounds
    long min_total_sum = 16 * 17 / 2;  // 1+2+...+16 = 136
    long max_total_sum = 0;
    for (long i = P; i > P - 16; i--) {
        max_total_sum += i;
    }
    long required_total_sum = 4 * target_sum;
    
    cout << "Required total sum: " << required_total_sum << endl;
    cout << "Theoretical total sum range: [" << min_total_sum << ", " << max_total_sum << "]" << endl;
    
    if (required_total_sum < min_total_sum || required_total_sum > max_total_sum) {
        cout << "Error: Cannot achieve the required constraints!" << endl;
        cout << "Required: " << required_total_sum << ", Range: [" << min_total_sum << ", " << max_total_sum << "]" << endl;
        return 1;
    }

    unordered_set<vector<long>, MatrixHash> seen;
    vector<Mat<ZZ>> matrices;

    random_device rd;
    mt19937 gen(rd());
    
    MatrixGenerator generator(P, n, target_sum, gen);

    auto gen_start = high_resolution_clock::now();

    cout << "Generating " << NUM_MATRICES << " unique constrained matrices..." << endl;
    int attempts = 0;
    int failed_generations = 0;
    int failed_verifications = 0;
    const int MAX_ATTEMPTS = NUM_MATRICES * 50; // Reasonable limit
    
    while ((int)matrices.size() < NUM_MATRICES && attempts < MAX_ATTEMPTS) {
        attempts++;
        
        Mat<ZZ> A;
        if (!generator.generate(A)) {
            failed_generations++;
            continue;
        }

        // Verify constraints
        if (!verifyMatrix(A, n, P, target_sum)) {
            failed_verifications++;
            continue;
        }

        // Convert to flat vector for uniqueness check
        vector<long> flat;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                flat.push_back(conv<long>(A[i][j]));
            }
        }

        if (seen.insert(flat).second) {
            matrices.push_back(A);
            if (matrices.size() % 50 == 0) {
                cout << "Generated " << matrices.size() << " unique matrices (attempts: " 
                     << attempts << ", failures: " << (failed_generations + failed_verifications) << ")..." << endl;
            }
        }
    }

    auto gen_end = high_resolution_clock::now();
    auto gen_duration = duration_cast<milliseconds>(gen_end - gen_start);

    cout << "Generated " << matrices.size() << " unique matrices in " << attempts 
         << " attempts." << endl;
    cout << "Failed generations: " << failed_generations << endl;
    cout << "Failed verifications: " << failed_verifications << endl;

    if (attempts >= MAX_ATTEMPTS && (int)matrices.size() < NUM_MATRICES) {
        cout << "Warning: Reached maximum attempts. Only generated " 
             << matrices.size() << " matrices." << endl;
    }

    long singular_count = 0, nonsingular_count = 0;

    auto det_start = high_resolution_clock::now();

    cout << "Computing determinants..." << endl;
    for (size_t i = 0; i < matrices.size(); i++) {
        ZZ det;
        determinant(det, matrices[i]);
        if (det == 0) {
            singular_count++;
        } else {
            nonsingular_count++;
        }
        
        if ((i + 1) % 100 == 0) {
            cout << "Processed " << (i + 1) << " matrices..." << endl;
        }
    }

    auto det_end = high_resolution_clock::now();
    auto det_duration = duration_cast<milliseconds>(det_end - det_start);

    // Calculate percentages
    double total = nonsingular_count + singular_count;
    double nonsingular_pct = (total > 0) ? (nonsingular_count / total) * 100.0 : 0.0;
    double singular_pct = (total > 0) ? (singular_count / total) * 100.0 : 0.0;

    // Write all matrices to file
    string matrices_filename = "efficient_unique_matrices_P" + to_string(P) + "_sum" + to_string(target_sum) + ".txt";
    ofstream matrices_file(matrices_filename);
    matrices_file << "All " << matrices.size() << " unique constrained 4x4 matrices:\n";
    matrices_file << "Constraints: entries 1 to " << P << ", each row sum = " << target_sum 
                  << ", ALL matrix elements unique, unique rows per matrix\n\n";
    
    for (size_t i = 0; i < matrices.size(); i++) {
        ZZ det;
        determinant(det, matrices[i]);
        matrices_file << "Matrix " << (i + 1) << " (determinant = " << det << "):\n";
        
        // Print matrix with row sums and verify all elements are unique
        set<long> all_elements;
        for (int row = 0; row < n; row++) {
            long row_sum = 0;
            matrices_file << "[";
            for (int col = 0; col < n; col++) {
                long val = conv<long>(matrices[i][row][col]);
                matrices_file << setw(3) << val;
                if (col < n-1) matrices_file << " ";
                row_sum += val;
                all_elements.insert(val);
            }
            matrices_file << "] (sum=" << row_sum << ")\n";
        }
        matrices_file << "All elements unique: " << (all_elements.size() == 16 ? "YES" : "NO") << "\n\n";
    }
    matrices_file.close();

    // Write results summary
    string results_filename = "efficient_unique_results_P" + to_string(P) + "_sum" + to_string(target_sum) + ".txt";
    ofstream results_file(results_filename);
    results_file << "Results Summary for " << matrices.size() << " constrained 4x4 matrices:\n\n";
    results_file << "Constraints:\n";
    results_file << "- Matrix entries: 1 to " << P << "\n";
    results_file << "- Each row sum: " << target_sum << "\n";
    results_file << "- ALL elements in entire matrix must be unique (16 unique values)\n";
    results_file << "- All rows unique within each matrix\n";
    results_file << "- All matrices unique\n\n";
    
    results_file << "Theoretical analysis:\n";
    results_file << "- Required total sum: " << required_total_sum << "\n";
    results_file << "- Minimum possible total sum (1+2+...+16): " << min_total_sum << "\n";
    results_file << "- Maximum possible total sum: " << max_total_sum << "\n";
    results_file << "- Constraint feasibility: " << (required_total_sum >= min_total_sum && 
                    required_total_sum <= max_total_sum ? "FEASIBLE" : "NOT FEASIBLE") << "\n\n";
    
    results_file << "Generation statistics:\n";
    results_file << "- Total attempts: " << attempts << "\n";
    results_file << "- Failed generations: " << failed_generations << "\n";
    results_file << "- Failed verifications: " << failed_verifications << "\n";
    results_file << "- Success rate: " << fixed << setprecision(2) 
                << (attempts > 0 ? (double)matrices.size() / attempts * 100.0 : 0.0) << "%\n\n";
    
    results_file << "Results:\n";
    results_file << "Non-singular (det != 0): " << nonsingular_count 
                << " (" << fixed << setprecision(2) << nonsingular_pct << "%)\n";
    results_file << "Singular (det == 0): " << singular_count 
                << " (" << fixed << setprecision(2) << singular_pct << "%)\n";
    results_file << "Ratio (nonsingular:singular): " << nonsingular_count 
                << ":" << singular_count << "\n\n";
    results_file << "Time taken for generation: " << gen_duration.count() << " ms\n";
    results_file << "Time taken for determinant computations: " << det_duration.count() << " ms\n";
    results_file << "Total time: " << (gen_duration + det_duration).count() << " ms\n\n";
    results_file << "All matrices stored in: " << matrices_filename << "\n";
    
    results_file.close();

    // Console output
    cout << "\nFinal Results:" << endl;
    cout << "Generated matrices: " << matrices.size() << endl;
    cout << "Success rate: " << (attempts > 0 ? (double)matrices.size() / attempts * 100.0 : 0.0) << "%" << endl;
    cout << "Non-singular: " << nonsingular_count << " (" << nonsingular_pct << "%)" << endl;
    cout << "Singular: " << singular_count << " (" << singular_pct << "%)" << endl;
    cout << "Ratio (nonsingular:singular): " << nonsingular_count << ":" << singular_count << endl;
    cout << "Generation time: " << gen_duration.count() << " ms" << endl;
    cout << "\nFiles created:" << endl;
    cout << "- Matrices (progressive): " << matrices_filename << endl;
    cout << "- Generation progress: " << progress_filename << endl;
    cout << "- Results summary: " << results_filename << endl;

    return 0;
}
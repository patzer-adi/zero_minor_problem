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
            return true; // Successfully filled all rows
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
    
    // Calculate theoretical bounds (as explained by your friend's analysis)
    long min_total_sum = 16 * 17 / 2;  // 1+2+...+16 = 136
    long max_total_sum = 0;
    for (long i = P; i > P - 16; i--) {
        max_total_sum += i;  // 16P - 120
    }
    long required_total_sum = 4 * target_sum;
    
    cout << "\n=== MATHEMATICAL FEASIBILITY CHECK ===" << endl;
    cout << "Your friend's analysis: For 16 unique elements with each row sum = (P-1):" << endl;
    cout << "- Required total sum: 4(P-1) = 4(" << P << "-1) = " << required_total_sum << endl;
    cout << "- Minimum possible sum (1+2+...+16): " << min_total_sum << endl;
    cout << "- Maximum possible sum (16P-120): 16*" << P << "-120 = " << max_total_sum << endl;
    cout << "- Critical constraint: " << min_total_sum << " <= " << required_total_sum << " <= " << max_total_sum << endl;
    
    if (required_total_sum < min_total_sum) {
        cout << "\nX IMPOSSIBLE: Required sum (" << required_total_sum 
             << ") < Minimum possible (" << min_total_sum << ")" << endl;
        cout << "For this constraint system, minimum feasible P satisfies: 4(P-1) >= 136" << endl;
        cout << "So P >= 35, and P = 37 is the first feasible value for typical use." << endl;
        return 1;
    }
    
    if (required_total_sum > max_total_sum) {
        cout << "\nX IMPOSSIBLE: Required sum (" << required_total_sum 
             << ") > Maximum possible (" << max_total_sum << ")" << endl;
        return 1;
    }
    
    cout << "✓ FEASIBLE: Solution space exists!" << endl;
    if (required_total_sum - min_total_sum < 20) {
        cout << "! WARNING: Very tight constraints (only " << (required_total_sum - min_total_sum) 
             << " units above minimum). Expect slow generation." << endl;
    }
    cout << "========================================\n" << endl;

    unordered_set<vector<long>, MatrixHash> seen;
    vector<Mat<ZZ>> matrices;

    random_device rd;
    mt19937 gen(rd());
    
    MatrixGenerator generator(P, n, target_sum, gen);

    auto gen_start = high_resolution_clock::now();

    // Open files for progressive writing
    string matrices_filename = "efficient_unique_matrices_P" + to_string(P) + "_sum" + to_string(target_sum) + ".txt";
    string progress_filename = "generation_progress_P" + to_string(P) + "_sum" + to_string(target_sum) + ".txt";
    
    ofstream matrices_file(matrices_filename);
    ofstream progress_file(progress_filename);
    
    matrices_file << "Unique constrained 4x4 matrices (P=" << P << ", row sum=" << target_sum << "):\n";
    matrices_file << "Constraints: entries 1 to " << P << ", each row sum = " << target_sum 
                  << ", ALL matrix elements unique, unique rows per matrix\n";
    matrices_file << "Generated progressively - check file for real-time updates!\n\n";
    matrices_file.flush();
    
    progress_file << "Generation Progress Log (P=" << P << ", target sum=" << target_sum << ")\n";
    progress_file << "=================================================================\n\n";
    progress_file.flush();

    cout << "Generating " << NUM_MATRICES << " unique constrained matrices..." << endl;
    cout << "Progress will be saved to: " << matrices_filename << endl;
    cout << "Detailed log saved to: " << progress_filename << endl << endl;
    
    int attempts = 0;
    int failed_generations = 0;
    int failed_verifications = 0;
    const int MAX_ATTEMPTS = NUM_MATRICES * 50; // Reasonable limit
    
    while ((int)matrices.size() < NUM_MATRICES && attempts < MAX_ATTEMPTS) {
        attempts++;
        
        Mat<ZZ> A;
        if (!generator.generate(A)) {
            failed_generations++;
            if (attempts % 1000 == 0) {
                progress_file << "After " << attempts << " attempts: " << matrices.size() 
                            << " matrices, " << failed_generations << " generation failures, " 
                            << failed_verifications << " verification failures\n";
                progress_file.flush();
            }
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
            
            // Immediately save this matrix to file
            ZZ det;
            determinant(det, A);
            matrices_file << "Matrix " << matrices.size() << " (determinant = " << det 
                         << ", generated at attempt " << attempts << "):\n";
            
            set<long> all_elements;
            for (int row = 0; row < n; row++) {
                long row_sum = 0;
                matrices_file << "[";
                for (int col = 0; col < n; col++) {
                    long val = conv<long>(A[row][col]);
                    matrices_file << setw(3) << val;
                    if (col < n-1) matrices_file << " ";
                    row_sum += val;
                    all_elements.insert(val);
                }
                matrices_file << "] (sum=" << row_sum << ")\n";
            }
            matrices_file << "All elements unique: " << (all_elements.size() == 16 ? "YES" : "NO");
            matrices_file << " | Singular: " << (det == 0 ? "YES" : "NO") << "\n\n";
            matrices_file.flush(); // Force write to disk
            
            // Update progress
            if (matrices.size() % 10 == 0) {
                cout << "Generated " << matrices.size() << " unique matrices (attempts: " 
                     << attempts << ", gen_failures: " << failed_generations 
                     << ", ver_failures: " << failed_verifications << ")..." << endl;
                
                progress_file << "MILESTONE: Generated " << matrices.size() << " matrices!\n";
                progress_file << "- Total attempts so far: " << attempts << "\n";
                progress_file << "- Generation failures: " << failed_generations << "\n";
                progress_file << "- Verification failures: " << failed_verifications << "\n";
                progress_file << "- Success rate: " << fixed << setprecision(2) 
                            << (double)matrices.size() / attempts * 100.0 << "%\n";
                progress_file << "- Time elapsed: " << duration_cast<seconds>(
                    high_resolution_clock::now() - gen_start).count() << " seconds\n\n";
                progress_file.flush();
            }
        }
    }
    
    matrices_file.close();
    progress_file.close();

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

    // Count determinants from already computed values (done during generation)
    long singular_count = 0, nonsingular_count = 0;
    
    cout << "Counting matrix types from generated data..." << endl;
    for (size_t i = 0; i < matrices.size(); i++) {
        ZZ det;
        determinant(det, matrices[i]);
        if (det == 0) {
            singular_count++;
        } else {
            nonsingular_count++;
        }
    }

    // Calculate percentages
    double total = nonsingular_count + singular_count;
    double nonsingular_pct = (total > 0) ? (nonsingular_count / total) * 100.0 : 0.0;
    double singular_pct = (total > 0) ? (singular_count / total) * 100.0 : 0.0;

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
    results_file << "Total time: " << gen_duration.count() << " ms\n\n";
    results_file << "Matrices saved progressively in: " << matrices_filename << "\n";
    results_file << "Generation progress logged in: " << progress_filename << "\n";
    
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
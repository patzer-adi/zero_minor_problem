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
#include <functional>
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

// Check if ALL 16 entries in the matrix are unique
bool hasAllUniqueElements(const Mat<ZZ>& matrix, int n) {
    std::set<long> all_elements;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            long val = conv<long>(matrix[i][j]);
            if (all_elements.find(val) != all_elements.end()) return false;
            all_elements.insert(val);
        }
    }
    return true;
}

// Check if all rows (as ordered 4-tuples) are unique within the matrix
bool hasUniqueRows(const Mat<ZZ>& matrix, int n) {
    std::set<vector<long>> row_set;
    for (int i = 0; i < n; i++) {
        vector<long> row;
        for (int j = 0; j < n; j++) row.push_back(conv<long>(matrix[i][j]));
        if (row_set.find(row) != row_set.end()) return false;
        row_set.insert(row);
    }
    return true;
}

// Verify matrix constraints (entries in [1..P], each row sums to target_sum, 16 unique entries, unique rows)
bool verifyMatrix(const Mat<ZZ>& matrix, int n, long P, long target_sum) {
    for (int i = 0; i < n; i++) {
        long row_sum = 0;
        for (int j = 0; j < n; j++) {
            long v = conv<long>(matrix[i][j]);
            if (v < 1 || v > P) return false;
            row_sum += v;
        }
        if (row_sum != target_sum) return false;
    }
    if (!hasAllUniqueElements(matrix, n)) return false;
    return hasUniqueRows(matrix, n);
}

// Generate all 4-combinations (a<b<c<d) of numbers in [1..P] that sum to target_sum
// Then expand each combination into all 4! permutations (ordered rows),
// so row orderings are preserved just like your original permutation approach.
static void generateAllValidRows(long P, long target_sum,
                                 vector<array<long,4>>& row_perms) {
    for (long a = 1; a <= P - 3; ++a) {
        for (long b = a + 1; b <= P - 2; ++b) {
            for (long c = b + 1; c <= P - 1; ++c) {
                long d = target_sum - (a + b + c);
                if (d > c && d <= P) {
                    array<long,4> comb = {a, b, c, d};
                    // generate all 4! permutations
                    array<long,4> p = comb;
                    sort(p.begin(), p.end());
                    do {
                        row_perms.push_back(p);
                    } while (next_permutation(p.begin(), p.end()));
                }
            }
        }
    }
}

// Backtracking to assemble 4 disjoint rows (16 unique numbers) into full matrices.
// We keep a cap to avoid generating an enormous number.
static void backtrackMatrices(const vector<array<long,4>>& row_perms,
                              int n,
                              long P,
                              long target_sum,
                              size_t start_index,
                              Mat<ZZ>& current,
                              vector<char>& used,                // used[v]==1 if value v is already used in the matrix
                              std::set<vector<long>>& used_rows, // prevent identical ordered rows within one matrix
                              vector<Mat<ZZ>>& out,
                              size_t& generated,
                              const size_t MAX_MATRICES_TO_GENERATE)
{
    int row = 0;
    for (int i = 0; i < n; ++i) {
        if (conv<long>(current[i][0]) == 0 &&
            conv<long>(current[i][1]) == 0 &&
            conv<long>(current[i][2]) == 0 &&
            conv<long>(current[i][3]) == 0) { row = i; break; }
        if (i == n-1) row = n; // all rows filled
    }

    if (row >= n) {
        // Completed a matrix
        if (verifyMatrix(current, n, P, target_sum)) {
            out.push_back(current);
            generated++;
        }
        return;
    }

    // We allow different row orders to produce different matrices (matches original behavior).
    // To get a wide variety quickly, we iterate through all row_perms.
    for (size_t idx = 0; idx < row_perms.size(); ++idx) {
        const auto& r = row_perms[idx];
        // Check row uniqueness inside matrix (ordered)
        vector<long> rvec = {r[0], r[1], r[2], r[3]};
        if (used_rows.count(rvec)) continue;

        // Check disjointness: all 4 values unused so far
        if (used[r[0]] || used[r[1]] || used[r[2]] || used[r[3]]) continue;

        // Place row
        current[row][0] = r[0];
        current[row][1] = r[1];
        current[row][2] = r[2];
        current[row][3] = r[3];

        used[r[0]] = used[r[1]] = used[r[2]] = used[r[3]] = 1;
        used_rows.insert(rvec);

        backtrackMatrices(row_perms, n, P, target_sum, idx + 1, current, used, used_rows,
                          out, generated, MAX_MATRICES_TO_GENERATE);

        // Undo
        used_rows.erase(rvec);
        used[r[0]] = used[r[1]] = used[r[2]] = used[r[3]] = 0;
        current[row][0] = 0; current[row][1] = 0; current[row][2] = 0; current[row][3] = 0;

        if (generated >= MAX_MATRICES_TO_GENERATE) return;
    }
}

// Optimized generator: build all valid rows first, then assemble matrices by picking 4 disjoint rows.
vector<Mat<ZZ>> generateAllValidMatrices(long P, int n, long target_sum) {
    vector<Mat<ZZ>> valid_matrices;

    const long total_elements = n * n; // 16
    const long min_total_sum = total_elements * (total_elements + 1) / 2; // 1+...+16 = 136
    long max_total_sum = 0;
    for (long i = P; i > P - total_elements; i--) max_total_sum += i;
    const long required_total_sum = n * target_sum;

    if (P < total_elements) {
        cout << "Error: P must be at least " << total_elements << " to have "
             << total_elements << " unique elements!" << endl;
        return valid_matrices;
    }
    if (required_total_sum < min_total_sum || required_total_sum > max_total_sum) {
        cout << "Error: Cannot achieve total sum " << required_total_sum
             << " with " << total_elements << " unique elements from 1 to " << P << endl;
        cout << "Possible total sum range: [" << min_total_sum << ", " << max_total_sum << "]" << endl;
        return valid_matrices;
    }

    cout << "Generating all valid " << n << "x" << n << " matrices (row-based)..." << endl;
    cout << "Required total sum: " << required_total_sum << endl;
    cout << "Possible total sum range: [" << min_total_sum << ", " << max_total_sum << "]" << endl;

    // Step 1: Generate all valid rows (ordered 4-tuples summing to target_sum)
    vector<array<long,4>> row_perms;
    generateAllValidRows(P, target_sum, row_perms);
    cout << "Valid row permutations found: " << row_perms.size() << endl;

    if (row_perms.empty()) {
        cout << "No valid rows exist with sum " << target_sum << "!" << endl;
        return valid_matrices;
    }

    // Step 2: Assemble matrices via backtracking (choose 4 disjoint rows)
    const size_t MAX_MATRICES_TO_GENERATE = 100000; // keep your original limit
    size_t generated = 0;

    Mat<ZZ> current;
    current.SetDims(n, n);
    // zero initialize
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            current[i][j] = 0;

    vector<char> used(P + 1, 0);           // track used numbers globally (1..P)
    std::set<vector<long>> used_rows;      // ensure no duplicate ordered rows within the same matrix

    backtrackMatrices(row_perms, n, P, target_sum, 0, current, used, used_rows,
                      valid_matrices, generated, MAX_MATRICES_TO_GENERATE);

    cout << "Row-based construction produced " << valid_matrices.size() << " matrices." << endl;
    return valid_matrices;
}

// Random pick from pre-generated list (unchanged)
Mat<ZZ> generateConstrainedMatrix(mt19937& gen, const vector<Mat<ZZ>>& valid_matrices) {
    if (valid_matrices.empty()) {
        Mat<ZZ> result;
        result.SetDims(4, 4);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                result[i][j] = (i == j) ? 1 : 0;
        return result;
    }
    int idx = gen() % valid_matrices.size();
    return valid_matrices[idx];
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
    const int NUM_MATRICES = 100;       // your current setting
    const long target_sum = P - 1;

    cout << "Matrix constraints:" << endl;
    cout << "- Entries: 1 to " << P << endl;
    cout << "- Each row sum: " << target_sum << endl;
    cout << "- ALL elements in the entire matrix must be unique" << endl;
    cout << "- All rows must be unique within each matrix" << endl;
    cout << "- All matrices must be unique" << endl << endl;

    if (P < 16) {
        cout << "Error: P must be at least 16 to have 16 unique elements in a 4x4 matrix!" << endl;
        return 1;
    }

    long min_total_sum = 16 * 17 / 2; // 136
    long max_total_sum = 0;
    for (long i = P; i > P - 16; i--) max_total_sum += i;
    long required_total_sum = 4 * target_sum;

    cout << "Required total sum: " << required_total_sum << endl;
    cout << "Theoretical total sum range: [" << min_total_sum << ", " << max_total_sum << "]" << endl;

    if (required_total_sum < min_total_sum || required_total_sum > max_total_sum) {
        cout << "Error: Cannot achieve the required constraints!" << endl;
        return 1;
    }

    cout << "Generating all valid matrices (this is the optimized row-based method)..." << endl;
    vector<Mat<ZZ>> valid_matrices = generateAllValidMatrices(P, n, target_sum);

    if (valid_matrices.empty()) {
        cout << "No valid matrices can be generated with the given constraints!" << endl;
        return 1;
    }

    cout << "Found " << valid_matrices.size() << " total valid matrices." << endl;

    int matrices_to_analyze = min(NUM_MATRICES, (int)valid_matrices.size());
    cout << "Analyzing " << matrices_to_analyze << " matrices..." << endl;

    unordered_set<vector<long>, MatrixHash> seen;
    vector<Mat<ZZ>> selected_matrices;

    random_device rd;
    mt19937 gen(rd());

    auto gen_start = high_resolution_clock::now();

    while ((int)selected_matrices.size() < matrices_to_analyze) {
        Mat<ZZ> A = generateConstrainedMatrix(gen, valid_matrices);

        if (!verifyMatrix(A, n, P, target_sum)) {
            cout << "Warning: Pre-generated matrix failed verification!" << endl;
            continue;
        }

        vector<long> flat;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                flat.push_back(conv<long>(A[i][j]));

        if (seen.insert(flat).second) {
            selected_matrices.push_back(A);
            if (selected_matrices.size() % 100 == 0) {
                cout << "Selected " << selected_matrices.size() << " unique matrices..." << endl;
            }
        }
    }

    auto gen_end = high_resolution_clock::now();
    auto gen_duration = duration_cast<milliseconds>(gen_end - gen_start);

    cout << "Selected " << selected_matrices.size() << " unique matrices." << endl;

    long singular_count = 0, nonsingular_count = 0;

    auto det_start = high_resolution_clock::now();

    cout << "Computing determinants..." << endl;
    for (size_t i = 0; i < selected_matrices.size(); i++) {
        ZZ det;
        determinant(det, selected_matrices[i]);
        if (det == 0) singular_count++;
        else nonsingular_count++;

        if ((i + 1) % 200 == 0) {
            cout << "Processed " << (i + 1) << " matrices..." << endl;
        }
    }

    auto det_end = high_resolution_clock::now();
    auto det_duration = duration_cast<milliseconds>(det_end - det_start);

    double total = nonsingular_count + singular_count;
    double nonsingular_pct = (total > 0) ? (nonsingular_count / total) * 100.0 : 0.0;
    double singular_pct = (total > 0) ? (singular_count / total) * 100.0 : 0.0;

    string matrices_filename = "all_unique_matrices_P" + to_string(P) + "_sum" + to_string(target_sum) + ".txt";
    ofstream matrices_file(matrices_filename);
    matrices_file << "All " << selected_matrices.size() << " unique constrained 4x4 matrices:\n";
    matrices_file << "Constraints: entries 1 to " << P << ", each row sum = " << target_sum
                  << ", ALL matrix elements unique, unique rows per matrix\n\n";

    for (size_t i = 0; i < selected_matrices.size(); i++) {
        ZZ det;
        determinant(det, selected_matrices[i]);
        matrices_file << "Matrix " << (i + 1) << " (determinant = " << det << "):\n";

        std::set<long> all_elements;
        for (int row = 0; row < n; row++) {
            long row_sum = 0;
            matrices_file << "[";
            for (int col = 0; col < n; col++) {
                long val = conv<long>(selected_matrices[i][row][col]);
                matrices_file << setw(3) << val;
                if (col < n - 1) matrices_file << " ";
                row_sum += val;
                all_elements.insert(val);
            }
            matrices_file << "] (sum=" << row_sum << ")\n";
        }
        matrices_file << "All elements unique: " << (all_elements.size() == 16 ? "YES" : "NO") << "\n\n";
    }
    matrices_file.close();

    string results_filename = "all_unique_results_summary_P" + to_string(P) + "_sum" + to_string(target_sum) + ".txt";
    ofstream results_file(results_filename);
    results_file << "Results Summary for " << selected_matrices.size() << " constrained 4x4 matrices:\n\n";
    results_file << "Constraints:\n";
    results_file << "- Matrix entries: 1 to " << P << "\n";
    results_file << "- Each row sum: " << target_sum << "\n";
    results_file << "- ALL elements in entire matrix must be unique (16 unique values)\n";
    results_file << "- All rows unique within each matrix\n";
    results_file << "- All matrices unique\n\n";

    long min_total_sum_out = 16 * 17 / 2;
    long max_total_sum_out = 0; for (long i = P; i > P - 16; i--) max_total_sum_out += i;
    long required_total_sum_out = 4 * target_sum;

    results_file << "Theoretical analysis:\n";
    results_file << "- Required total sum: " << required_total_sum_out << "\n";
    results_file << "- Minimum possible total sum (1+2+...+16): " << min_total_sum_out << "\n";
    results_file << "- Maximum possible total sum: " << max_total_sum_out << "\n";
    results_file << "- Constraint feasibility: "
                 << (required_total_sum_out >= min_total_sum_out && required_total_sum_out <= max_total_sum_out ? "FEASIBLE" : "NOT FEASIBLE") << "\n\n";

    results_file << "Total valid matrices found: " << valid_matrices.size() << "\n";
    results_file << "Matrices analyzed: " << selected_matrices.size() << "\n\n";
    results_file << "Results:\n";
    results_file << "Non-singular (det != 0): " << nonsingular_count
                 << " (" << fixed << setprecision(2) << nonsingular_pct << "%)\n";
    results_file << "Singular (det == 0): " << singular_count
                 << " (" << fixed << setprecision(2) << singular_pct << "%)\n";
    results_file << "Ratio (nonsingular:singular): " << nonsingular_count << ":" << singular_count << "\n\n";
    results_file << "Time taken for selection: " << gen_duration.count() << " ms\n";
    results_file << "Time taken for determinant computations: " << det_duration.count() << " ms\n";
    results_file << "Total time: " << (gen_duration + det_duration).count() << " ms\n\n";
    results_file << "All matrices stored in: " << matrices_filename << "\n";
    results_file.close();

    cout << "\nFinal Results:" << endl;
    cout << "Total valid matrices found: " << valid_matrices.size() << endl;
    cout << "Matrices analyzed: " << selected_matrices.size() << endl;
    cout << "Non-singular: " << nonsingular_count << " (" << nonsingular_pct << "%)" << endl;
    cout << "Singular: " << singular_count << " (" << singular_pct << "%)" << endl;
    cout << "Ratio (nonsingular:singular): " << nonsingular_count << ":" << singular_count << endl;
    cout << "\nFiles created:" << endl;
    cout << "- All matrices: " << matrices_filename << endl;
    cout << "- Results summary: " << results_filename << endl;

    return 0;
}


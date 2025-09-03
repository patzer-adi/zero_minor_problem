#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <chrono>
#include <fstream>

using namespace std;

// Function to check if a number is prime
bool isPrime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

// Generate list of primes up to limit
vector<int> generatePrimes(int limit) {
    vector<int> primes;
    for (int i = 2; i <= limit; i++) {
        if (isPrime(i)) {
            primes.push_back(i);
        }
    }
    return primes;
}

// Find all prime pairs that sum to target
vector<pair<int,int>> findPrimePairs(const vector<int>& primes, int target) {
    vector<pair<int,int>> pairs;
    set<int> prime_set(primes.begin(), primes.end());
    
    for (int p : primes) {
        int complement = target - p;
        if (complement > p && prime_set.count(complement)) {
            pairs.push_back({p, complement});
        }
    }
    return pairs;
}

// Solve 2x2 case directly
bool solve2x2(const vector<pair<int,int>>& pairs, vector<vector<int>>& matrix) {
    // Try all combinations of 2 pairs
    for (int i = 0; i < pairs.size(); i++) {
        for (int j = i + 1; j < pairs.size(); j++) {
            auto [a, b] = pairs[i];
            auto [c, d] = pairs[j];
            
            // Check if all 4 values are unique
            set<int> values = {a, b, c, d};
            if (values.size() == 4) {
                // Found valid solution
                matrix[0][0] = a; matrix[0][1] = b;
                matrix[1][0] = c; matrix[1][1] = d;
                return true;
            }
        }
    }
    return false;
}

// Backtracking solver for n≥3
bool solveGeneral(vector<vector<int>>& matrix, const vector<int>& primes, 
                 set<int>& used, int n, int target_sum, int pos) {
    
    if (pos == n * n) {
        return true; // All positions filled and constraints checked during construction
    }
    
    int row = pos / n;
    int col = pos % n;
    
    for (int prime : primes) {
        if (used.count(prime)) continue;
        
        matrix[row][col] = prime;
        used.insert(prime);
        
        // Check constraints
        bool valid = true;
        
        // If we're at the end of a row, check row sum
        if (col == n - 1) {
            int row_sum = 0;
            for (int j = 0; j < n; j++) {
                row_sum += matrix[row][j];
            }
            if (row_sum != target_sum) {
                valid = false;
            }
        }
        
        // If current partial row sum exceeds target, invalid
        if (valid) {
            int partial_sum = 0;
            for (int j = 0; j <= col; j++) {
                partial_sum += matrix[row][j];
            }
            if (partial_sum > target_sum) {
                valid = false;
            }
        }
        
        if (valid && solveGeneral(matrix, primes, used, n, target_sum, pos + 1)) {
            return true;
        }
        
        matrix[row][col] = 0;
        used.erase(prime);
    }
    
    return false;
}

// Generate matrix with given constraints
bool generateMatrix(int n, int p, vector<vector<int>>& result) {
    if (!isPrime(p)) {
        cout << "Error: " << p << " is not prime!" << endl;
        return false;
    }
    
    int target_sum = p - 1;
    vector<int> primes = generatePrimes(target_sum);
    
    cout << "Using primes up to " << target_sum << ": " << primes.size() << " available" << endl;
    
    if (primes.size() < n * n) {
        cout << "Error: Not enough primes ≤ " << target_sum << " for " << n << "×" << n << " matrix" << endl;
        cout << "Need " << n*n << " primes, but only have " << primes.size() << endl;
        return false;
    }
    
    result.assign(n, vector<int>(n, 0));
    
    auto start_time = chrono::high_resolution_clock::now();
    bool found = false;
    
    if (n == 2) {
        // Special handling for 2x2
        vector<pair<int,int>> pairs = findPrimePairs(primes, target_sum);
        cout << "Valid prime pairs that sum to " << target_sum << ":" << endl;
        for (auto [a, b] : pairs) {
            cout << "  " << a << " + " << b << " = " << target_sum << endl;
        }
        cout << "Total valid pairs: " << pairs.size() << endl;
        
        if (pairs.size() >= 2) {
            found = solve2x2(pairs, result);
        }
    } else {
        // General case for n≥3
        set<int> used;
        found = solveGeneral(result, primes, used, n, target_sum, 0);
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    
    cout << "Search completed in " << duration.count() << " ms" << endl;
    
    return found;
}

// Print matrix
void printMatrix(const vector<vector<int>>& matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << matrix[i][j] << "\t";
        }
        cout << endl;
    }
}

// Verify matrix constraints
void verifyMatrix(const vector<vector<int>>& matrix, int n, int p) {
    cout << "\n=== VERIFICATION ===" << endl;
    
    // Check all elements are prime and unique
    set<int> used;
    bool all_prime = true;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (!isPrime(matrix[i][j])) {
                cout << "Element [" << i << "][" << j << "] = " << matrix[i][j] << " is not prime!" << endl;
                all_prime = false;
            }
            if (used.count(matrix[i][j])) {
                cout << "Element " << matrix[i][j] << " appears multiple times!" << endl;
            }
            used.insert(matrix[i][j]);
        }
    }
    if (all_prime) cout << "✓ All elements are prime" << endl;
    if (used.size() == n*n) cout << "✓ All elements are unique" << endl;
    
    // Check row sums
    bool rows_valid = true;
    for (int i = 0; i < n; i++) {
        int sum = 0;
        for (int j = 0; j < n; j++) {
            sum += matrix[i][j];
        }
        cout << "Row " << i << " sum: " << sum;
        if (sum == p - 1) {
            cout << " ✓" << endl;
        } else {
            cout << " ✗ (should be " << p-1 << ")" << endl;
            rows_valid = false;
        }
    }
    
    // Check column sums (informational only)
    cout << "\nColumn sums (informational):" << endl;
    for (int j = 0; j < n; j++) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += matrix[i][j];
        }
        cout << "Col " << j << " sum: " << sum << endl;
    }
    
    if (all_prime && used.size() == n*n && rows_valid) {
        cout << "\n🎉 Matrix satisfies ALL constraints!" << endl;
    } else {
        cout << "\n❌ Matrix does NOT satisfy all constraints!" << endl;
    }
}

int main() {
    int numPrimes;
    cout << "Enter number of primes: ";
    cin >> numPrimes;

    vector<int> primes(numPrimes);
    for (int i = 0; i < numPrimes; i++) {
        cout << "Enter prime #" << (i+1) << ": ";
        cin >> primes[i];
        if (!isPrime(primes[i])) {
            cout << primes[i] << " is not prime! Please enter a valid prime." << endl;
            i--; // retry
        }
    }

    int numSizes;
    cout << "Enter number of matrix sizes: ";
    cin >> numSizes;

    vector<int> sizes(numSizes);
    for (int i = 0; i < numSizes; i++) {
        cout << "Enter matrix size n #" << (i+1) << ": ";
        cin >> sizes[i];
        if (sizes[i] < 2) {
            cout << "Matrix size must be at least 2. Retry." << endl;
            i--; // retry
        }
    }

    // Loop through primes and sizes
    for (int p : primes) {
        for (int n : sizes) {
            cout << "\nGenerating " << n << "×" << n 
                 << " matrix for prime p=" << p << " (target sum=" << p-1 << ")..." << endl;

            vector<vector<int>> matrix;
            if (generateMatrix(n, p, matrix)) {
                cout << "Solution found!" << endl;
                printMatrix(matrix, n);

                // Generate dynamic filename
                std::string filename = "prime_matrix_solution_" 
                                       + std::to_string(p) + "_" 
                                       + std::to_string(n) + "x" + std::to_string(n) + ".txt";

                std::ofstream outfile(filename);
                if (!outfile.is_open()) {
                    std::cerr << "Error opening file " << filename << std::endl;
                    continue;
                }

                outfile << "Prime Matrix Solution (p=" << p << ", n=" << n << ")\n";
                outfile << "Target sum for each row: " << p-1 << "\n\n";
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        outfile << matrix[i][j] << "\t";
                    }
                    outfile << "\n";
                }
                outfile.close();
                cout << "Matrix saved to '" << filename << "'\n";

                // Optional: verify matrix
                verifyMatrix(matrix, n, p);
            } else {
                cout << "❌ No solution found for p=" << p << ", n=" << n << endl;
            }
        }
    }

    return 0;
}

#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <random>
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

// Check if current matrix state is valid (partial solution)
bool isValidState(const vector<vector<int>>& matrix, int n, int target_sum, int row, int col) {
    // Check row sum doesn't exceed target
    for (int r = 0; r <= row; r++) {
        int sum = 0;
        bool complete_row = true;
        for (int c = 0; c < n; c++) {
            if (matrix[r][c] == 0) {
                if (r == row && c > col) break;
                complete_row = false;
                break;
            }
            sum += matrix[r][c];
        }
        if (complete_row && sum != target_sum) return false;
        if (sum > target_sum) return false;
    }
    
    // Check column sum doesn't exceed target
    for (int c = 0; c <= col; c++) {
        int sum = 0;
        bool complete_col = true;
        for (int r = 0; r < n; r++) {
            if (matrix[r][c] == 0) {
                if (r > row) {
                    complete_col = false;
                    break;
                }
            }
            sum += matrix[r][c];
        }
        if (complete_col && sum != target_sum) return false;
        if (sum > target_sum) return false;
    }
    
    return true;
}

// Check if the complete matrix satisfies all constraints
bool isValidComplete(const vector<vector<int>>& matrix, int n, int target_sum) {
    // Check all elements are unique
    set<int> used;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (used.count(matrix[i][j])) return false;
            used.insert(matrix[i][j]);
        }
    }
    
    // Check row sums
    for (int i = 0; i < n; i++) {
        int sum = 0;
        for (int j = 0; j < n; j++) {
            sum += matrix[i][j];
        }
        if (sum != target_sum) return false;
    }
    
    // Check column sums
    for (int j = 0; j < n; j++) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += matrix[i][j];
        }
        if (sum != target_sum) return false;
    }
    
    return true;
}

// Backtracking solver
bool solve(vector<vector<int>>& matrix, const vector<int>& primes, 
          set<int>& used, int n, int target_sum, int pos) {
    
    if (pos == n * n) {
        return isValidComplete(matrix, n, target_sum);
    }
    
    int row = pos / n;
    int col = pos % n;
    
    // Try each available prime
    vector<int> shuffled_primes = primes;
    random_device rd;
    mt19937 g(rd());
    shuffle(shuffled_primes.begin(), shuffled_primes.end(), g);
    
    for (int prime : shuffled_primes) {
        if (used.count(prime)) continue;
        
        matrix[row][col] = prime;
        used.insert(prime);
        
        if (isValidState(matrix, n, target_sum, row, col)) {
            if (solve(matrix, primes, used, n, target_sum, pos + 1)) {
                return true;
            }
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
    
    // Check if solution is theoretically possible
    if (target_sum * n != target_sum * n) {
        // This is always true, but we need to check if we have enough small primes
        vector<int> primes = generatePrimes(target_sum);
        if (primes.size() < n * n) {
            cout << "Error: Not enough primes ≤ " << target_sum << " for " << n << "×" << n << " matrix" << endl;
            cout << "Need " << n*n << " primes, but only have " << primes.size() << endl;
            return false;
        }
    }
    
    vector<int> primes = generatePrimes(target_sum);
    cout << "Using primes up to " << target_sum << ": " << primes.size() << " available" << endl;
    
    result.assign(n, vector<int>(n, 0));
    set<int> used;
    
    auto start_time = chrono::high_resolution_clock::now();
    
    bool found = solve(result, primes, used, n, target_sum, 0);
    
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
    
    // Check column sums
    bool cols_valid = true;
    for (int j = 0; j < n; j++) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += matrix[i][j];
        }
        cout << "Col " << j << " sum: " << sum;
        if (sum == p - 1) {
            cout << " ✓" << endl;
        } else {
            cout << " ✗ (should be " << p-1 << ")" << endl;
            cols_valid = false;
        }
    }
    
    if (all_prime && used.size() == n*n && rows_valid && cols_valid) {
        cout << "\n🎉 Matrix satisfies ALL constraints!" << endl;
    } else {
        cout << "\n❌ Matrix does NOT satisfy all constraints!" << endl;
    }
}

int main() {
    int n, p;
    
    cout << "Enter matrix size (n for n×n): ";
    cin >> n;
    cout << "Enter prime number p: ";
    cin >> p;
    
    cout << "\nGenerating " << n << "×" << n << " matrix where:" << endl;
    cout << "- All elements are prime numbers" << endl;
    cout << "- All elements are unique" << endl;
    cout << "- Each row sum = " << p-1 << endl;
    cout << "- Each column sum = " << p-1 << endl;
    cout << "\nSearching..." << endl;
    
    vector<vector<int>> matrix;
    
    if (generateMatrix(n, p, matrix)) {
        cout << "\n🎉 Solution found!" << endl;
        cout << "\nGenerated Matrix:" << endl;
        printMatrix(matrix, n);
        
        // Save to file
        ofstream outfile("prime_matrix_solution.txt");
        outfile << "Prime Matrix Solution (p=" << p << ", n=" << n << ")\n";
        outfile << "Target sum for each row/column: " << p-1 << "\n\n";
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                outfile << matrix[i][j] << "\t";
            }
            outfile << "\n";
        }
        outfile.close();
        cout << "Matrix saved to 'prime_matrix_solution.txt'" << endl;
        
        verifyMatrix(matrix, n, p);
    } else {
        cout << "\n❌ No solution found!" << endl;
        cout << "This might happen because:" << endl;
        cout << "- Not enough primes ≤ " << p-1 << " for unique " << n << "×" << n << " matrix" << endl;
        cout << "- Mathematical constraints cannot be satisfied" << endl;
        cout << "Try with:" << endl;
        cout << "- Smaller matrix size (n)" << endl;
        cout << "- Larger prime number (p)" << endl;
    }
    
    return 0;
}
#include <iostream>
#include <vector>
#include <set>
#include <fstream>
#include <NTL/ZZ.h>

using namespace std;
using namespace NTL;

// Function to check if a number is prime using NTL
bool isPrimeZZ(const ZZ& n) {
    return ProbPrime(n, 10); // 10 rounds of Miller-Rabin primality test
}

// Generate list of primes up to limit
vector<ZZ> generatePrimesZZ(const ZZ& limit) {
    vector<ZZ> primes;
    for (ZZ i = ZZ(2); i <= limit; i++) {
        if (isPrimeZZ(i)) {
            primes.push_back(i);
        }
    }
    return primes;
}

// Solve 2x2 matrix directly using prime pairs
bool solve2x2ZZ(const vector<pair<ZZ,ZZ>>& pairs, vector<vector<ZZ>>& matrix) {
    for (int i = 0; i < pairs.size(); i++) {
        for (int j = i+1; j < pairs.size(); j++) {
            auto [a, b] = pairs[i];
            auto [c, d] = pairs[j];
            set<ZZ> s = {a, b, c, d};
            if (s.size() == 4) {
                matrix[0][0]=a; matrix[0][1]=b;
                matrix[1][0]=c; matrix[1][1]=d;
                return true;
            }
        }
    }
    return false;
}

// Find all prime pairs that sum to target
vector<pair<ZZ,ZZ>> findPrimePairsZZ(const vector<ZZ>& primes, const ZZ& target) {
    vector<pair<ZZ,ZZ>> pairs;
    set<ZZ> s(primes.begin(), primes.end());
    for (const ZZ& p : primes) {
        ZZ complement = target - p;
        if (complement > p && s.count(complement)) {
            pairs.push_back({p, complement});
        }
    }
    return pairs;
}

// Generate n x n prime matrix (supports n>=2)
bool generateMatrixZZ(int n, const ZZ& p, vector<vector<ZZ>>& result) {
    if (!isPrimeZZ(p)) {
        cout << "Error: " << p << " is not prime!" << endl;
        return false;
    }

    ZZ target = p - 1;
    vector<ZZ> primes = generatePrimesZZ(target);

    if (primes.size() < n*n) {
        cout << "Not enough primes ≤ " << target << " for " << n << "x" << n << " matrix" << endl;
        return false;
    }

    result.assign(n, vector<ZZ>(n, ZZ(0)));
    bool found = false;

    if (n == 2) {
        auto pairs = findPrimePairsZZ(primes, target);
        if (pairs.size() >= 2) {
            found = solve2x2ZZ(pairs, result);
        }
    } else {
        // Simple backtracking for general n≥3 (small n)
        set<ZZ> used;
        function<bool(int)> backtrack = [&](int pos) {
            if (pos == n*n) return true;
            int row = pos / n;
            int col = pos % n;

            for (const ZZ& prime : primes) {
                if (used.count(prime)) continue;
                result[row][col] = prime;
                used.insert(prime);

                bool valid = true;
                ZZ sum_row = ZZ(0);
                for (int j = 0; j <= col; j++) sum_row += result[row][j];
                if (sum_row > target) valid = false;
                if (col == n-1 && sum_row != target) valid = false;

                if (valid && backtrack(pos+1)) return true;

                result[row][col] = ZZ(0);
                used.erase(prime);
            }
            return false;
        };

        found = backtrack(0);
    }

    return found;
}

// Print matrix
void printMatrixZZ(const vector<vector<ZZ>>& matrix) {
    for (auto& row : matrix) {
        for (auto& val : row) cout << val << "\t";
        cout << endl;
    }
}

int main() {
    int n;
    ZZ p;

    cout << "Enter matrix size (n x n): ";
    cin >> n;
    cout << "Enter prime number p: ";
    cin >> p;

    vector<vector<ZZ>> matrix;
    if (generateMatrixZZ(n, p, matrix)) {
        cout << "Solution found!" << endl;
        printMatrixZZ(matrix);

        string filename = "prime_matrix_solution_" + to_string(conv<long>(p)) + "_" 
                        + to_string(n) + "x" + to_string(n) + ".txt";
        ofstream outfile(filename);
        outfile << "Prime Matrix Solution (p=" << p << ", n=" << n << ")\n";
        outfile << "Target sum per row: " << p-1 << "\n\n";
        for (auto& row : matrix) {
            for (auto& val : row) outfile << val << "\t";
            outfile << "\n";
        }
        outfile.close();
        cout << "Matrix saved to " << filename << endl;
    } else {
        cout << "No solution found!" << endl;
    }

    return 0;
}

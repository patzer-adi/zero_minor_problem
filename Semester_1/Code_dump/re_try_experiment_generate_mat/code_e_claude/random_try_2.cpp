#include <NTL/ZZ_p.h>
#include <NTL/mat_ZZ_p.h>
#include <NTL/vector.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>
#include <chrono>
#include <iomanip>
using namespace std;
using namespace NTL;

class ProgressBar {
private:
    int width;
    string description;
    
public:
    ProgressBar(int w = 50, string desc = "Progress") : width(w), description(desc) {}
    
    void update(double progress, long current, long total) {
        int pos = width * progress;
        cout << "\r" << description << ": [";
        for (int i = 0; i < width; ++i) {
            if (i < pos) cout << "=";
            else if (i == pos) cout << ">";
            else cout << " ";
        }
        cout << "] " << int(progress * 100.0) << "% (" << current << "/" << total << ")";
        cout.flush();
    }
    
    void finish() {
        cout << endl;
    }
};

// Generate a random matrix with unique elements
Mat<ZZ_p> generateUniqueMatrix(long rows, long cols, long p) {
    if (rows * cols > p) {
        throw runtime_error("Matrix too large for unique elements modulo p");
    }
    
    // Create pool of numbers 1..p (changed from 0..p-1 to 1..p)
    vector<long> pool(p);
    iota(pool.begin(), pool.end(), 1);
    
    // Shuffle
    random_device rd;
    mt19937 g(rd());
    shuffle(pool.begin(), pool.end(), g);
    
    // Pick first rows*cols numbers
    pool.resize(rows * cols);
    
    // Fill matrix
    Mat<ZZ_p> M;
    M.SetDims(rows, cols);
    long idx = 0;
    for (long i = 0; i < rows; i++) {
        for (long j = 0; j < cols; j++) {
            M[i][j] = ZZ_p(pool[idx++]);
        }
    }
    return M;
}

// Check if prime number
bool isPrime(long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    
    for (long i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) {
            return false;
        }
    }
    return true;
}

// Validate inputs
bool validateInputs(long p, long rows, long cols, long n) {
    if (p <= 1) {
        cerr << "Error: p must be greater than 1" << endl;
        return false;
    }
    
    if (!isPrime(p)) {
        cout << "Warning: " << p << " is not prime. Continuing anyway..." << endl;
    }
    
    if (rows <= 0 || cols <= 0) {
        cerr << "Error: Rows and columns must be positive" << endl;
        return false;
    }
    
    if (n <= 0) {
        cerr << "Error: Number of matrices must be positive" << endl;
        return false;
    }
    
    if (rows * cols > p) {
        cerr << "Error: Need " << rows * cols << " unique elements, but only " 
             << p << " elements available (1 to " << p << ")" << endl;
        return false;
    }
    
    return true;
}

// Write matrix statistics to file
void writeMatrixStats(ofstream& fout, long matrixNum, const Mat<ZZ_p>& M, 
                     const ZZ_p& det, bool singular, long rows, long cols) {
    fout << "Matrix #" << matrixNum << " (" << rows << "x" << cols << "):\n";
    
    // Write matrix elements
    for (long i = 0; i < rows; i++) {
        for (long j = 0; j < cols; j++) {
            fout << setw(4) << M[i][j];
            if (j < cols - 1) fout << " ";
        }
        fout << "\n";
    }
    
    fout << "Determinant: " << det << "\n";
    fout << "Status: " << (singular ? "Singular" : "Nonsingular") << "\n";
    fout << string(50, '-') << "\n";
}

int main() {
    long p, rows, cols, n;
    
    cout << "=== Matrix Generator with Determinant Analysis ===" << endl;
    cout << "Enter prime p (elements will be from 1 to p): ";
    cin >> p;
    
    cout << "Enter rows: ";
    cin >> rows;
    cout << "Enter cols: ";
    cin >> cols;
    
    cout << "How many matrices to generate? ";
    cin >> n;
    
    // Validate inputs
    if (!validateInputs(p, rows, cols, n)) {
        return 1;
    }
    
    // Check if matrices are square for determinant calculation
    if (rows != cols) {
        cerr << "Error: Only square matrices can have determinants calculated." << endl;
        cout << "For non-square matrices, only matrix generation will be performed." << endl;
        
        char choice;
        cout << "Continue with matrix generation only? (y/n): ";
        cin >> choice;
        if (choice != 'y' && choice != 'Y') {
            return 0;
        }
    }
    
    ZZ_p::init(ZZ(p)); // Initialize field GF(p)
    
    // Create output file with timestamp
    auto now = chrono::system_clock::now();
    auto time_t = chrono::system_clock::to_time_t(now);
    
    string filename = "matrices_" + to_string(rows) + "x" + to_string(cols) + 
                     "_p" + to_string(p) + ".txt";
    
    ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "Error: Cannot create output file " << filename << endl;
        return 1;
    }
    
    // Write header
    fout << "=== MATRIX GENERATION REPORT ===" << endl;
    fout << "Generated on: " << put_time(localtime(&time_t), "%Y-%m-%d %H:%M:%S") << endl;
    fout << "Parameters:" << endl;
    fout << "  Prime p: " << p << endl;
    fout << "  Matrix size: " << rows << " x " << cols << endl;
    fout << "  Number of matrices: " << n << endl;
    fout << "  Elements range: 1 to " << p << endl;
    fout << string(60, '=') << endl << endl;
    
    long singularCount = 0, nonsingularCount = 0;
    long successfulGenerations = 0;
    
    ProgressBar progressBar(50, "Generating matrices");
    auto start_time = chrono::high_resolution_clock::now();
    
    for (long k = 0; k < n; k++) {
        try {
            Mat<ZZ_p> M = generateUniqueMatrix(rows, cols, p);
            successfulGenerations++;
            
            bool singular = false;
            ZZ_p det;
            
            // Calculate determinant only for square matrices
            if (rows == cols) {
                det = determinant(M);
                singular = (det == 0);
                
                if (singular) {
                    singularCount++;
                } else {
                    nonsingularCount++;
                }
            }
            
            // Write matrix to file
            if (rows == cols) {
                writeMatrixStats(fout, k + 1, M, det, singular, rows, cols);
            } else {
                fout << "Matrix #" << (k + 1) << " (" << rows << "x" << cols << "):\n";
                for (long i = 0; i < rows; i++) {
                    for (long j = 0; j < cols; j++) {
                        fout << setw(4) << M[i][j];
                        if (j < cols - 1) fout << " ";
                    }
                    fout << "\n";
                }
                fout << string(50, '-') << "\n";
            }
            
        } catch (const exception &e) {
            fout << "Error generating matrix #" << (k + 1) << ": " << e.what() << "\n";
            fout << string(50, '-') << "\n";
        }
        
        // Update progress
        double progress = (double)(k + 1) / n;
        progressBar.update(progress, k + 1, n);
    }
    
    progressBar.finish();
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    
    // Write summary
    fout << "\n" << string(60, '=') << endl;
    fout << "=== GENERATION SUMMARY ===" << endl;
    fout << "Total matrices requested: " << n << endl;
    fout << "Successfully generated: " << successfulGenerations << endl;
    
    if (rows == cols) {
        fout << "Singular matrices: " << singularCount << endl;
        fout << "Nonsingular matrices: " << nonsingularCount << endl;
        
        if (successfulGenerations > 0) {
            double singularPercent = (double)singularCount / successfulGenerations * 100;
            double nonsingularPercent = (double)nonsingularCount / successfulGenerations * 100;
            fout << "Singular percentage: " << fixed << setprecision(2) << singularPercent << "%" << endl;
            fout << "Nonsingular percentage: " << fixed << setprecision(2) << nonsingularPercent << "%" << endl;
        }
    }
    
    fout << "Generation time: " << duration.count() << " milliseconds" << endl;
    fout << string(60, '=') << endl;
    
    fout.close();
    
    // Console output
    cout << "\n=== RESULTS ===" << endl;
    cout << "Successfully generated: " << successfulGenerations << " out of " << n << " matrices" << endl;
    
    if (rows == cols) {
        cout << "Singular matrices: " << singularCount << endl;
        cout << "Nonsingular matrices: " << nonsingularCount << endl;
        
        if (successfulGenerations > 0) {
            double singularPercent = (double)singularCount / successfulGenerations * 100;
            cout << "Singular percentage: " << fixed << setprecision(2) << singularPercent << "%" << endl;
        }
    }
    
    cout << "Generation time: " << duration.count() << " milliseconds" << endl;
    cout << "Results saved to: " << filename << endl;
    
    return 0;
}

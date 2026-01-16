#include <NTL/ZZ_p.h>
#include <NTL/mat_ZZ_p.h>
#include <NTL/vector.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>

using namespace std;
using namespace NTL;

// Generate a random matrix with unique elements
Mat<ZZ_p> generateUniqueMatrix(long rows, long cols, long p) {
    if (rows * cols > p) {
        throw runtime_error("Matrix too large for unique elements modulo p");
    }

    // Create pool of numbers 0..p-1
    vector<long> pool(p);
    iota(pool.begin(), pool.end(), 0);

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

int main() {
    long p, rows, cols, n;
    cout << "Enter prime p: ";
    cin >> p;
    cout << "Enter rows and cols: ";
    cin >> rows >> cols;
    cout << "How many matrices to generate? ";
    cin >> n;

    if (rows != cols) {
        cerr << "Warning: Only square matrices can be checked for singularity." << endl;
        return 1;
    }

    ZZ_p::init(ZZ(p)); // Initialize field GF(p)

    ofstream fout("matrices.txt", ios::app); // append mode
    if (!fout.is_open()) {
        cerr << "Error opening file." << endl;
        return 1;
    }

    long singularCount = 0, nonsingularCount = 0;

    for (long k = 0; k < n; k++) {
        try {
            Mat<ZZ_p> M = generateUniqueMatrix(rows, cols, p);

            ZZ_p det = determinant(M);

            bool singular = (det == 0);
            if (singular)
                singularCount++;
            else
                nonsingularCount++;

            // Write matrix to file
            fout << "Matrix #" << (k + 1) << ":\n";
            for (long i = 0; i < rows; i++) {
                for (long j = 0; j < cols; j++) {
                    fout << M[i][j] << " ";
                }
                fout << "\n";
            }
            fout << (singular ? "Singular" : "Nonsingular") << "\n";
            fout << "-----------------------\n";
        }
        catch (const exception &e) {
            cerr << "Error: " << e.what() << endl;
        }
    }

    fout << "\n=== SUMMARY ===\n";
    fout << "Total Singular Matrices: " << singularCount << "\n";
    fout << "Total Nonsingular Matrices: " << nonsingularCount << "\n";
    fout << "=================\n\n";

    fout.close();

    cout << "Matrices written to matrices.txt" << endl;
    cout << "Singular: " << singularCount
    << ", Nonsingular: " << nonsingularCount << endl;

    return 0;
}

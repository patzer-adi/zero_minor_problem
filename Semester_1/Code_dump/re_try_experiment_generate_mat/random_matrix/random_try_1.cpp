#include <NTL/ZZ_p.h>
#include <NTL/mat_ZZ_p.h>
#include <NTL/vector.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

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
    long p, rows, cols;
    cout << "Enter prime p: ";
    cin >> p;
    cout << "Enter rows and cols: ";
    cin >> rows >> cols;

    ZZ_p::init(ZZ(p)); // Initialize field GF(p)

    try {
        Mat<ZZ_p> M = generateUniqueMatrix(rows, cols, p);

        cout << "Generated Matrix:" << endl;
        cout << M << endl;
    }
    catch (const exception &e) {
        cerr << "Error: " << e.what() << endl;
    }

    return 0;
}

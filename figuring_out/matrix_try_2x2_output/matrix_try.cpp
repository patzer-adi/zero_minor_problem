#include <iostream>
#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>
#include <fstream>
#include <chrono>  // For timing

using namespace std;
using namespace NTL;
using namespace chrono;  // For timing

void run()
{
    ofstream outfile("output_for_2x2_singular.txt");

    long m = 25, n = 25;
    Mat<ZZ> arr;
    arr.SetDims(m, n);

    // Start timing
    auto start = high_resolution_clock::now();

    outfile << "Generating random matrix A (" << m << "x" << n << "):\n";

    // Fill matrix with random values and write to file
    for (long i = 0; i < m; i++) {
        for (long j = 0; j < n; j++) {
            arr[i][j] = RandomBnd(100);
            outfile << arr[i][j] << "\t";
        }
        outfile << "\n";
    }

    long count = 0;

    outfile << "\nDeterminants of all 2x2 submatrices:\n";

    for (long r1 = 0; r1 < m; r1++) {
        for (long r2 = r1 + 1; r2 < m; r2++) {
            for (long c1 = 0; c1 < n; c1++) {
                for (long c2 = c1 + 1; c2 < n; c2++) {
                    ZZ d = arr[r1][c1] * arr[r2][c2] - arr[r1][c2] * arr[r2][c1];

                    // Store each determinant
                    outfile << "Rows(" << r1 << "," << r2 << ") Cols(" << c1 << "," << c2 << ") Det: " << d << "\n";

                    if (d == 0)
                        count++;
                }
            }
        }
    }

    long total = (m * (m - 1) / 2) * (n * (n - 1) / 2);

    // Stop timing
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    outfile << "\nNumber of 2x2 singular submatrices: " << count << endl;
    outfile << "Total 2x2 submatrices: " << total << endl;
    outfile << "Percentage singular: " << (100.0 * count / total) << "%" << endl;
    outfile << "Computation Time: " << duration.count() << " milliseconds\n";

    outfile.close();
}

int main()
{
    run();
    return 0;
}

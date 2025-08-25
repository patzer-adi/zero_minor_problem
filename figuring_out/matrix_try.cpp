#include <iostream>
#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>
#include <NTL/LLL.h>  // ⬅️ Needed for det()

using namespace std;
using namespace NTL;

void run()
{
    long m = 25, n = 25;

    Mat<ZZ> arr;
    arr.SetDims(m, n);

    cout << "Generating random matrix A (" << m << "x" << n << "):" << endl;
    for (long i = 0; i < m; i++) {
        for (long j = 0; j < n; j++) {
            arr[i][j] = RandomBnd(100);
        }
    }

    long count = 0;

    for (long r1 = 0; r1 < m; r1++) {
        for (long r2 = r1 + 1; r2 < m; r2++) {
            for (long c1 = 0; c1 < n; c1++) {
                for (long c2 = c1 + 1; c2 < n; c2++) {
                    Mat<ZZ> A;
                    A.SetDims(2, 2);
                    A[0][0] = arr[r1][c1];
                    A[0][1] = arr[r1][c2];
                    A[1][0] = arr[r2][c1];
                    A[1][1] = arr[r2][c2];

                    ZZ d = det(A);

                    cout << "Det: " << d << endl;

                    if (d == 0)
                        count++;
                }
            }
        }
    }

    cout << "Number of 2x2 singular submatrices: " << count << endl;
}

int main()
{
    run();
    return 0;
}

#include "EC_GF2E.hpp"
#include "EC_ZZp.hpp"
#include "EC_GF2E_Point.hpp"
#include "EC_lasVegas.tcc"

#include "MPI_utils.hpp"

#include <iomanip>
#include <NTL/matrix.h>
#include <NTL/mat_GF2E.h>

#include "constants.hpp"

template <class _mat_>
bool IsKernelVaid(const _mat_ &nonIdentityKernel)
{
    for (ulong i = 0; i < nonIdentityKernel.NumRows(); ++i)
        if (IsZero(nonIdentityKernel[i][0]) || IsZero(nonIdentityKernel[i][1]) || IsZero(nonIdentityKernel[i][2]))
            return false;

    for (ulong i = 0; i < nonIdentityKernel.NumRows(); ++i)
        for (ulong j = 0; j < nonIdentityKernel.NumCols(); ++j)
            if (IsZero(nonIdentityKernel[i][j]))
                return false;

    return true;
}

/**
 * @param P : Point P
 * @param Q : Point Q i.e. Q = mP
 * @param ordP : Order of P
 * @return : DLP
 */
template <class _EC_Point_, class V, class _mat_, class FiniteField>
// ORIGINAL: void makeKernelDB(_EC_Point_ &P, _EC_Point_ &Q, ZZ ordP, ulong _p, const int offset, V *EC, ulong numberOfKernelsToGenerate)
void makeKernelDB(_EC_Point_ &P, _EC_Point_ &Q, ZZ ordP, ulong _p, const double offset, V *EC, ulong numberOfKernelsToGenerate)
{
    int processorId;
    MPI_Comm_rank(MPI_COMM_WORLD, &processorId);

    if (processorId != MASTER_NODE)
        return;

    cout << "\n Generating " << numberOfKernelsToGenerate << " kernels on MASTER_NODE => pId :: " << processorId << endl;

    // ORIGINAL: const ulong n = offset * _p;
    // ORIGINAL: const ulong r = 3 * n;
    // ORIGINAL: // NEW FORMULA: n stays full (numberOfBits), only r scales with offset
    // ORIGINAL: // This makes saved output matrix r×r = 20% of full r×r at offset=1
    // ORIGINAL: // e.g. 25-bit offset=0.2 -> r=15 -> 15x15 matrix (vs 75x75 at offset=1)
    // ORIGINAL: // Note: floor truncation affects 27,33,34-bit (e.g. 0.2*27*3=16.2 -> r=16), accept this
    // ---
    // thesis: n' = log2(p), r = 3*n' (Algorithm 1, p.30, thesis by Abdullah Ansari)
    // offset here is a fractional multiplier on r (not additive as in thesis Table 4.2)
    // ORIGINAL: offset scaling broke k+t=mat_row invariant and caused zero-kernel
    // FIX: run full scale internally, truncate only at file-write time
    // This preserves all generateMatrix() assumptions while achieving the 20% output size
    const ulong n = _p;
    const ulong r = 3 * n;                                          // full scale always
    const ulong r_target = (ulong)(offset * 3.0 * (double)n);      // 15 for 25-bit, 16 for 27-bit
    const ulong k_randomNums = (3 * n) - 1, t_randomNums = r + 1;  // full scale
    const ulong mat_row = r + r, mat_col = ((n + 1) * (n + 2)) / 2;

    // PQ_randomNumbers sized to k+t = 6*n (full scale, correct for generateMatrix)
    ZZ PQ_randomNumbers[(k_randomNums + t_randomNums)];

    ulong weightedVector_arr[mat_col][3];
    generateWeightedVector(n, weightedVector_arr);

    ulong iterationCnt = 1;
    ulong accidentCnt = 0;
    ulong attempts = 0;
    const ulong maxAttempts = numberOfKernelsToGenerate * 5000;
    ulong invalidMatrixCnt = 0;
    ulong zeroKernelCnt = 0;
    ulong smallKernelCnt = 0;
    ulong invalidNonIdentityCnt = 0;

    cout << "\n iterationCnt :: " << iterationCnt << "\t numberOfkernesTOGenerate :: " << numberOfKernelsToGenerate << endl;
    // ORIGINAL: while (iterationCnt <= numberOfKernelsToGenerate)
    while (iterationCnt <= numberOfKernelsToGenerate && attempts < maxAttempts)
    {
        attempts++;
        cout << " Processing kernel :: " << iterationCnt << endl;

        double s_time = GetTime();
        // Fill k+t random numbers (now = 2*r = mat_row at any scale)
        generateRandomNumbers(k_randomNums + t_randomNums, PQ_randomNumbers, ordP);

        _mat_ M;
        M.SetDims(mat_row, mat_col);
        double time_MStart = GetTime();
        int result = generateMatrix(M, P, Q, k_randomNums, t_randomNums, PQ_randomNumbers, weightedVector_arr, EC);
        double time_MEnd = GetTime();
        if (result == 1)
        {
            accidentCnt++;
            invalidMatrixCnt++;
            // ORIGINAL: if (attempts % 1 == 0)
            if (attempts % 1000 == 0)
                cout << "  attempts=" << attempts << " invalidMatrixCnt=" << invalidMatrixCnt << endl;
            continue;
        }

        _mat_ ker;
        double start_kTime = GetTime();
        kernel(ker, M);
        double end_kTime = GetTime();

        if (ker.NumRows() == 0)
        {
            zeroKernelCnt++;
            // ORIGINAL: if (attempts % 1 == 0)
            if (attempts % 1000 == 0)
                cout << "  attempts=" << attempts << " zeroKernelCnt=" << zeroKernelCnt << endl;
            continue;
        }

        if (ker.NumRows() < r)
        {
            smallKernelCnt++;
            // ORIGINAL: if (attempts % 1 == 0)
            if (attempts % 1000 == 0)
                cout << "  attempts=" << attempts << " smallKernelCnt=" << smallKernelCnt << " ker.NumRows()=" << ker.NumRows() << " r=" << r << endl;
            continue;
        }

        _mat_ nonIdentityKernel = getNonIdentityMatrixFromKernel(ker);

        if (!IsKernelVaid(nonIdentityKernel))
        {
            invalidNonIdentityCnt++;
            // ORIGINAL: if (attempts % 1 == 0)
            if (attempts % 1000 == 0)
                cout << "  attempts=" << attempts << " invalidNonIdentityCnt=" << invalidNonIdentityCnt << endl;
            continue;
        }

        char *kernelFileName = new char[200];
        // ORIGINAL: sprintf(kernelFileName, "kernel_DB/new/kernel_%u_%u.txt", _p, iterationCnt);
        // Output now routed to Kernel_output/<numberOfBits>/ per bit-size
        // FIX: use names Kernel_output/<bits>/kernel_<bits>_<i>.txt
        sprintf(kernelFileName, "kernel_output/%lu/kernel_%lu_%lu.txt", _p, _p, iterationCnt);
        cout << " FILE-NAME :: " << kernelFileName << endl;
        ofstream kernelFile(kernelFileName);
        if (!kernelFile.is_open()) {
            cerr << " ERROR: Cannot open kernel file: " << kernelFileName << endl;
            delete[] kernelFileName;
            continue;
        }

        // ORIGINAL: kernelFile << nonIdentityKernel;  (full r×r matrix)
        // FIX: write only r_target×r_target top-left submatrix
        _mat_ truncatedKernel;
        truncatedKernel.SetDims(r_target, r_target);
        for (ulong i = 0; i < r_target; ++i)
            for (ulong j = 0; j < r_target; ++j)
                truncatedKernel[i][j] = nonIdentityKernel[i][j];
        kernelFile << truncatedKernel;
        kernelFile.close();

        char *randomNumberfileName = new char[200];
        // ORIGINAL: sprintf(randomNumberfileName, "kernel_DB/new/kernel_%u_%u_RN.txt", _p, iterationCnt);
        // FIX: match kernel naming for RN files as Kernel_output/<bits>/kernel_<bits>_<i>_RN.txt
        sprintf(randomNumberfileName, "kernel_output/%lu/kernel_%lu_%lu_RN.txt", _p, _p, iterationCnt);
        saveRandomNumberToFile(randomNumberfileName, PQ_randomNumbers, mat_row);
        iterationCnt++;
    }

    if (iterationCnt <= numberOfKernelsToGenerate)
    {
        cout << "\n WARNING: stopped after maxAttempts=" << maxAttempts
             << " with generated kernels=" << (iterationCnt - 1) << " / " << numberOfKernelsToGenerate << endl;
        cout << "  invalidMatrixCnt=" << invalidMatrixCnt
             << " zeroKernelCnt=" << zeroKernelCnt
             << " smallKernelCnt=" << smallKernelCnt
             << " invalidNonIdentityCnt=" << invalidNonIdentityCnt << endl;
    }
}
/*
 * File:   main.cpp
 * Author: abdullah and now currently Aditya Gowari
 *
 * Created on 28 November, 2017, 11:12 AM  and now updated on 30/3/26, 3:48 PM
 */

#include "EC_GF2E.hpp"
#include "EC_ZZp.hpp"
#include "EC_ZZp_Point.hpp"
#include "constants.hpp"
#include "containment.tcc"
#include "dlp_input.hpp"
#include "dlp_input_2m.hpp"
#include "hypothesis_A.tcc"
#include "lasVegas.tcc"
#include "playground.tcc"

#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;
using namespace NTL;

const ulong numberOfInputs = 1;
const ulong numberOfKernelsToGenerate = 100;
// These two function can be converted into template.
// Re-write these two function in a master-slave model.
#if 0
void fun_ZZp()
{
    int processorId, numberOfProcessors;

    MPI_Comm_rank(MPI_COMM_WORLD, &processorId);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);
    string fileName = "input/3.txt";

    dlp_input dd(fileName);
    masterPrint(processorId) << "\n Number of inputs :: " << dd.numberOfInputs << endl;

    for (int i = 0; i < 1; ++i)
    {
        // char *fileName_o = new char[200];
        // sprintf(fileName_o, "output/p_%u_%u_%d.txt", processorId, numberOfProcessors, i);
        // freopen(fileName_o, "w", stdout);

        if (dd.numberOfInputs < i)
            break;

        EC_ZZp EC(dd.data[i].p, dd.data[i].a, dd.data[i].b, dd.data[i].ordP);
        EC_ZZp_Point P, Q;

        P.x = conv<ZZ_p>(dd.data[i].Px);
        P.y = conv<ZZ_p>(dd.data[i].Py);

        Q.x = conv<ZZ_p>(dd.data[i].Qx);
        Q.y = conv<ZZ_p>(dd.data[i].Qy);

        ulong numberOfBits = NumBits(dd.data[i].ordP);
        if (processorId == MASTER_NODE)
        {
            masterPrint(processorId) << "\n Field Size :: " << dd.data[i].p << endl;
            P.printPoint("\n P ");
            Q.printPoint("\t Q ");
            masterPrint(processorId) << "\n\n Ord :: " << dd.data[i].ordP << "\t sqrt(Ord) :: " << SqrRoot(dd.data[i].ordP);
            masterPrint(processorId) << "\t m :: " << dd.data[i].e << "\t num-Of-Bits :: " << numberOfBits << endl;
        }

        // ZZ ans = hypothesis_A<EC_ZZp_Point, EC_ZZp, mat_ZZ_p, ZZ_p>(P, Q, dd.data[i].ordP, numberOfBits, 1, EC.address());

        ZZ ans = lasVegas<EC_ZZp_Point, EC_ZZp, mat_ZZ_p, ZZ_p>(P, Q, dd.data[i].ordP, numberOfBits, 1, EC.address());

        // makeKernelDB<EC_ZZp_Point, EC_ZZp, mat_ZZ_p, ZZ_p>(P, Q, dd.data[i].ordP, numberOfBits,
        //                                                    1, EC.address(), numberOfKernelsToGenerate);

        MPI_Barrier(MPI_COMM_WORLD);
    }
}

#if 0
// ORIGINAL: void fun_ZZp()
void fun_ZZp()
{
    int processorId, numberOfProcessors;
    MPI_Comm_rank(MPI_COMM_WORLD, &processorId);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);

    // ORIGINAL: const int offset = 1;
    // offset=0.2 -> saved matrix is 20% of full size (r = floor(0.2 * 3 * numberOfBits))
    // e.g. 25-bit: r_full=75 -> r_scaled=15 -> saved matrix 15x15
    // 27,33,34-bit will have floor truncation (e.g. 0.2*3*27=16.2 -> r=16), this is accepted
    const double offset = 0.35; //change offset here

    // One input file per bit-size only, using _1.txt files
    // Do NOT use _2.txt _3.txt etc. Do NOT modify any input files
    // Last number before # in each file is the prime used
    // FIX: 25–29 from input/25_29/, 30–35 from input/exp/
    std::vector<std::string> inputFiles = {
        "input/25_29/25_1.txt",
        "input/25_29/26_1.txt",
        "input/25_29/27_1.txt",
        "input/25_29/28_1.txt",
        "input/25_29/29_1.txt",
        "input/exp/30_1.txt",
        "input/exp/31_1.txt",
        "input/exp/32_1.txt",
        "input/exp/33_1.txt",
        "input/exp/34_1.txt",
        "input/exp/35_1.txt"
    };

    for (const std::string &fileName : inputFiles)
    {
        if (processorId == MASTER_NODE)
            cout << "\n\n====== Processing file: " << fileName << " ======" << endl;

        dlp_input dd(fileName);

        if (dd.numberOfInputs < 1)
        {
            if (processorId == MASTER_NODE)
                cout << "\n No inputs found in: " << fileName << ", skipping." << endl;
            continue;
        }

        EC_ZZp EC(dd.data[0].p, dd.data[0].a, dd.data[0].b, dd.data[0].ordP);
        EC_ZZp_Point P, Q;

        P.x = conv<ZZ_p>(dd.data[0].Px);
        P.y = conv<ZZ_p>(dd.data[0].Py);
        Q.x = conv<ZZ_p>(dd.data[0].Qx);
        Q.y = conv<ZZ_p>(dd.data[0].Qy);

        // ORIGINAL: ulong numberOfBits = NumBits(dd.data[0].ordP);
        // FIX: use NumBits of field prime p, not ordP
        // ordP (group order) can be 1 bit larger than p (e.g. 33-bit field gives 34-bit ordP)
        // Using ordP caused some inputs (e.g. 27_1.txt, 33_1.txt) to be classified under the wrong bit-size
        // The thesis defines n' = log2(p) where p is the field prime (Algorithm 1, p.30)
        ulong numberOfBits = NumBits(dd.data[0].p);

        if (processorId == MASTER_NODE)
        {
            masterPrint(processorId) << "\n Field Size   :: " << dd.data[0].p << endl;
            P.printPoint("\n P ");
            Q.printPoint("\t Q ");
            masterPrint(processorId) << "\n Ord          :: " << dd.data[0].ordP << endl;
            masterPrint(processorId) << " numberOfBits :: " << numberOfBits << endl;
            masterPrint(processorId) << " offset       :: " << offset << endl;
            masterPrint(processorId) << " n            :: " << numberOfBits << " (full, not scaled)" << endl;
            masterPrint(processorId) << " r            :: " << (ulong)(offset * 3.0 * (double)numberOfBits)
                                     << " (= floor(offset * 3 * n))" << endl;
            masterPrint(processorId) << " saved matrix :: "
                                     << (ulong)(offset * 3.0 * (double)numberOfBits)
                                     << "x"
                                     << (ulong)(offset * 3.0 * (double)numberOfBits)
                                     << endl;
            // NOTE: for 27,33,34-bit the matrix dimension will not be exactly 20% of full
            // due to floor truncation. This is expected and accepted.
        }

        // Create output directory Kernel_output/<numberOfBits>/
        // Kernels will be saved as Kernel_output/<bits>/<bits>_1.txt to <bits>_100.txt
        char mkdirCmd[300];
        sprintf(mkdirCmd, "mkdir -p kernel_output/%lu", numberOfBits);
        system(mkdirCmd);

        // ORIGINAL: ZZ ans = lasVegas<EC_ZZp_Point, EC_ZZp, mat_ZZ_p, ZZ_p>(P, Q, dd.data[i].ordP, numberOfBits, 1, EC.address());
        // Using makeKernelDB instead of lasVegas, with double offset and loop over input files
        makeKernelDB<EC_ZZp_Point, EC_ZZp, mat_ZZ_p, ZZ_p>(
            P, Q, dd.data[0].ordP, numberOfBits, offset, EC.address(), numberOfKernelsToGenerate);

        MPI_Barrier(MPI_COMM_WORLD);
    }
}
#endif

void fun_ZZp()
{
    int processorId, numberOfProcessors;
    MPI_Comm_rank(MPI_COMM_WORLD, &processorId);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);

    // ORIGINAL: const int offset = 1;
    // offset is a fractional multiplier on r = 3*n' (thesis Algorithm 1, p.30)
    // offset=1.0 -> full matrix, r=3*n', saved kernel rxr (e.g. 25-bit: 75x75)
    // offset=0.2 -> 20% matrix, r=floor(0.2*3*n'), saved kernel rxr (e.g. 25-bit: 15x15)
    // Reference: thesis "A new attack on the Elliptic Curve Discrete Logarithm Problem"
    //            Abdullah Zubair Ansari, SPPU 2023, Section 2.2 and Algorithm 1
    const double offset = 0.35;

    // Input files: one per bit-size, _1.txt only (not _2.txt, _3.txt etc.)
    // These are ZZp (prime field) inputs from input/25_29/ and input/exp/
    // Last number before # in each file is the prime used (Appendix A.2, thesis p.74)
    // DO NOT modify input files
    std::vector<std::string> inputFiles = {
        "input/25_29/25_1.txt",
        "input/25_29/26_1.txt",
        "input/25_29/27_1.txt",
        "input/25_29/28_1.txt",
        "input/25_29/29_1.txt",
        "input/exp/35_1.txt"
        // Add more input/exp/*_1.txt entries here as needed
    };

    for (const std::string &fileName : inputFiles)
    {
        if (processorId == MASTER_NODE)
            cout << "\n\n====== Processing: " << fileName << " ======" << endl;

        dlp_input dd(fileName);

        if (dd.numberOfInputs < 1)
        {
            if (processorId == MASTER_NODE)
                cout << "\n No inputs in: " << fileName << ", skipping." << endl;
            continue;
        }

        EC_ZZp EC(dd.data[0].p, dd.data[0].a, dd.data[0].b, dd.data[0].ordP);
        EC_ZZp_Point P, Q;

        P.x = conv<ZZ_p>(dd.data[0].Px);
        P.y = conv<ZZ_p>(dd.data[0].Py);
        Q.x = conv<ZZ_p>(dd.data[0].Qx);
        Q.y = conv<ZZ_p>(dd.data[0].Qy);

        // ORIGINAL: numberOfBits = n' = log2(ordP) per thesis Algorithm 1 line 1 (p.30)
        // FIX: use field prime p not ordP — ordP can be 1 bit larger than p
        // e.g. 27-bit field: p=134217689 (27-bit), ordP=134218103 (28-bit)
        // e.g. 33-bit field: p=8589934583 (33-bit), ordP=8589937303 (34-bit)
        ulong numberOfBits = NumBits(dd.data[0].p);
        ulong r_scaled = (ulong)(offset * 3.0 * (double)numberOfBits);

        if (processorId == MASTER_NODE)
        {
            masterPrint(processorId) << "\n Field Size   :: " << dd.data[0].p << endl;
            P.printPoint("\n P ");
            Q.printPoint("\t Q ");
            masterPrint(processorId) << "\n Ord          :: " << dd.data[0].ordP << endl;
            masterPrint(processorId) << " n' (numberOfBits)    :: " << numberOfBits << endl;
            masterPrint(processorId) << " r_full (3*n')        :: " << 3 * numberOfBits << endl;
            masterPrint(processorId) << " offset               :: " << offset << endl;
            masterPrint(processorId) << " r_scaled             :: " << r_scaled << endl;
            masterPrint(processorId) << " saved matrix size    :: " << r_scaled << "x" << r_scaled << endl;
            // Note: for 27,33,34-bit floor truncation gives non-exact 20% - this is expected
        }

        // Create output directory: kernel_output/<numberOfBits>/
        // Output files: kernel_output/<bits>/<bits>_1.txt to <bits>_100.txt
        char mkdirCmd[300];
        sprintf(mkdirCmd, "mkdir -p kernel_output/%lu", numberOfBits);
        system(mkdirCmd);

        // ORIGINAL: ZZ ans = lasVegas<EC_ZZp_Point, EC_ZZp, mat_ZZ_p, ZZ_p>(P, Q, dd.data[i].ordP, numberOfBits, 1, EC.address());
        // Changed: use makeKernelDB instead of lasVegas, with double offset=0.2
        // This generates 100 kernels per input file (numberOfKernelsToGenerate=100, set at top of main.cpp)
        makeKernelDB<EC_ZZp_Point, EC_ZZp, mat_ZZ_p, ZZ_p>(
            P, Q, dd.data[0].ordP, numberOfBits, offset, EC.address(), numberOfKernelsToGenerate);

        MPI_Barrier(MPI_COMM_WORLD);
    }
}
#endif

void fun_ZZp() {
  int processorId, numberOfProcessors;
  MPI_Comm_rank(MPI_COMM_WORLD, &processorId);
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);

  // ORIGINAL: const int offset = 1;
  // offset=0.2 -> saved matrix is 20% of full size (r = floor(0.2 * 3 *
  // numberOfBits)) e.g. 25-bit: r_full=75 -> r_scaled=15 -> saved matrix 15x15
  // 27,33,34-bit will have floor truncation (e.g. 0.2*3*27=16.2 -> r=16), this
  // is accepted
  const double offset = 0.35;

  // One input file per bit-size only, using _1.txt files
  // Do NOT use _2.txt _3.txt etc. Do NOT modify any input files
  // Last number before # in each file is the prime used
  // FIX: 25–29 from input/25_29/, 30–35 from input/exp/

/*
  std::vector<std::string> inputFiles = {
      "input/25_29/25_1.txt", "input/25_29/26_1.txt", "input/25_29/27_1.txt",
      "input/25_29/28_1.txt", "input/25_29/29_1.txt", "input/exp/30_1.txt",
      "input/exp/31_1.txt",   "input/exp/32_1.txt",   "input/exp/33_1.txt",
      "input/exp/34_1.txt",   "input/exp/35_1.txt"};*/

std::vector<std::string> inputFiles = {
    "input/36_40/36_1.txt","input/36_40/37_1.txt","input/36_40/38_1.txt","input/36_40/39_1.txt","input/36_40/40_1.txt"};

  for (const std::string &fileName : inputFiles) {
    if (processorId == MASTER_NODE)
      cout << "\n\n====== Processing file: " << fileName << " ======" << endl;

    dlp_input dd(fileName);

    if (dd.numberOfInputs < 1) {
      if (processorId == MASTER_NODE)
        cout << "\n No inputs found in: " << fileName << ", skipping." << endl;
      continue;
    }

    EC_ZZp EC(dd.data[0].p, dd.data[0].a, dd.data[0].b, dd.data[0].ordP);
    EC_ZZp_Point P, Q;

    P.x = conv<ZZ_p>(dd.data[0].Px);
    P.y = conv<ZZ_p>(dd.data[0].Py);
    Q.x = conv<ZZ_p>(dd.data[0].Qx);
    Q.y = conv<ZZ_p>(dd.data[0].Qy);

    ulong numberOfBits = NumBits(dd.data[0].p);

    if (processorId == MASTER_NODE) {
      masterPrint(processorId) << "\n Field Size   :: " << dd.data[0].p << endl;
      P.printPoint("\n P ");
      Q.printPoint("\t Q ");
      masterPrint(processorId)
          << "\n Ord          :: " << dd.data[0].ordP << endl;
      masterPrint(processorId) << " numberOfBits :: " << numberOfBits << endl;
      masterPrint(processorId) << " offset       :: " << offset << endl;
      masterPrint(processorId) << " n            :: " << numberOfBits
                               << " (full, not scaled)" << endl;
      masterPrint(processorId)
          << " r            :: " << (ulong)(offset * 3.0 * (double)numberOfBits)
          << " (= floor(offset * 3 * n))" << endl;
      masterPrint(processorId)
          << " saved matrix :: " << (ulong)(offset * 3.0 * (double)numberOfBits)
          << "x" << (ulong)(offset * 3.0 * (double)numberOfBits) << endl;
      // NOTE: for 27,33,34-bit the matrix dimension will not be exactly 20% of
      // full due to floor truncation. This is expected and accepted.
    }

    // Create output directory Kernel_output/<numberOfBits>/
    // Kernels will be saved as Kernel_output/<bits>/<bits>_1.txt to
    // <bits>_100.txt
    char mkdirCmd[300];
    sprintf(mkdirCmd, "mkdir -p kernel_output/%lu", numberOfBits);
    system(mkdirCmd);

    // ORIGINAL: ZZ ans = lasVegas<EC_ZZp_Point, EC_ZZp, mat_ZZ_p, ZZ_p>(P, Q,
    // dd.data[i].ordP, numberOfBits, 1, EC.address()); Using makeKernelDB
    // instead of lasVegas, with double offset and loop over input files
    makeKernelDB<EC_ZZp_Point, EC_ZZp, mat_ZZ_p, ZZ_p>(
        P, Q, dd.data[0].ordP, numberOfBits, offset, EC.address(),
        numberOfKernelsToGenerate);

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void fun_GF2EX() {
  int processorId, numberOfProcessors;

  MPI_Comm_rank(MPI_COMM_WORLD, &processorId);
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);
  string fileName = "newNewInput/2_25.txt";

  dlp_input_2m dd(fileName);

  for (int i = 0; i < 1; ++i) {
    // char *fileName = new char[200];
    // sprintf(fileName, "output/p_%u_%u_%d.txt", processorId,
    // numberOfProcessors, i); freopen(fileName, "w", stdout);

    masterPrint(processorId) << "\n Processing input :: " << (i + 1) << endl;
    if (dd.numberOfInputs < i)
      break;

    EC_GF2E EC(dd.data[i].p, dd.data[i].irrd, dd.data[i].a, dd.data[i].b);
    EC_GF2E_Point P, Q;

    P.x._GF2E__rep = dd.data[i].Px;
    P.y._GF2E__rep = dd.data[i].Py;

    Q.x._GF2E__rep = dd.data[i].Qx;
    Q.y._GF2E__rep = dd.data[i].Qy;

    if (processorId == MASTER_NODE) {
      cout << "\n Field Size :: 2^" << dd.data[i].p << endl;
      P.printPoint1("\n P ");
      Q.printPoint1("\t Q ");
      cout << "\n\n Ord :: " << dd.data[i].ordP
           << "\t sqrt(Ord) :: " << SqrRoot(dd.data[i].ordP);
      cout << "\t m :: " << dd.data[i].e << endl;
    }
    ulong numberOfBits = NumBits(dd.data[i].ordP);
    ZZ ans = lasVegas<EC_GF2E_Point, EC_GF2E, mat_GF2E, GF2E>(
        P, Q, dd.data[i].ordP, numberOfBits, 1, EC.address());
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void inverseTest_ZZp() {
  ulong p = 1009;
  ZZ_p::init(conv<ZZ>(p));

  string fileName = "kernel_DB/1009/kernel_p_1009_2.txt";
  ifstream kernelFile(fileName);
  cout << "\n fileName :: " << fileName << endl;

  mat_ZZ_p mat, invMat;
  kernelFile >> mat;
  invMat = inv(mat);

  cout << "\n mat :: \n" << mat << endl;

  cout << "\n invMat :: \n" << invMat << endl;

  isMinorPresent(mat, 2, 11);
  cout << "\n++++++++++++++++++++++++++++++++++++\n";
  isMinorPresent(invMat, 2, 11);
}

int main(int argc, char **argv) {
  MPI_Init(NULL, NULL);

  int numberOfProcessors;
  int processorId;

  MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);
  MPI_Comm_rank(MPI_COMM_WORLD, &processorId);

  // processBigMinors_parallel();
  // processBiggerMinors();

  // LU_Circular_PrincipleMinorTest();
  // gaussianElimination_multiple();

  // fun_ZZp();
  // ORIGINAL: fun_GF2EX();
  fun_ZZp();

  MPI_Finalize();

  return 0;
}

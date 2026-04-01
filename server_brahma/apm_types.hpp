// =============================================================================
// apm_types.hpp — Shared data structures and constants for APM search
// =============================================================================

#ifndef APM_TYPES_HPP
#define APM_TYPES_HPP

#include <string>
#include <vector>
using namespace std;

// =============================================================================
// Compile-time constants
// =============================================================================

static const int PM_SIZE        = 2;    // principal minor block size — FIXED at 2
static const int MIN_DEV        = 2;    // minimum deviation to test
static const int MAX_IDX_STATIC = 50;   // max indices in any IndexSet (supports up to 52×52)
static const int EARLY_STOP_HIT = 100;  // early-stop threshold for matrices hit

// =============================================================================
// Data structures
// =============================================================================

// One row-set or col-set: sorted indices into the matrix
struct IndexSet {
    int k;                       // number of indices (= PM_SIZE + dev)
    int idx[MAX_IDX_STATIC];     // sorted matrix indices
};

// Parsed matrix
struct MatrixData {
    string filename;
    int n;
    vector<long long> data; // flat n×n row-major
};

// One recorded zero minor
struct ZeroMinor {
    int k;                       // minor size
    int dev;                     // deviation level
    int s;                       // principal block anchor
    int row_idx[MAX_IDX_STATIC];
    int col_idx[MAX_IDX_STATIC];
    double time_ms;              // time since matrix start (ms)
};

// Prime descriptor for one folder/group
struct FolderPrime {
    int folder_id;
    long long prime;
    char label[64];
};

#endif // APM_TYPES_HPP

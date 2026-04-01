// =============================================================================
// apm_search.cuh — APM search engine header (CUDA kernel + host search logic)
// =============================================================================

#ifndef APM_SEARCH_CUH
#define APM_SEARCH_CUH

#include "apm_types.hpp"
#include "gpu_buffers.cuh"
#include <string>
#include <vector>

class ApmSearchEngine {
public:
    // -------------------------------------------------------------------------
    // Search one matrix for the FIRST zero minor at a given deviation level.
    // Returns at most 1 ZeroMinor (first-hit logic).
    //
    // gpu:              pre-allocated GPU buffers for this (n, dev)
    // h_mat:            host pointer to flat n×n matrix data
    // n:                matrix dimension
    // prime:            the prime for modular arithmetic
    // dev:              deviation level
    // minors_tested:    [out] total number of minors tested
    // matrix_start_ms:  timestamp when matrix processing started
    // -------------------------------------------------------------------------
    static std::vector<ZeroMinor>
    search_matrix(GpuBuffers &gpu, const long long *h_mat, int n,
                  long long prime, int dev, double &minors_tested,
                  double matrix_start_ms);

    // -------------------------------------------------------------------------
    // Process one prime group at one deviation level.
    // Returns the number of matrices that had at least one zero minor.
    //
    // group:      prime group ID
    // prime:      the prime value
    // dev:        deviation level
    // files:      list of matrix file paths
    // n:          expected matrix dimension
    // gpu:        pre-allocated GPU buffers
    // base_dir:   output base directory (e.g. "Results_brahma")
    // -------------------------------------------------------------------------
    static int run_group_deviation(int group, long long prime, int dev,
                                   const std::vector<std::string> &files,
                                   int n, GpuBuffers &gpu,
                                   const std::string &base_dir);

private:
    // Generate all C(pool.size(), r) index sets for one anchor position.
    // principal: the PM_SIZE indices forming the principal block.
    static void gen_combos(const std::vector<int> &pool, int r,
                           const int *principal,
                           std::vector<IndexSet> &out);
};

#endif // APM_SEARCH_CUH

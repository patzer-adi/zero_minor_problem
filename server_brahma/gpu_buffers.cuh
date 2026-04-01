// =============================================================================
// gpu_buffers.cuh — GPU memory manager with runtime VRAM detection
//
// Uses cudaMemGetInfo() to detect available VRAM at runtime instead of a
// hardcoded limit. Reserves 90% of free VRAM for safety margin.
// =============================================================================

#ifndef GPU_BUFFERS_CUH
#define GPU_BUFFERS_CUH

#include "apm_types.hpp"
#include "file_utils.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// =============================================================================
// CUDA error check macro
// =============================================================================

#define CK(call)                                                               \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d -- %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(_e));                                    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// =============================================================================
// GpuBuffers class — manages 4 device pointers with VRAM-aware allocation
// =============================================================================

class GpuBuffers {
public:
    IndexSet  *d_row_sets;
    IndexSet  *d_col_sets;
    long long *d_matrix;
    int       *d_zero_flags;

    size_t cap_rows;   // allocated row-set capacity
    size_t col_chunk;  // column chunk size (computed at runtime)
    size_t flag_cap;   // cap_rows * col_chunk

    GpuBuffers()
        : d_row_sets(NULL), d_col_sets(NULL), d_matrix(NULL), d_zero_flags(NULL),
          cap_rows(0), col_chunk(0), flag_cap(0) {}

    ~GpuBuffers() { free_all(); }

    // -------------------------------------------------------------------------
    // Allocate GPU buffers for a given matrix size n and deviation level dev.
    // Queries actual free VRAM and uses 90% of it.
    // -------------------------------------------------------------------------
    void alloc(int n, int dev) {
        free_all();

        // Query actual GPU VRAM
        size_t free_bytes = 0, total_bytes = 0;
        CK(cudaMemGetInfo(&free_bytes, &total_bytes));

        // Use 90% of free VRAM to leave headroom for the driver/other allocations
        size_t avail_bytes = static_cast<size_t>(free_bytes * 0.90);

        printf("    [GPU VRAM] Total: %.0f MB | Free: %.0f MB | Usable (90%%): %.0f MB\n",
               total_bytes / 1e6, free_bytes / 1e6, avail_bytes / 1e6);

        // Upper bound: C(n - PM_SIZE, dev) = C(n-2, dev)
        int pool = n - PM_SIZE;
        cap_rows = static_cast<size_t>(FileUtils::nCr(pool, dev));

        // Subtract fixed allocations
        size_t fixed = cap_rows * sizeof(IndexSet)                      // d_row_sets
                     + static_cast<size_t>(n) * n * sizeof(long long);  // d_matrix

        if (fixed >= avail_bytes) {
            fprintf(stderr,
                    "[ERROR] Fixed allocations (%zu MB) exceed available VRAM (%zu MB)\n",
                    fixed / (1024 * 1024), avail_bytes / (1024 * 1024));
            exit(1);
        }
        size_t remain = avail_bytes - fixed;

        // remain must hold: col_chunk * sizeof(IndexSet) + cap_rows * col_chunk * sizeof(int)
        //                  = col_chunk * (sizeof(IndexSet) + cap_rows * sizeof(int))
        size_t per_col = sizeof(IndexSet) + cap_rows * sizeof(int);
        size_t max_chunk = (per_col > 0) ? remain / per_col : 512;
        if (max_chunk < 1)         max_chunk = 1;
        if (max_chunk > cap_rows)  max_chunk = cap_rows;

        // Round down to power of 2 for alignment (capped at cap_rows)
        col_chunk = 1;
        while (col_chunk * 2 <= max_chunk)
            col_chunk *= 2;
        if (col_chunk > cap_rows)  col_chunk = cap_rows;
        if (col_chunk < 1)         col_chunk = 1;

        flag_cap = cap_rows * col_chunk;

        printf("    [GPU] Allocating: rows=%zu  col_chunk=%zu  flags=%zu (%.1f MB)"
               "  matrix=%dx%d\n",
               cap_rows, col_chunk, flag_cap,
               flag_cap * sizeof(int) / 1e6, n, n);
        fflush(stdout);

        CK(cudaMalloc(&d_row_sets,  cap_rows * sizeof(IndexSet)));
        CK(cudaMalloc(&d_col_sets,  col_chunk * sizeof(IndexSet)));
        CK(cudaMalloc(&d_matrix,    static_cast<size_t>(n) * n * sizeof(long long)));
        CK(cudaMalloc(&d_zero_flags, flag_cap * sizeof(int)));
    }

    // -------------------------------------------------------------------------
    // Free all device memory
    // -------------------------------------------------------------------------
    void free_all() {
        if (d_row_sets)  { cudaFree(d_row_sets);  d_row_sets  = NULL; }
        if (d_col_sets)  { cudaFree(d_col_sets);  d_col_sets  = NULL; }
        if (d_matrix)    { cudaFree(d_matrix);    d_matrix    = NULL; }
        if (d_zero_flags){ cudaFree(d_zero_flags); d_zero_flags = NULL; }
    }

private:
    // Non-copyable
    GpuBuffers(const GpuBuffers &);
    GpuBuffers &operator=(const GpuBuffers &);
};

#endif // GPU_BUFFERS_CUH

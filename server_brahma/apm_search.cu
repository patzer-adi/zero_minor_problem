// =============================================================================
// apm_search.cu — APM search engine implementation (CUDA kernel + host logic)
// =============================================================================

#include "apm_search.cuh"
#include "file_utils.hpp"
#include "matrix_parser.hpp"
#include "mod_arith.cuh"
#include "result_writer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

// =============================================================================
// CUDA kernel
//
// Each thread handles one (row_set, col_set) pair.
// Thread computes (r, c) from tid — NO MinorJob array.
// Writes d_zero_flags[tid] = 1 if det == 0, else 0.
// Host scans flags and stops at the FIRST zero found (first-hit logic).
// =============================================================================

__global__ void
apm_kernel(const long long *d_matrix,   // flat n×n raw integers
           int n,
           const IndexSet *d_row_sets,  // all row index sets
           const IndexSet *d_col_sets,  // current chunk of col index sets
           int num_row_sets,
           int num_col_sets,            // actual count in this chunk
           int k,                       // minor size = PM_SIZE + dev
           long long prime,
           int *d_zero_flags)           // output: 1 if det==0, 0 otherwise
{
    long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = static_cast<long long>(num_row_sets) * num_col_sets;
    if (tid >= total) return;

    int r = static_cast<int>(tid / num_col_sets);
    int c = static_cast<int>(tid % num_col_sets);

    // Extract and reduce k×k submatrix
    long long sub[MAX_IDX_STATIC * MAX_IDX_STATIC];
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            long long v = d_matrix[d_row_sets[r].idx[i] * n + d_col_sets[c].idx[j]];
            sub[i * k + j] = ((v % prime) + prime) % prime;
        }
    }

    d_zero_flags[tid] = (det_mod(sub, k, prime) == 0LL) ? 1 : 0;
}

// =============================================================================
// Generate all C(pool_sz, r) index sets for one anchor position
// =============================================================================

void ApmSearchEngine::gen_combos(const std::vector<int> &pool, int r,
                                  const int *principal,
                                  std::vector<IndexSet> &out) {
    int psz = static_cast<int>(pool.size());
    if (r > psz) return;

    std::vector<int> cidx(r);
    for (int i = 0; i < r; i++) cidx[i] = i;

    while (true) {
        IndexSet is;
        is.k = PM_SIZE + r;
        for (int i = 0; i < PM_SIZE; i++)
            is.idx[i] = principal[i];
        for (int i = 0; i < r; i++)
            is.idx[PM_SIZE + i] = pool[cidx[i]];
        std::sort(is.idx, is.idx + is.k);
        out.push_back(is);

        // Advance combination
        int carry = r - 1;
        while (carry >= 0 && cidx[carry] == psz - r + carry)
            carry--;
        if (carry < 0) break;
        cidx[carry]++;
        for (int i = carry + 1; i < r; i++)
            cidx[i] = cidx[i - 1] + 1;
    }
}

// =============================================================================
// Search one matrix for the FIRST zero minor (first-hit logic)
// =============================================================================

std::vector<ZeroMinor>
ApmSearchEngine::search_matrix(GpuBuffers &gpu, const long long *h_mat,
                                int n, long long prime, int dev,
                                double &minors_tested,
                                double matrix_start_ms) {
    std::vector<ZeroMinor> found;
    minors_tested = 0.0;

    const int BLOCK = 256;
    const int k = PM_SIZE + dev;

    // Upload matrix once
    CK(cudaMemcpy(gpu.d_matrix, h_mat,
                  static_cast<size_t>(n) * n * sizeof(long long),
                  cudaMemcpyHostToDevice));

    int pool_sz = n - PM_SIZE;

    // Loop over every principal block position s
    for (int s = 0; s <= n - PM_SIZE; s++) {
        int principal[PM_SIZE];
        for (int i = 0; i < PM_SIZE; i++)
            principal[i] = s + i;

        // Build deviation pool: all indices NOT in the principal block
        std::vector<int> pool;
        pool.reserve(pool_sz);
        for (int idx = 0; idx < n; idx++) {
            bool in_pm = false;
            for (int j = 0; j < PM_SIZE; j++) {
                if (idx == principal[j]) { in_pm = true; break; }
            }
            if (!in_pm) pool.push_back(idx);
        }

        // Generate all C(pool_sz, dev) index sets
        std::vector<IndexSet> sets;
        gen_combos(pool, dev, principal, sets);
        int num_sets = static_cast<int>(sets.size());
        if (num_sets == 0) continue;

        if (static_cast<size_t>(num_sets) > gpu.cap_rows) {
            fprintf(stderr,
                    "  [WARN] s=%d dev=%d: %d sets > GPU buffer %zu -- skipping\n",
                    s, dev, num_sets, gpu.cap_rows);
            continue;
        }

        // Upload all row index sets
        CK(cudaMemcpy(gpu.d_row_sets, sets.data(),
                      num_sets * sizeof(IndexSet), cudaMemcpyHostToDevice));

        // Chunk over col-sets
        int chunk_sz = static_cast<int>(gpu.col_chunk);
        bool hit_found = false;

        for (int col_start = 0; col_start < num_sets; col_start += chunk_sz) {
            int col_end = std::min(col_start + chunk_sz, num_sets);
            int num_cols_this_chunk = col_end - col_start;

            long long total_jobs =
                static_cast<long long>(num_sets) * num_cols_this_chunk;
            minors_tested += static_cast<double>(total_jobs);

            // Clear zero flags
            CK(cudaMemset(gpu.d_zero_flags, 0,
                          static_cast<size_t>(num_sets)
                          * num_cols_this_chunk * sizeof(int)));

            // Upload col chunk
            CK(cudaMemcpy(gpu.d_col_sets, sets.data() + col_start,
                          num_cols_this_chunk * sizeof(IndexSet),
                          cudaMemcpyHostToDevice));

            // Launch kernel
            int grid = static_cast<int>((total_jobs + 255LL) / 256LL);
            apm_kernel<<<grid, BLOCK>>>(gpu.d_matrix, n, gpu.d_row_sets,
                                         gpu.d_col_sets, num_sets,
                                         num_cols_this_chunk, k, prime,
                                         gpu.d_zero_flags);
            CK(cudaDeviceSynchronize());
            CK(cudaGetLastError());

            // Copy flags back and scan for the FIRST zero only
            std::vector<int> h_flags(
                static_cast<size_t>(num_sets) * num_cols_this_chunk);
            CK(cudaMemcpy(h_flags.data(), gpu.d_zero_flags,
                          h_flags.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));

            for (long long ji = 0; ji < total_jobs; ji++) {
                if (h_flags[ji]) {
                    int ri = static_cast<int>(ji / num_cols_this_chunk);
                    int ci = static_cast<int>(ji % num_cols_this_chunk)
                           + col_start;

                    ZeroMinor zm;
                    zm.k   = k;
                    zm.dev = dev;
                    zm.s   = s;
                    for (int j = 0; j < k; j++)
                        zm.row_idx[j] = sets[ri].idx[j];
                    for (int j = 0; j < k; j++)
                        zm.col_idx[j] = sets[ci].idx[j];
                    zm.time_ms = FileUtils::now_ms() - matrix_start_ms;
                    found.push_back(zm);

                    // ── FIRST HIT: stop all further searching ──
                    hit_found = true;
                    break;
                }
            }
            if (hit_found) break; // stop chunk loop
        } // col chunk loop

        if (hit_found) break; // stop s loop — matrix is done
    } // s loop

    return found;
}

// =============================================================================
// Process one prime group at one deviation level
// Returns: matrices_hit (number of matrices with at least one zero minor)
// =============================================================================

int ApmSearchEngine::run_group_deviation(int group, long long prime, int dev,
                                          const std::vector<std::string> &files,
                                          int n, GpuBuffers &gpu,
                                          const std::string &base_dir) {
    std::string out_dir = FileUtils::make_out_dir(base_dir, group, dev);
    std::string input_dir = "kernel_output/" + std::to_string(group) + "/";
    std::string prefix = "kernel_" + std::to_string(group) + "_";

    // Header
    printf("\n");
    printf("  +----------------------------------------------------------+\n");
    printf("  |  Group %d  |  Deviation %d  |  %zu matrices\n",
           group, dev, files.size());
    printf("  |  Input : %s  (prefix: %s)\n", input_dir.c_str(), prefix.c_str());
    printf("  |  Output: %s\n", out_dir.c_str());
    printf("  |  Prime : %lld\n", prime);
    printf("  |  Minor : %dx%d\n", PM_SIZE + dev, PM_SIZE + dev);
    printf("  +----------------------------------------------------------+\n");
    fflush(stdout);

    // Open SUMMARY_detailed.txt
    std::string det_path = out_dir + "/SUMMARY_detailed.txt";
    FILE *det_f = fopen(det_path.c_str(), "w");
    if (det_f) {
        fprintf(det_f, "APM Summary\n");
        fprintf(det_f, "Deviation level : %d\n", dev);
        fprintf(det_f, "Prime group     : %d\n", group);
        fprintf(det_f, "Input folder    : %s\n", input_dir.c_str());
        fprintf(det_f, "File prefix     : %s\n", prefix.c_str());
        fprintf(det_f, "Prime           : %lld\n", prime);
        fprintf(det_f, "Matrix size n   : %d\n", n);
        fprintf(det_f, "PM block size   : %d\n", PM_SIZE);
        fprintf(det_f, "Matrices        : %zu\n", files.size());
        fprintf(det_f, "======================================================\n\n");
    }

    // Accumulators
    double folder_start_ms  = FileUtils::now_ms();
    long long total_zero_minors   = 0;
    double total_minors_tested    = 0.0;
    int matrices_hit              = 0;

    // Per-matrix loop
    for (int fi = 0; fi < static_cast<int>(files.size()); fi++) {
        printf("\n  [%d/%zu] %s\n", fi + 1, files.size(), files[fi].c_str());
        fflush(stdout);

        MatrixData md;
        try {
            md = MatrixParser::parse(files[fi]);
        } catch (std::exception &ex) {
            printf("    [ERROR] %s -- skipping\n", ex.what());
            continue;
        }
        if (md.n != n) {
            printf("    [WARN] n=%d != expected %d -- skipping\n", md.n, n);
            continue;
        }

        double matrix_start_ms = FileUtils::now_ms();
        double minors_tested = 0.0;

        printf("    Searching for first zero minor at dev=%d ...\n", dev);
        fflush(stdout);

        std::vector<ZeroMinor> found = search_matrix(
            gpu, md.data.data(), n, prime, dev, minors_tested, matrix_start_ms);

        double matrix_ms = FileUtils::now_ms() - matrix_start_ms;

        total_zero_minors += static_cast<long long>(found.size());
        total_minors_tested += minors_tested;
        if (!found.empty()) matrices_hit++;

        // Terminal summary per matrix
        printf("    Time: %.3f s | Tested: %.0f minors | Zero minors: %zu\n",
               matrix_ms / 1000.0, minors_tested, found.size());

        if (found.empty()) {
            printf("    -> No zero minor at deviation %d\n", dev);
        } else {
            for (int mi = 0; mi < static_cast<int>(found.size()); mi++) {
                const ZeroMinor &zm = found[mi];
                printf("    [Zero #%d] size=%d  dev=%d  s=%d  t=%.2f ms"
                       "  minors_tested=%.0f\n",
                       mi + 1, zm.k, zm.dev, zm.s, zm.time_ms, minors_tested);
                printf("      Rows [%d]: ", zm.k);
                for (int j = 0; j < zm.k; j++)
                    printf("%d%s", zm.row_idx[j], j < zm.k - 1 ? " " : "");
                printf("\n      Cols [%d]: ", zm.k);
                for (int j = 0; j < zm.k; j++)
                    printf("%d%s", zm.col_idx[j], j < zm.k - 1 ? " " : "");
                printf("\n");
            }
        }
        fflush(stdout);

        // Write per-matrix result file
        ResultWriter::write_result_file(out_dir, md, prime, dev, found,
                                         matrix_ms, minors_tested);

        // Append to SUMMARY_detailed.txt
        ResultWriter::append_detailed_entry(det_f, fi,
                                             static_cast<int>(files.size()),
                                             md, matrix_ms, minors_tested, found);
    } // end matrix loop

    double folder_ms = FileUtils::now_ms() - folder_start_ms;

    // Close SUMMARY_detailed.txt with footer
    if (det_f) {
        fprintf(det_f, "======================================================\n");
        fprintf(det_f, "FOLDER TOTALS\n");
        fprintf(det_f, "Total time     : %.4f s\n", folder_ms / 1000.0);
        fprintf(det_f, "Matrices       : %zu\n", files.size());
        fprintf(det_f, "Minors tested  : %.0f\n", total_minors_tested);
        fprintf(det_f, "Zero minors    : %lld\n", total_zero_minors);
        fclose(det_f);
        printf("  Summary (detailed) -> %s\n", det_path.c_str());
    }

    // Write SUMMARY_brief.txt
    ResultWriter::write_summary_brief(out_dir, group, prime, dev, n,
                                       files.size(), matrices_hit,
                                       total_minors_tested, total_zero_minors,
                                       folder_ms);

    // Terminal totals
    printf("\n  -- group=%d  dev=%d complete --\n", group, dev);
    printf("  Time           : %.3f s\n", folder_ms / 1000.0);
    printf("  Matrices       : %zu\n", files.size());
    printf("  Minors tested  : %.0f\n", total_minors_tested);
    printf("  Zero minors    : %lld\n", total_zero_minors);
    printf("  Matrices hit   : %d / %zu\n", matrices_hit, files.size());
    fflush(stdout);

    return matrices_hit;
}

// =============================================================================
// main.cu — APM Brahma: Modular APM First-Hit Search with Early Stop
//
// Usage:
//   ./apm_brahma [gmin gmax]
//
// Examples:
//   ./apm_brahma           (default: groups 25 to 50)
//   ./apm_brahma 32 35     (groups 32 to 35 only)
//   ./apm_brahma 25 27     (groups 25 to 27 only)
//   ./apm_brahma 35 50     (groups 35 to 50)
//
// The program searches for zero-determinant submatrices (minors) mod prime
// in ECDLP kernel matrices stored in kernel_output/<group>/ directories.
//
// For each prime group, it iterates deviation levels 2..n-3.  At each
// deviation, it finds the FIRST zero minor per matrix.  Once 100 matrices
// have been "hit" (any zero minor found), it stops checking further
// deviations for that group (early stop).
//
// Results are written to Results_brahma/<group>/deviation_<d>/
// =============================================================================

#include "apm_search.cuh"
#include "file_utils.hpp"
#include "matrix_parser.hpp"
#include "prime_loader.hpp"
#include "result_writer.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>

static const char *RESULT_BASE_DIR = "Results_brahma";

// =============================================================================
// Print GPU info
// =============================================================================

static void print_gpu_info() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("  [GPU] No CUDA devices found!\n");
        return;
    }
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("  [GPU %d] %s\n", i, prop.name);
        printf("           Compute Capability : %d.%d\n",
               prop.major, prop.minor);
        printf("           Total VRAM         : %.0f MB\n",
               prop.totalGlobalMem / 1e6);
        printf("           SM count           : %d\n",
               prop.multiProcessorCount);
        printf("           Max threads/block   : %d\n",
               prop.maxThreadsPerBlock);
    }
    // Use device 0
    cudaSetDevice(0);
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    printf("  [GPU 0] Free VRAM: %.0f MB / %.0f MB\n\n",
           free_bytes / 1e6, total_bytes / 1e6);
    fflush(stdout);
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char **argv) {
    int gmin = 25, gmax = 50;
    if (argc == 3) {
        gmin = atoi(argv[1]);
        gmax = atoi(argv[2]);
        if (gmin > gmax) {
            int tmp = gmin;
            gmin = gmax;
            gmax = tmp;
        }
    } else if (argc != 1) {
        printf("Usage: %s [gmin gmax]\n", argv[0]);
        printf("  Example: %s 32 35\n", argv[0]);
        printf("  Example: %s 25 27\n", argv[0]);
        printf("  Example: %s 35 50\n", argv[0]);
        return 1;
    }

    // ── Program header ──
    printf("=================================================================\n");
    printf(" APM Brahma — Modular First-Hit Search with Early Stop\n");
    printf(" KERNEL_OUTPUT VERSION — prime groups from kernel_output/<g>/\n");
    printf(" GROUP-OUTER, DEVIATION-INNER LOOP\n");
    printf(" STOPS after the FIRST zero minor per matrix\n");
    printf(" EARLY STOP: skips remaining deviations once %d hits reached\n",
           EARLY_STOP_HIT);
    printf(" PM block size    : %d (fixed)\n", PM_SIZE);
    printf(" Max index static : %d\n", MAX_IDX_STATIC);
    printf(" Supports primes up to 2^50 (safe modular arithmetic)\n");
    printf("=================================================================\n\n");
    printf(" Group range     : %d to %d\n\n", gmin, gmax);
    fflush(stdout);

    // ── GPU info ──
    printf("GPU Information:\n");
    print_gpu_info();

    // ── Read primes from files ──
    printf("Reading primes from files...\n");
    PrimeLoader loader(gmin, gmax);
    loader.load_from_files();
    loader.print_table();

    // ── Check kernel_output directory ──
    struct stat st;
    if (stat("kernel_output", &st) != 0) {
        printf("[FATAL] kernel_output/ directory not found. Nothing to do.\n");
        return 1;
    }

    double prog_start = FileUtils::now_ms();

    // =========================================================================
    // MAIN LOOP — group-outer, deviation-inner
    //
    // For each prime group g = gmin..gmax:
    //   Read prime, collect files, detect n
    //   For each deviation d = 2, 3, ..., n-3:
    //     Process all matrices — track matrices_hit
    //     If matrices_hit >= EARLY_STOP_HIT -> write result.txt, STOP
    //   Write result.txt with best dev
    // =========================================================================

    const std::vector<FolderPrime> &primes = loader.primes();

    for (size_t gi = 0; gi < primes.size(); gi++) {
        int group = primes[gi].folder_id;
        long long prime = primes[gi].prime;

        if (prime <= 0) {
            printf("\n  [SKIP] Group %d — no prime available\n", group);
            continue;
        }

        printf("\n");
        printf("#################################################################\n");
        printf("# GROUP %d   prime = %lld\n", group, prime);
        printf("#################################################################\n");
        fflush(stdout);

        // Collect files
        std::string input_dir = "kernel_output/" + std::to_string(group);
        std::string prefix = "kernel_" + std::to_string(group) + "_";
        std::vector<std::string> files = FileUtils::collect_files(input_dir, prefix);

        if (files.empty()) {
            std::string alt_prefix = std::to_string(group) + "_";
            files = FileUtils::collect_files(input_dir, alt_prefix);
            if (!files.empty()) {
                printf("  [INFO] No files matching prefix '%s'. "
                       "Using prefix '%s' instead.\n",
                       prefix.c_str(), alt_prefix.c_str());
                prefix = alt_prefix;
            }
        }

        if (files.empty()) {
            printf("  [SKIP] No matching files in %s\n", input_dir.c_str());
            continue;
        }

        // Parse first matrix to detect n
        MatrixData first;
        try {
            first = MatrixParser::parse(files[0]);
        } catch (std::exception &ex) {
            printf("  [ERROR] Cannot parse first file: %s\n", ex.what());
            continue;
        }
        int n = first.n;

        int max_dev = n - (PM_SIZE + 1); // n - 3
        printf("  Matrix size n   = %d\n", n);
        printf("  Files found     = %zu\n", files.size());
        printf("  Deviation range = %d to %d\n", MIN_DEV, max_dev);
        fflush(stdout);

        if (max_dev < MIN_DEV) {
            printf("  [SKIP] max_dev=%d < MIN_DEV=%d -- matrix too small\n",
                   max_dev, MIN_DEV);
            continue;
        }

        double group_start = FileUtils::now_ms();

        // ── Track best deviation across the loop ──
        int best_dev  = -1;
        int best_hits = 0;
        bool reached_target = false;

        // Deviation-inner loop
        for (int dev = MIN_DEV; dev <= max_dev; dev++) {
            // Check feasibility
            if (dev > n - PM_SIZE) {
                printf("  [SKIP] dev=%d needs pool >= %d but pool=%d\n",
                       dev, dev, n - PM_SIZE);
                continue;
            }

            printf("\n");
            printf("  ===============================================================\n");
            printf("  [group=%d | dev=%d/%d]  minor=%dx%d\n",
                   group, dev, max_dev, PM_SIZE + dev, PM_SIZE + dev);
            printf("  ===============================================================\n");
            fflush(stdout);

            // Allocate GPU buffers for this (n, dev) combination
            GpuBuffers gpu;
            gpu.alloc(n, dev);

            int hits = ApmSearchEngine::run_group_deviation(
                group, prime, dev, files, n, gpu, RESULT_BASE_DIR);

            // gpu freed by destructor
            printf("  [group=%d  dev=%d done]  hits=%d/%zu\n",
                   group, dev, hits, files.size());
            fflush(stdout);

            // Track best deviation
            if (hits > best_hits) {
                best_hits = hits;
                best_dev  = dev;
            }

            // ── EARLY STOP: all matrices hit → write result, skip rest ──
            if (hits >= EARLY_STOP_HIT) {
                printf("\n  *** %d HITS REACHED at deviation %d! ***\n",
                       EARLY_STOP_HIT, dev);
                printf("  *** Skipping remaining deviations for group %d ***\n",
                       group);
                fflush(stdout);
                reached_target = true;
                break;
            }
        }

        // Write result.txt for this group
        if (best_dev >= 0) {
            ResultWriter::write_group_result(RESULT_BASE_DIR, group,
                                              best_dev, best_hits,
                                              reached_target);
        }

        double group_ms = FileUtils::now_ms() - group_start;
        printf("\n");
        printf("#################################################################\n");
        printf("# GROUP %d COMPLETE   (%.2f min)\n", group, group_ms / 60000.0);
        if (reached_target)
            printf("# EARLY STOP at deviation %d (%d hits)\n",
                   best_dev, best_hits);
        else
            printf("# Best deviation: %d  (%d hits)\n", best_dev, best_hits);
        printf("#################################################################\n");
        fflush(stdout);
    }

    double total_ms = FileUtils::now_ms() - prog_start;
    printf("\n=================================================================\n");
    printf(" ALL DONE\n");
    printf(" Total wall time : %.3f s  (%.2f min)\n",
           total_ms / 1000.0, total_ms / 60000.0);
    printf("\n Results in: %s/<group>/\n", RESULT_BASE_DIR);
    printf(" Each group has a result.txt with the best deviation.\n");
    printf("=================================================================\n");

    return 0;
}

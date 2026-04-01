// =============================================================================
// result_writer.cpp — Write result files and summaries
// =============================================================================

#include "result_writer.hpp"
#include "file_utils.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

// =============================================================================
// Helper: compute display width for a long long value
// =============================================================================
static int digit_width(long long v) {
    int w = 1;
    if (v < 0) { v = -v; w = 2; }
    while (v >= 10) { v /= 10; w++; }
    return w;
}

// =============================================================================
// Write the principal block + submatrix detail for one zero minor
// =============================================================================

void ResultWriter::write_zero_minor_detail(FILE *f, const ZeroMinor &zm,
                                            const MatrixData &md) {
    // Column width from full matrix for consistent alignment
    int col_w = 1;
    for (int i = 0; i < md.n * md.n; i++) {
        int w = digit_width(md.data[i]);
        if (w > col_w) col_w = w;
    }

    // Principal 2x2 block
    fprintf(f, "  Principal 2x2 block (s=%d, rows/cols {%d,%d}):\n",
            zm.s, zm.s, zm.s + 1);
    fprintf(f, "    [ %*lld  %*lld ]\n", col_w,
            md.data[zm.s * md.n + zm.s], col_w,
            md.data[zm.s * md.n + zm.s + 1]);
    fprintf(f, "    [ %*lld  %*lld ]\n", col_w,
            md.data[(zm.s + 1) * md.n + zm.s], col_w,
            md.data[(zm.s + 1) * md.n + zm.s + 1]);

    // Submatrix column width (tighter)
    int sw = 1;
    for (int r = 0; r < zm.k; r++)
        for (int c = 0; c < zm.k; c++) {
            int w = digit_width(md.data[zm.row_idx[r] * md.n + zm.col_idx[c]]);
            if (w > sw) sw = w;
        }

    fprintf(f, "\n  Extracted %dx%d submatrix  (det mod p = 0):\n", zm.k, zm.k);

    // Column index header — mark principal block cols with '*'
    fprintf(f, "       ");
    for (int c = 0; c < zm.k; c++) {
        bool is_pm = (zm.col_idx[c] == zm.s || zm.col_idx[c] == zm.s + 1);
        fprintf(f, " %*d%c", sw, zm.col_idx[c], is_pm ? '*' : ' ');
    }
    fprintf(f, "\n       ");
    for (int c = 0; c < zm.k; c++) {
        fprintf(f, " ");
        for (int w = 0; w <= sw; w++) fprintf(f, "-");
    }
    fprintf(f, "\n");

    // Rows — mark principal block rows with '*', bracket the 2x2 cells
    for (int r = 0; r < zm.k; r++) {
        bool pm_r = (zm.row_idx[r] == zm.s || zm.row_idx[r] == zm.s + 1);
        fprintf(f, "  %3d%c |", zm.row_idx[r], pm_r ? '*' : ' ');
        for (int c = 0; c < zm.k; c++) {
            long long val = md.data[zm.row_idx[r] * md.n + zm.col_idx[c]];
            bool pm_c = (zm.col_idx[c] == zm.s || zm.col_idx[c] == zm.s + 1);
            if (pm_r && pm_c)
                fprintf(f, "[%*lld]", sw, val);
            else
                fprintf(f, " %*lld ", sw, val);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "\n");
}

// =============================================================================
// Write per-matrix result file
// =============================================================================

void ResultWriter::write_result_file(const std::string &out_dir,
                                      const MatrixData &md,
                                      long long prime, int dev,
                                      const std::vector<ZeroMinor> &minors,
                                      double matrix_ms, double minors_tested) {
    std::string base = md.filename;
    size_t dot = base.rfind('.');
    if (dot != std::string::npos) base = base.substr(0, dot);
    std::string outpath = out_dir + "/" + base + "_result.txt";

    FILE *f = fopen(outpath.c_str(), "w");
    if (!f) {
        fprintf(stderr, "  [WARN] Cannot write: %s\n", outpath.c_str());
        return;
    }

    double avg_ms = (minors_tested > 0.0) ? matrix_ms / minors_tested : 0.0;

    fprintf(f, "============================================================\n");
    fprintf(f, "APM Result\n");
    fprintf(f, "============================================================\n");
    fprintf(f, "Matrix file       : %s\n", md.filename.c_str());
    fprintf(f, "Matrix size       : %d x %d\n", md.n, md.n);
    fprintf(f, "Prime (mod p)     : %lld\n", prime);
    fprintf(f, "PM block size     : %d (fixed)\n", PM_SIZE);
    fprintf(f, "Deviation level   : %d\n", dev);
    fprintf(f, "Minor size tested : %d x %d\n", PM_SIZE + dev, PM_SIZE + dev);
    fprintf(f, "------------------------------------------------------------\n");
    fprintf(f, "Timing\n");
    fprintf(f, "------------------------------------------------------------\n");
    fprintf(f, "Matrix total time : %.4f ms  (%.6f s)\n",
            matrix_ms, matrix_ms / 1000.0);
    fprintf(f, "Minors tested     : %.0f\n", minors_tested);
    fprintf(f, "Avg per minor     : %.8f ms\n", avg_ms);
    fprintf(f, "------------------------------------------------------------\n");
    fprintf(f, "Zero Minors Found : %zu\n", minors.size());
    fprintf(f, "============================================================\n\n");

    // Column width for full matrix
    int col_w = 1;
    for (int i = 0; i < md.n * md.n; i++) {
        int w = digit_width(md.data[i]);
        if (w > col_w) col_w = w;
    }

    // Write the full source matrix
    fprintf(f, "Full matrix (%dx%d) mod %lld:\n", md.n, md.n, prime);
    fprintf(f, "     ");
    for (int c = 0; c < md.n; c++)
        fprintf(f, " %*d", col_w, c);
    fprintf(f, "\n     ");
    for (int c = 0; c < md.n; c++) {
        fprintf(f, " ");
        for (int w = 0; w < col_w; w++) fprintf(f, "-");
    }
    fprintf(f, "\n");
    for (int r = 0; r < md.n; r++) {
        fprintf(f, "%3d |", r);
        for (int c = 0; c < md.n; c++)
            fprintf(f, " %*lld", col_w, md.data[r * md.n + c]);
        fprintf(f, "\n");
    }
    fprintf(f, "\n");

    if (minors.empty()) {
        fprintf(f, "No zero minor found at deviation level %d.\n", dev);
    } else {
        for (int mi = 0; mi < static_cast<int>(minors.size()); mi++) {
            const ZeroMinor &zm = minors[mi];
            fprintf(f, "--- Zero Minor #%d ---\n", mi + 1);
            fprintf(f, "  Minor size (k)   : %d\n", zm.k);
            fprintf(f, "  Deviation        : %d\n", zm.dev);
            fprintf(f, "  Principal block s: %d  (indices {%d, %d})\n",
                    zm.s, zm.s, zm.s + 1);
            fprintf(f, "  Row indices [%d]  : ", zm.k);
            for (int j = 0; j < zm.k; j++)
                fprintf(f, "%d%s", zm.row_idx[j], j < zm.k - 1 ? " " : "");
            fprintf(f, "\n");
            fprintf(f, "  Col indices [%d]  : ", zm.k);
            for (int j = 0; j < zm.k; j++)
                fprintf(f, "%d%s", zm.col_idx[j], j < zm.k - 1 ? " " : "");
            fprintf(f, "\n");
            fprintf(f, "  Time found       : %.4f ms  (since matrix start)\n",
                    zm.time_ms);
            fprintf(f, "  Minors tested    : %.0f  (tested before this zero)\n",
                    minors_tested);
            fprintf(f, "\n");
            write_zero_minor_detail(f, zm, md);
        }
    }

    fclose(f);
    printf("      -> %s\n", outpath.c_str());
    fflush(stdout);
}

// =============================================================================
// Write group-level result.txt (early-stop tracking)
// =============================================================================

void ResultWriter::write_group_result(const std::string &base_dir, int group,
                                       int best_dev, int best_hits,
                                       bool reached_100) {
    FileUtils::mkdir_safe(base_dir);
    std::string dir = base_dir + "/" + std::to_string(group);
    FileUtils::mkdir_safe(dir);
    std::string path = dir + "/result.txt";

    FILE *f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "[WARN] Cannot write: %s\n", path.c_str());
        return;
    }

    fprintf(f, "============================================================\n");
    fprintf(f, "APM Early-Stop Result  (check_till_100)\n");
    fprintf(f, "============================================================\n");
    fprintf(f, "Prime group      : %d\n", group);
    if (reached_100) {
        fprintf(f, "Best deviation   : %d\n", best_dev);
        fprintf(f, "Matrices hit     : %d  (reached %d -- EARLY STOP)\n",
                best_hits, EARLY_STOP_HIT);
        fprintf(f, "Status           : All matrices hit at deviation %d.\n",
                best_dev);
        fprintf(f, "                   No further deviations were checked.\n");
    } else {
        fprintf(f, "Best deviation   : %d\n", best_dev);
        fprintf(f, "Matrices hit     : %d  (did NOT reach %d)\n",
                best_hits, EARLY_STOP_HIT);
        fprintf(f, "Status           : All deviations checked.\n");
        fprintf(f, "                   Deviation %d had the most hits (%d).\n",
                best_dev, best_hits);
    }
    fprintf(f, "============================================================\n");
    fclose(f);

    printf("  -> result.txt written: %s\n", path.c_str());
    fflush(stdout);
}

// =============================================================================
// Append a zero-minor entry to SUMMARY_detailed.txt
// =============================================================================

void ResultWriter::append_detailed_entry(FILE *det_f, int fi, int total_files,
                                          const MatrixData &md,
                                          double matrix_ms,
                                          double minors_tested,
                                          const std::vector<ZeroMinor> &found) {
    if (!det_f || found.empty()) return;

    fprintf(det_f, "------------------------------------------------------------\n");
    fprintf(det_f, "[%d/%d] %s\n", fi + 1, total_files, md.filename.c_str());
    fprintf(det_f, "  time=%.4f s | tested=%.0f | zeros=%zu\n",
            matrix_ms / 1000.0, minors_tested, found.size());

    for (int mi = 0; mi < static_cast<int>(found.size()); mi++) {
        const ZeroMinor &zm = found[mi];
        fprintf(det_f, "\n  --- Zero Minor #%d ---\n", mi + 1);
        fprintf(det_f, "  Minor size (k)   : %d\n", zm.k);
        fprintf(det_f, "  Deviation        : %d\n", zm.dev);
        fprintf(det_f, "  Principal block s: %d  (indices {%d, %d})\n",
                zm.s, zm.s, zm.s + 1);
        fprintf(det_f, "  Row indices [%d]  : ", zm.k);
        for (int j = 0; j < zm.k; j++)
            fprintf(det_f, "%d%s", zm.row_idx[j], j < zm.k - 1 ? " " : "");
        fprintf(det_f, "\n");
        fprintf(det_f, "  Col indices [%d]  : ", zm.k);
        for (int j = 0; j < zm.k; j++)
            fprintf(det_f, "%d%s", zm.col_idx[j], j < zm.k - 1 ? " " : "");
        fprintf(det_f, "\n");
        fprintf(det_f, "  Time found       : %.4f ms\n", zm.time_ms);
        fprintf(det_f, "  Minors tested    : %.0f  (tested before finding this zero)\n\n",
                minors_tested);
        write_zero_minor_detail(det_f, zm, md);
    }
}

// =============================================================================
// Write SUMMARY_brief.txt
// =============================================================================

void ResultWriter::write_summary_brief(const std::string &out_dir, int group,
                                        long long prime, int dev, int n,
                                        size_t total_files, int matrices_hit,
                                        double total_minors_tested,
                                        long long total_zero_minors,
                                        double folder_ms) {
    std::string brief_path = out_dir + "/SUMMARY_brief.txt";
    FILE *brief_f = fopen(brief_path.c_str(), "w");
    if (!brief_f) return;

    double hit_pct = (total_files > 0)
        ? 100.0 * matrices_hit / static_cast<double>(total_files) : 0.0;

    fprintf(brief_f, "============================================================\n");
    fprintf(brief_f, "APM Brief Summary\n");
    fprintf(brief_f, "============================================================\n");
    fprintf(brief_f, "Prime group      : %d\n", group);
    fprintf(brief_f, "Prime (p)        : %lld\n", prime);
    fprintf(brief_f, "Deviation level  : %d\n", dev);
    fprintf(brief_f, "Minor size       : %d x %d\n", PM_SIZE + dev, PM_SIZE + dev);
    fprintf(brief_f, "PM block size    : %d\n", PM_SIZE);
    fprintf(brief_f, "Matrix size (n)  : %d\n", n);
    fprintf(brief_f, "Input folder     : kernel_output/%d/\n", group);
    fprintf(brief_f, "Output folder    : %s\n", out_dir.c_str());
    fprintf(brief_f, "------------------------------------------------------------\n");
    fprintf(brief_f, "Total matrices   : %zu\n", total_files);
    fprintf(brief_f, "Matrices hit     : %d      "
                     "(contain at least one zero minor)\n", matrices_hit);
    fprintf(brief_f, "Hit ratio        : %d/%zu = %.2f%%\n",
            matrices_hit, total_files, hit_pct);
    fprintf(brief_f, "Total minors tested  : %.0f\n", total_minors_tested);
    fprintf(brief_f, "Total zero minors    : %lld\n", total_zero_minors);
    fprintf(brief_f, "Total time           : %.3f s\n", folder_ms / 1000.0);
    fprintf(brief_f, "Avg time per matrix  : %.3f s\n",
            (total_files > 0) ? folder_ms / 1000.0 / static_cast<double>(total_files) : 0.0);
    fprintf(brief_f, "============================================================\n");
    fclose(brief_f);
    printf("  Summary (brief)    -> %s\n", brief_path.c_str());
}

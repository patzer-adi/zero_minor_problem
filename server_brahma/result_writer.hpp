// =============================================================================
// result_writer.hpp — Write result files and summaries
// =============================================================================

#ifndef RESULT_WRITER_HPP
#define RESULT_WRITER_HPP

#include "apm_types.hpp"
#include <cstdio>
#include <string>
#include <vector>

class ResultWriter {
public:
    // Write per-matrix result file into out_dir.
    static void write_result_file(const string &out_dir,
                                   const MatrixData &md,
                                   long long prime, int dev,
                                   const vector<ZeroMinor> &minors,
                                   double matrix_ms, double minors_tested);

    // Write the principal block + submatrix detail for one zero minor.
    static void write_zero_minor_detail(FILE *f, const ZeroMinor &zm,
                                         const MatrixData &md);

    // Write group-level result.txt (early-stop tracking).
    static void write_group_result(const string &base_dir, int group,
                                    int best_dev, int best_hits, bool reached_100);

    // Append a zero-minor entry to the SUMMARY_detailed.txt file handle.
    static void append_detailed_entry(FILE *det_f, int fi, int total_files,
                                       const MatrixData &md, double matrix_ms,
                                       double minors_tested,
                                       const vector<ZeroMinor> &found);

    // Write SUMMARY_brief.txt for one group/deviation.
    static void write_summary_brief(const string &out_dir, int group,
                                     long long prime, int dev, int n,
                                     size_t total_files, int matrices_hit,
                                     double total_minors_tested,
                                     long long total_zero_minors,
                                     double folder_ms);
};

#endif // RESULT_WRITER_HPP

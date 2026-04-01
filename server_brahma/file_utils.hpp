// =============================================================================
// file_utils.hpp — File and directory utility functions
// =============================================================================

#ifndef FILE_UTILS_HPP
#define FILE_UTILS_HPP

#include <string>
#include <vector>

class FileUtils {
public:
    // Create a single directory (no error if it already exists)
    static void mkdir_safe(const std::string &path);

    // Check if string ends/starts with a given substring
    static bool ends_with(const std::string &str, const std::string &suffix);
    static bool starts_with(const std::string &str, const std::string &prefix);

    // Collect all .txt files in dir_path matching prefix, sorted.
    // Skips files ending in _RN.txt (row-number metadata).
    static std::vector<std::string> collect_files(const std::string &dir_path,
                                                   const std::string &prefix);

    // Create output directory: <base_dir>/<group>/deviation_<dev>/
    // Returns the full path to the deepest directory.
    static std::string make_out_dir(const std::string &base_dir, int group, int dev);

    // High-resolution timer (milliseconds since epoch)
    static double now_ms();

    // Binomial coefficient C(n, r). Returns 0 on overflow or invalid input.
    static long long nCr(int n, int r);
};

#endif // FILE_UTILS_HPP

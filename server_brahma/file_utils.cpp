// =============================================================================
// file_utils.cpp — File and directory utility functions
// =============================================================================

#include "file_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>
using namespace std;

// -----------------------------------------------------------------------------
void FileUtils::mkdir_safe(const std::string &path) {
    mkdir(path.c_str(), 0755);
}

// -----------------------------------------------------------------------------
bool FileUtils::ends_with(const string &str, const string &suffix) {
    if (suffix.size() > str.size()) return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// -----------------------------------------------------------------------------
bool FileUtils::starts_with(const string &str, const string &prefix) {
    if (prefix.size() > str.size()) return false;
    return str.compare(0, prefix.size(), prefix) == 0;
}

// -----------------------------------------------------------------------------
vector<string> FileUtils::collect_files(const string &dir_path,
                                                   const string &prefix) {
    vector<string> files;
    DIR *d = opendir(dir_path.c_str());
    if (!d) return files;

    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        string nm = e->d_name;
        if (!starts_with(nm, prefix))      continue;
        if (nm.size() <= 4)                continue;
        if (nm.substr(nm.size() - 4) != ".txt") continue;
        if (ends_with(nm, "_RN.txt"))      continue;
        files.push_back(dir_path + "/" + nm);
    }
    closedir(d);
    sort(files.begin(), files.end());
    return files;
}

// -----------------------------------------------------------------------------
string FileUtils::make_out_dir(const string &base_dir, int group, int dev) {
    mkdir_safe(base_dir);
    string d1 = base_dir + "/" + to_string(group);
    mkdir_safe(d1);
    string d2 = d1 + "/deviation_" + to_string(dev);
    mkdir_safe(d2);
    return d2;
}

// -----------------------------------------------------------------------------
double FileUtils::now_ms() {
    using namespace chrono;
    return static_cast<double>(
        duration_cast<microseconds>(
            high_resolution_clock::now().time_since_epoch()
        ).count()
    ) / 1000.0;
}

// -----------------------------------------------------------------------------
long long FileUtils::nCr(int n, int r) {
    if (r < 0 || r > n) return 0;
    if (r == 0 || r == n) return 1;
    if (r > n - r) r = n - r; // symmetry
    long long result = 1;
    for (int i = 0; i < r; i++) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

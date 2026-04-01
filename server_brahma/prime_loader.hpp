// =============================================================================
// prime_loader.hpp — Load primes from input files with hardcoded fallbacks
// =============================================================================

#ifndef PRIME_LOADER_HPP
#define PRIME_LOADER_HPP

#include "apm_types.hpp"
#include <vector>

class PrimeLoader {
public:
    // Initialize the prime table for groups gmin..gmax.
    // Groups 25-35 have hardcoded fallback primes.
    // Groups 36-50 rely entirely on file reading.
    PrimeLoader(int gmin = 25, int gmax = 50);

    // Load primes from files, overriding defaults where possible.
    void load_from_files();

    // Print the prime table to stdout.
    void print_table() const;

    // Get the list of all folder primes.
    const vector<FolderPrime>& primes() const { return primes_; }

    // Find a prime by group ID. Returns -1 if not found.
    long long get_prime(int group) const;

private:
    vector<FolderPrime> primes_;

    // Read a prime value from a single file.
    // Returns -1 if file cannot be read or prime not found.
    static long long read_prime_from_file(const string &path);
};

#endif // PRIME_LOADER_HPP

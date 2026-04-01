// =============================================================================
// prime_loader.cpp — Load primes from input files with hardcoded fallbacks
// =============================================================================

#include "prime_loader.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

// =============================================================================
// Hardcoded fallback primes for groups 25-35
// =============================================================================

static const struct {
  int id;
  long long prime;
} HARDCODED_PRIMES[] = {
    {25, 33554393LL},    {26, 44923183LL},    {27, 134217689LL},
    {28, 268435399LL},   {29, 536870909LL},   {30, 1073741789LL},
    {31, 2147483647LL},  {32, 4294967291LL},  {33, 8589934583LL},
    {34, 17179869143LL}, {35, 34359738337LL},
};
static const int NUM_HARDCODED = 11;

// =============================================================================
// Constructor: build the prime table for groups gmin..gmax
// =============================================================================

PrimeLoader::PrimeLoader(int gmin, int gmax) {
  for (int g = gmin; g <= gmax; g++) {
    FolderPrime fp;
    fp.folder_id = g;
    fp.prime = -1; // unknown until loaded

    // Check hardcoded fallback
    for (int i = 0; i < NUM_HARDCODED; i++) {
      if (HARDCODED_PRIMES[i].id == g) {
        fp.prime = HARDCODED_PRIMES[i].prime;
        break;
      }
    }

    snprintf(fp.label, sizeof(fp.label), "%d (p=%lld)", g, fp.prime);
    primes_.push_back(fp);
  }
}

// =============================================================================
// Read a prime from one file
// Format: lines of tab-separated integers, then a line with "#".
// The prime is the first line that has exactly ONE integer token.
// =============================================================================

long long PrimeLoader::read_prime_from_file(const string &path) {
  ifstream ifs(path.c_str());
  if (!ifs.is_open())
    return -1;

  string line;
  while (getline(ifs, line)) {
    // Trim leading whitespace
    size_t fs = line.find_first_not_of(" \t\r\n");
    if (fs == string::npos)
      continue;
    line = line.substr(fs);

    // Stop at "#"
    if (line[0] == '#')
      break;

    // Tokenize by whitespace
    istringstream ss(line);
    vector<string> tokens;
    string tok;
    while (ss >> tok)
      tokens.push_back(tok);

    // Single-token line = the prime
    if (tokens.size() == 1) {
      long long val = 0;
      try {
        val = stoll(tokens[0]);
      } catch (...) {
        continue;
      }
      return val;
    }
  }
  return -1;
}

// =============================================================================
// Load primes from files, overriding defaults
// =============================================================================

void PrimeLoader::load_from_files() {
  for (size_t i = 0; i < primes_.size(); i++) {
    int fid = primes_[i].folder_id;
    string path;
    long long p = -1;

    // Groups 25-29: try 25_29/<N>_1.txt and 25_29/<N>/<N>_1.txt
    if (fid >= 25 && fid <= 29) {
      path = "25_29/" + to_string(fid) + "_1.txt";
      p = read_prime_from_file(path);
      if (p <= 0) {
        path = "25_29/" + to_string(fid) + "/" + to_string(fid) +
               "_1.txt";
        p = read_prime_from_file(path);
      }
    }

    // Groups 30+: try exp/<N>_1.txt
    if (p <= 0 && fid >= 30) {
      path = "exp/" + to_string(fid) + "_1.txt";
      p = read_prime_from_file(path);
    }

    if (p > 0) {
      primes_[i].prime = p;
      snprintf(primes_[i].label, sizeof(primes_[i].label), "%d (p=%lld)", fid,
               p);
      printf("  [PRIME] %d -> %lld  (from %s)\n", fid, p, path.c_str());
    } else if (primes_[i].prime > 0) {
      printf("  [PRIME] %d -> %lld  (HARDCODED FALLBACK)\n", fid,
             primes_[i].prime);
    } else {
      printf("  [PRIME] %d -> NOT AVAILABLE (no file, no fallback)\n", fid);
    }
  }
}

// =============================================================================
// Print the prime table
// =============================================================================

void PrimeLoader::print_table() const {
  printf("\nPrime table:\n");
  for (size_t i = 0; i < primes_.size(); i++) {
    if (primes_[i].prime > 0) {
      printf("  %2d -> %15lld  %s\n", primes_[i].folder_id, primes_[i].prime,
             primes_[i].label);
    } else {
      printf("  %2d -> %15s  (not available)\n", primes_[i].folder_id, "N/A");
    }
  }
  printf("\n");
  fflush(stdout);
}

// =============================================================================
// Get prime by group ID
// =============================================================================

long long PrimeLoader::get_prime(int group) const {
  for (size_t i = 0; i < primes_.size(); i++) {
    if (primes_[i].folder_id == group)
      return primes_[i].prime;
  }
  return -1;
}

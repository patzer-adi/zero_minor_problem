// =============================================================================
// mod_arith.cuh — Device-side modular arithmetic for APM search
//
// ALL operations use long long.
// mod_mul uses a 25-bit split with unsigned long long intermediates
// to safely handle primes up to 2^50 without overflow.
// =============================================================================

#ifndef MOD_ARITH_CUH
#define MOD_ARITH_CUH

#include "apm_types.hpp"

// =============================================================================
// Modular subtraction: (a - b) mod p, result in [0, p-1]
// =============================================================================

__device__ inline long long mod_sub(long long a, long long b, long long p) {
    long long r = a - b;
    return (r < 0) ? r + p : r;
}

// =============================================================================
// Modular multiplication: (a * b) mod p
//
// Safe for p up to 2^50.
// Strategy: split a into high 25 bits and low 25 bits.
//   a = a_hi * 2^25 + a_lo
//   a * b = a_hi * b * 2^25 + a_lo * b
// Each partial product fits in unsigned 64-bit:
//   a_hi < 2^25, b < 2^50  =>  a_hi * b < 2^75  (too large for 64-bit)
//
// So we use modular reduction after each partial multiply.
// Since a_hi < 2^25 and b < p < 2^50, and a_hi * (b % p) < 2^25 * 2^50 = 2^75,
// we need to be careful. We reduce b first, then use that:
//   a_hi < 2^25, b_mod < p < 2^50 => a_hi * b_mod < 2^75 — overflows!
//
// Solution: use two-level split.
//   Split a into 3 parts of ~17 bits each, or use __int128.
//   CUDA does not support __int128 on device, so we use a wider split strategy.
//
// Actual safe approach: Since both a,b < p < 2^50, and we need a*b mod p,
// we split a into TWO 25-bit halves:
//   a_hi (top 25 bits, up to 2^25-1), a_lo (bottom 25 bits, up to 2^25-1)
//   a_hi * b < 2^25 * 2^50 = 2^75 — too large for 64 bits.
//
// So we need a different approach. We use the classic doubling approach or
// split both operands. The safest GPU-compatible approach:
// Use unsigned long long with step-by-step modular reduction.
//
// We break the multiplication into smaller safe pieces using a shift-and-add
// modular multiply (Russian peasant / binary method).
// =============================================================================

__device__ inline long long mod_mul(long long a, long long b, long long p) {
    // Normalize to [0, p-1]
    a %= p;
    if (a < 0) a += p;
    b %= p;
    if (b < 0) b += p;

    // For small primes (< 2^31), direct multiplication is safe:
    // a * b < 2^62, fits in signed long long.
    if (p < (1LL << 31)) {
        return (a * b) % p;
    }

    // For larger primes, use binary (Russian peasant) modular multiplication.
    // This is O(log b) multiplications, each staying within 64-bit range.
    unsigned long long ua = static_cast<unsigned long long>(a);
    unsigned long long ub = static_cast<unsigned long long>(b);
    unsigned long long up = static_cast<unsigned long long>(p);
    unsigned long long result = 0;

    ua %= up;
    while (ub > 0) {
        if (ub & 1ULL) {
            result = (result + ua) % up;
        }
        ua = (ua * 2ULL) % up;
        ub >>= 1;
    }
    return static_cast<long long>(result);
}

// =============================================================================
// Modular inverse via Extended Euclidean algorithm — all long long
// =============================================================================

__device__ inline long long mod_inv(long long a, long long p) {
    long long t = 0, nt = 1, r = p, nr = a % p;
    if (nr < 0) nr += p;
    while (nr) {
        long long q = r / nr, tmp;
        tmp = t;  t = nt;  nt = tmp - q * nt;
        tmp = r;  r = nr;  nr = tmp - q * nr;
    }
    return (t < 0) ? t + p : t;
}

// =============================================================================
// Determinant mod p via Gaussian elimination — all long long
// sub is flat k×k row-major, values already in [0, p-1]
// =============================================================================

__device__ inline long long det_mod(const long long *sub, int k, long long p) {
    long long a[MAX_IDX_STATIC][MAX_IDX_STATIC];
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
            a[i][j] = sub[i * k + j];

    long long det = 1LL;
    for (int col = 0; col < k; col++) {
        // Find pivot
        int piv = -1;
        for (int row = col; row < k; row++) {
            if (a[row][col]) {
                piv = row;
                break;
            }
        }
        if (piv < 0) return 0LL;

        // Swap rows
        if (piv != col) {
            for (int j = col; j < k; j++) {
                long long tmp = a[col][j];
                a[col][j] = a[piv][j];
                a[piv][j] = tmp;
            }
            det = (p - det) % p; // row swap negates determinant
        }

        det = mod_mul(det, a[col][col], p);
        long long inv = mod_inv(a[col][col], p);

        for (int row = col + 1; row < k; row++) {
            if (!a[row][col]) continue;
            long long f = mod_mul(a[row][col], inv, p);
            for (int j = col + 1; j < k; j++)
                a[row][j] = mod_sub(a[row][j], mod_mul(f, a[col][j], p), p);
            a[row][col] = 0LL;
        }
    }
    return det;
}

#endif // MOD_ARITH_CUH

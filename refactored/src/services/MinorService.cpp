#include "services/MinorService.hpp"
#include <stdexcept>

using namespace refactored::services;
using MatrixT = MinorService::MatrixT;
using Value = MinorService::Value;

// Helper: extract kxk submatrix beginning at (r0,c0)
static std::vector<std::vector<Value>> extractSub(const MatrixT &m, std::size_t r0, std::size_t c0, std::size_t k) {
    std::vector<std::vector<Value>> out;
    out.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        std::vector<Value> row;
        row.reserve(k);
        for (std::size_t j = 0; j < k; ++j) {
            row.push_back(m[r0 + i][c0 + j]);
        }
        out.push_back(std::move(row));
    }
    return out;
}

std::vector<Value> MinorService::computeKxKMinors(const MatrixT &mat, std::size_t k) const {
    std::vector<Value> results;
    if (k == 0) return results;
    if (mat.rows() < k || mat.cols() < k) return results;

    for (std::size_t r = 0; r + k <= mat.rows(); ++r) {
        for (std::size_t c = 0; c + k <= mat.cols(); ++c) {
            MatrixT sub(extractSub(mat, r, c, k));
            auto detOpt = sub.determinant();
            results.push_back(detOpt.value_or(0));
        }
    }
    return results;
}

bool MinorService::hasZeroKxKMinor(const MatrixT &mat, std::size_t k) const {
    if (k == 0) return false;
    if (mat.rows() < k || mat.cols() < k) return false;
    for (std::size_t r = 0; r + k <= mat.rows(); ++r) {
        for (std::size_t c = 0; c + k <= mat.cols(); ++c) {
            MatrixT sub(extractSub(mat, r, c, k));
            auto detOpt = sub.determinant();
            if (detOpt && *detOpt == 0) return true;
        }
    }
    return false;
}

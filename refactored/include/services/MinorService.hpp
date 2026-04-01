#pragma once

#include "core/Matrix.hpp"
#include <vector>
#include <optional>

namespace refactored::services {

// Service responsible for computing minors and checking singularity conditions.
class MinorService {
public:
    using MatrixT = refactored::core::Matrix;
    using Value = MatrixT::Value;

    MinorService() = default;

    // Return all kxk minors' determinants for a given matrix.
    std::vector<Value> computeKxKMinors(const MatrixT &mat, std::size_t k) const;

    // Check if any kxk minor is zero.
    bool hasZeroKxKMinor(const MatrixT &mat, std::size_t k) const;
};

} // namespace

#pragma once

#include <vector>
#include <cstddef>
#include <optional>

namespace refactored::core {

// Lightweight, strongly-typed Matrix model.
// Owns its data and provides safe accessors.
class Matrix {
public:
    using Value = long long;

    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols);
    Matrix(const std::vector<std::vector<Value>> &data);

    std::size_t rows() const noexcept;
    std::size_t cols() const noexcept;
    const std::vector<Value>& operator[](std::size_t r) const;
    std::vector<Value>& operator[](std::size_t r);

    // Return determinant if square, otherwise nullopt.
    std::optional<Value> determinant() const;

private:
    std::vector<std::vector<Value>> data_;

    // helper used internally
    Value determinantSquare(std::vector<std::vector<Value>> mat) const;
};

} // namespace

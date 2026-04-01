#include "core/Matrix.hpp"
#include <stdexcept>

using namespace refactored::core;

Matrix::Matrix(std::size_t rows, std::size_t cols) : data_(rows, std::vector<Value>(cols, 0)) {}
Matrix::Matrix(const std::vector<std::vector<Value>> &data) : data_(data) {}

std::size_t Matrix::rows() const noexcept { return data_.size(); }
std::size_t Matrix::cols() const noexcept { return data_.empty() ? 0 : data_[0].size(); }

const std::vector<Matrix::Value>& Matrix::operator[](std::size_t r) const { return data_.at(r); }
std::vector<Matrix::Value>& Matrix::operator[](std::size_t r) { return data_.at(r); }

Matrix::Value Matrix::determinantSquare(std::vector<std::vector<Value>> mat) const {
    const std::size_t n = mat.size();
    if (n == 0) return 1;
    if (n == 1) return mat[0][0];
    if (n == 2) return mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0];

    Value det = 0;
    for (std::size_t p = 0; p < n; ++p) {
        // build submatrix
        std::vector<std::vector<Value>> sub;
        sub.reserve(n-1);
        for (std::size_t i = 1; i < n; ++i) {
            std::vector<Value> row;
            row.reserve(n-1);
            for (std::size_t j = 0; j < n; ++j) {
                if (j == p) continue;
                row.push_back(mat[i][j]);
            }
            sub.push_back(std::move(row));
        }
        Value subdet = determinantSquare(std::move(sub));
        if (p % 2 == 0)
            det += mat[0][p] * subdet;
        else
            det -= mat[0][p] * subdet;
    }
    return det;
}

std::optional<Matrix::Value> Matrix::determinant() const {
    if (rows() != cols()) return std::nullopt;
    auto copy = data_;
    return determinantSquare(std::move(copy));
}

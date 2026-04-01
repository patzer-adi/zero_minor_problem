#pragma once

#include "core/Matrix.hpp"
#include <string>
#include <optional>

namespace refactored::utils {

class IO {
public:
    // Parse whitespace-separated integers into a matrix with given rows/cols
    static std::optional<refactored::core::Matrix> parseMatrixFromString(const std::string &s, std::size_t rows, std::size_t cols);

    // Simple file load: reads whitespace-separated values
    static std::optional<refactored::core::Matrix> loadMatrixFromFile(const std::string &path, std::size_t rows, std::size_t cols);
};

} // namespace

#include "utils/IO.hpp"
#include <sstream>
#include <fstream>

using namespace refactored::utils;
using refactored::core::Matrix;
using Value = Matrix::Value;

std::optional<Matrix> IO::parseMatrixFromString(const std::string &s, std::size_t rows, std::size_t cols) {
    std::istringstream iss(s);
    std::vector<std::vector<Value>> data(rows, std::vector<Value>(cols));
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            if (!(iss >> data[i][j])) return std::nullopt;
        }
    }
    return Matrix(data);
}

std::optional<Matrix> IO::loadMatrixFromFile(const std::string &path, std::size_t rows, std::size_t cols) {
    std::ifstream ifs(path);
    if (!ifs) return std::nullopt;
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    return parseMatrixFromString(content, rows, cols);
}

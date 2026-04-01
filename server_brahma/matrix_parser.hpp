// =============================================================================
// matrix_parser.hpp — Parse matrices from Sage/Python format
// =============================================================================

#ifndef MATRIX_PARSER_HPP
#define MATRIX_PARSER_HPP

#include "apm_types.hpp"
#include <string>
using namespace std;

class MatrixParser {
public:
    // Parse a matrix file in Sage/Python [[...],[...],...] format.
    // Throws std::runtime_error on failure.
    static MatrixData parse(const string &path);
};

#endif // MATRIX_PARSER_HPP

#include <iostream>
#include "core/Matrix.hpp"
#include "services/MinorService.hpp"
#include "utils/IO.hpp"

int main() {
    using namespace refactored::core;
    using namespace refactored::services;
    using namespace refactored::utils;

    // Example usage: construct a 3x3 matrix and compute 2x2 minors
    Matrix m({{1,2,3},{4,5,6},{7,8,9}});
    MinorService ms;

    auto minors = ms.computeKxKMinors(m, 2);
    std::cout << "Found " << minors.size() << " 2x2 minors:\n";
    for (auto v : minors) std::cout << v << "  ";
    std::cout << "\nHas zero 2x2 minor: " << (ms.hasZeroKxKMinor(m,2) ? "yes" : "no") << '\n';

    return 0;
}

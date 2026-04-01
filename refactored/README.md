Refactored zero_minor_problem — initial OOP skeleton

Overview:
- Core model: `refactored/include/core/Matrix.hpp` — encapsulates matrix storage and determinant calculation.
- Service: `refactored/include/services/MinorService.hpp` — computes kxk minors and checks for zero minors.
- Utils: `refactored/include/utils/IO.hpp` — helpers for parsing/loading matrices.

Build:
  mkdir build && cd build
  cmake ..
  cmake --build .

Design notes:
- Converted key procedural concepts (matrix storage, minor computation) into classes.
- Determinant uses a recursive cofactor expansion for clarity (can be optimized later).
- No changes were made to original sources. This directory is independent and builds separately.

Next steps:
- Add automated tests comparing outputs against legacy programs.
- Replace naive determinant with LU decomposition for performance.
- Consolidate other repeated matrix logic from legacy into services here.

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <experimental/filesystem>
#include <chrono>
#include <mpi.h>
#include <NTL/ZZ_p.h>
#include <NTL/mat_ZZ_p.h>

namespace fs = std::experimental::filesystem;
using namespace std;
using namespace NTL;

// Function to read primes from a file
vector<long> read_primes(const string &filename) {
    vector<long> primes;
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error opening prime file: " << filename << endl;
        exit(1);
    }
    long p;
    while (infile >> p) {
        primes.push_back(p);
    }
    infile.close();
    return primes;
}

// Generate candidate row with values in [1, p-1]
vector<ZZ_p> generate_row(long dim) {
    vector<ZZ_p> row(dim);
    for (long i = 0; i < dim; i++) {
        row[i] = 1 + RandomBnd(ZZ_p::modulus() - 1); // 1..p-1
    }
    return row;
}

// Check if new row is independent of existing matrix rows
bool is_independent(const Mat<ZZ_p> &mat, const vector<ZZ_p> &row) {
    Mat<ZZ_p> temp = mat;
    long current_rows = mat.NumRows();
    if (current_rows == 0) return true;
    temp.SetDims(current_rows + 1, mat.NumCols());
    for (long j = 0; j < mat.NumCols(); j++)
        temp[current_rows][j] = row[j];
    return (determinant(temp) != ZZ_p(0));
}

// DFS to build full matrix recursively
bool build_matrix(Mat<ZZ_p> &mat, long target_dim) {
    if (mat.NumRows() == target_dim) return true;
    vector<ZZ_p> row = generate_row(target_dim);
    if (is_independent(mat, row)) {
        Mat<ZZ_p> new_mat = mat;
        new_mat.SetDims(mat.NumRows() + 1, target_dim);
        for (long i = 0; i < mat.NumRows(); i++)
            new_mat[i] = mat[i];
        new_mat[mat.NumRows()] = row;
        mat = new_mat;
        if (build_matrix(mat, target_dim)) return true;
        mat.SetDims(mat.NumRows() - 1, target_dim); // backtrack
    }
    return false;
}

// Attempt to find max dimension for a given prime
long find_max_dim(long prime, Mat<ZZ_p> &result_matrix) {
    ZZ_p::init(ZZ(prime));
    long max_dim = prime - 1; // cannot exceed p-1 due to linear independence
    for (long d = max_dim; d >= 1; d--) {
        Mat<ZZ_p> mat;
        mat.SetDims(0, d);
        if (build_matrix(mat, d)) {
            result_matrix = mat;
            return d;
        }
    }
    return 0;
}

// Write matrix to file
void write_matrix(const Mat<ZZ_p> &mat, const string &filename) {
    ofstream outfile(filename);
    if (!outfile.is_open()) {
        cerr << "Error opening output file: " << filename << endl;
        return;
    }
    for (long i = 0; i < mat.NumRows(); i++) {
        for (long j = 0; j < mat.NumCols(); j++) {
            outfile << mat[i][j] << (j + 1 == mat.NumCols() ? "\n" : " ");
        }
    }
    outfile.close();
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <prime_file> <output_dir>" << endl;
        return 1;
    }

    string prime_file = argv[1];
    string output_dir = argv[2];

    if (!fs::exists(output_dir)) fs::create_directory(output_dir);

    // Initialize MPI
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    vector<long> primes;
    if (world_rank == 0) primes = read_primes(prime_file);

    // Broadcast number of primes to all processes
    long num_primes = primes.size();
    MPI_Bcast(&num_primes, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    if (world_rank != 0) primes.resize(num_primes);

    // Broadcast primes to all processes
    MPI_Bcast(primes.data(), num_primes, MPI_LONG, 0, MPI_COMM_WORLD);

    // Distribute primes among ranks
    for (long idx = world_rank; idx < num_primes; idx += world_size) {
        long p = primes[idx];
        auto start_time = chrono::high_resolution_clock::now();

        Mat<ZZ_p> matrix;
        long max_dim = find_max_dim(p, matrix);

        stringstream ss;
        ss << output_dir << "/matrix_" << p << ".txt";
        write_matrix(matrix, ss.str());

        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end_time - start_time;

        cout << "Rank " << world_rank << ": Prime " << p
             << ", max dimension " << max_dim
             << ", time " << elapsed.count() << " s" << endl;
    }

    MPI_Finalize();
    return 0;
}

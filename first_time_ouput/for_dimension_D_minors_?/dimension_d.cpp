#include <iostream>
#include <fstream>
#include <string>

using namespace std;

// Compute n choose k
unsigned long long binomial(long n, long k) {
    if (k > n - k) k = n - k;
    unsigned long long result = 1;
    for (long i = 1; i <= k; ++i) {
        result *= (n - i + 1);
        result /= i;
    }
    return result;
}

int main() {
    long D;
    cout << "Enter matrix dimension D (e.g., 25): ";
    cin >> D;

    ofstream outfile("minors_count_D_" + to_string(D) + ".txt");

    outfile << "Number of k×k minors in a " << D << "x" << D << " matrix:\n";
    outfile << "---------------------------------------------\n";
    outfile << " k\tTotal k×k minors\n";

    for (long k = 2; k <= D; ++k) {
        unsigned long long count = binomial(D, k) * binomial(D, k);
        outfile << " " << k << "\t" << count << "\n";
    }

    outfile << "---------------------------------------------\n";
    outfile.close();

    cout << "Results saved in file: minors_count_D_" << D << ".txt" << endl;
    return 0;
}

// =============================================================================
// matrix_parser.cpp — Parse matrices from Sage/Python [[...],[...],...] format
// =============================================================================

#include "matrix_parser.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
using namespace std;

MatrixData MatrixParser::parse(const string &path) {
    ifstream ifs(path.c_str());
    if (!ifs.is_open())
        throw runtime_error("Cannot open: " + path);

    MatrixData md;
    md.filename = path.substr(path.find_last_of("/\\") + 1);

    vector<vector<long long>> rows;
    string line;
    bool started = false;

    while (getline(ifs, line)) {
        if (!started) {
            if (line.find("[[") != string::npos)
                started = true;
            else
                continue;
        }
        size_t lb = line.find('[');
        size_t rb = line.find(']');
        if (lb == string::npos || rb == string::npos || rb <= lb)
            continue;

        string tok = line.substr(lb + 1, rb - lb - 1);
        size_t fs = tok.find_first_not_of(" [");
        if (fs == string::npos)
            continue;
        tok = tok.substr(fs);
        replace(tok.begin(), tok.end(), ',', ' ');

        istringstream ss(tok);
        vector<long long> row;
        long long v;
        while (ss >> v)
            row.push_back(v);
        if (!row.empty())
            rows.push_back(row);
        if (line.find("]]") != string::npos)
            break;
    }

    if (rows.empty())
        throw runtime_error("No matrix data in: " + path);

    md.n = static_cast<int>(rows.size());
    md.data.assign(static_cast<size_t>(md.n) * md.n, 0LL);
    for (int i = 0; i < md.n; i++) {
        for (int j = 0; j < static_cast<int>(rows[i].size()) && j < md.n; j++) {
            md.data[static_cast<size_t>(i) * md.n + j] = rows[i][j];
        }
    }
    return md;
}

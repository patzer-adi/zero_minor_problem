#include <NTL/ZZ_p.h>
#include <NTL/mat_ZZ_p.h>
#include <NTL/vector.h>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
using namespace std;
using namespace NTL;

class ProgressBar {
private:
    int width;
    string description;
    
public:
    ProgressBar(int w = 50, string desc = "Progress") : width(w), description(desc) {}
    
    void update(double progress) {
        int pos = width * progress;
        cout << "\r" << description << ": [";
        for (int i = 0; i < width; ++i) {
            if (i < pos) cout << "=";
            else if (i == pos) cout << ">";
            else cout << " ";
        }
        cout << "] " << int(progress * 100.0) << "%";
        cout.flush();
    }
    
    void finish() {
        cout << endl;
    }
};

// Check if a row is unique compared to existing rows
bool isRowUnique(const Vec<ZZ_p>& newRow, const vector<Vec<ZZ_p>>& existingRows) {
    for (const auto& row : existingRows) {
        if (newRow == row) {
            return false;
        }
    }
    return true;
}

// Generate a single unique row that doesn't conflict with existing rows
Vec<ZZ_p> generateUniqueRow(long cols, set<long>& usedElements, 
                           const vector<Vec<ZZ_p>>& existingRows, long p) {
    const int MAX_ATTEMPTS = 10000;
    
    for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
        Vec<ZZ_p> row;
        row.SetLength(cols);
        
        // Get available elements
        vector<long> available;
        for (long i = 1; i <= p; ++i) {
            if (usedElements.find(i) == usedElements.end()) {
                available.push_back(i);
            }
        }
        
        if (available.size() < cols) {
            throw runtime_error("Not enough unused elements to create a unique row");
        }
        
        // Shuffle and pick first 'cols' elements
        random_device rd;
        mt19937 g(rd());
        shuffle(available.begin(), available.end(), g);
        
        set<long> rowElements;
        for (long j = 0; j < cols; ++j) {
            long val = available[j];
            row[j] = ZZ_p(val);
            rowElements.insert(val);
        }
        
        // Check if this row is unique
        if (isRowUnique(row, existingRows)) {
            // Add these elements to the used set
            for (long elem : rowElements) {
                usedElements.insert(elem);
            }
            return row;
        }
    }
    
    throw runtime_error("Could not generate a unique row after maximum attempts");
}

// Check mathematical feasibility
bool checkFeasibility(long rows, long cols, long p) {
    // Need at least rows * cols unique elements
    if (rows * cols > p) {
        cout << "Error: Need " << rows * cols << " unique elements, but only " 
             << p << " elements available (1 to " << p << ")" << endl;
        return false;
    }
    
    // For large matrices, warn about potential difficulty
    if (rows > 10 && cols > 10) {
        cout << "Warning: Large matrix requested. This may take significant time." << endl;
    }
    
    return true;
}

Mat<ZZ_p> generateUniqueMatrix(long rows, long cols, long p) {
    if (!checkFeasibility(rows, cols, p)) {
        throw runtime_error("Matrix generation not feasible with given parameters");
    }
    
    Mat<ZZ_p> M;
    M.SetDims(rows, cols);
    
    set<long> usedElements;
    vector<Vec<ZZ_p>> existingRows;
    
    ProgressBar progressBar(50, "Generating matrix");
    
    for (long i = 0; i < rows; ++i) {
        try {
            Vec<ZZ_p> row = generateUniqueRow(cols, usedElements, existingRows, p);
            
            // Copy row to matrix
            for (long j = 0; j < cols; ++j) {
                M[i][j] = row[j];
            }
            
            existingRows.push_back(row);
            
            // Update progress
            double progress = (double)(i + 1) / rows;
            progressBar.update(progress);
            
        } catch (const exception& e) {
            progressBar.finish();
            throw runtime_error("Failed to generate row " + to_string(i + 1) + ": " + e.what());
        }
    }
    
    progressBar.finish();
    return M;
}

void printMatrixInfo(const Mat<ZZ_p>& M, long p) {
    long rows = M.NumRows();
    long cols = M.NumCols();
    
    cout << "\nMatrix Information:" << endl;
    cout << "Size: " << rows << " x " << cols << endl;
    cout << "Field: GF(" << p << ")" << endl;
    cout << "Total elements: " << rows * cols << endl;
    
    // Verify uniqueness of elements
    set<long> allElements;
    for (long i = 0; i < rows; ++i) {
        for (long j = 0; j < cols; ++j) {
            long val = conv<long>(M[i][j]);
            allElements.insert(val);
        }
    }
    cout << "Unique elements used: " << allElements.size() << endl;
    
    // Verify row uniqueness
    set<string> rowStrings;
    for (long i = 0; i < rows; ++i) {
        string rowStr = "";
        for (long j = 0; j < cols; ++j) {
            rowStr += to_string(conv<long>(M[i][j])) + ",";
        }
        rowStrings.insert(rowStr);
    }
    cout << "Unique rows: " << rowStrings.size() << " out of " << rows << endl;
}

int main() {
    long p, rows, cols;
    
    cout << "=== Unique Matrix Generator ===" << endl;
    cout << "Enter prime p (elements will be from 1 to p): ";
    cin >> p;
    
    if (p <= 1) {
        cerr << "Error: p must be greater than 1" << endl;
        return 1;
    }
    
    cout << "Enter number of rows: ";
    cin >> rows;
    cout << "Enter number of columns: ";
    cin >> cols;
    
    if (rows <= 0 || cols <= 0) {
        cerr << "Error: Rows and columns must be positive" << endl;
        return 1;
    }
    
    ZZ_p::init(p); // Initialize field GF(p)
    
    auto start_time = chrono::high_resolution_clock::now();
    
    try {
        cout << "\nGenerating " << rows << " x " << cols << " matrix with unique elements and rows..." << endl;
        Mat<ZZ_p> M = generateUniqueMatrix(rows, cols, p);
        
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        
        cout << "\nGenerated Matrix:" << endl;
        cout << M << endl;
        
        printMatrixInfo(M, p);
        cout << "\nGeneration completed in " << duration.count() << " milliseconds." << endl;
        
    } catch (const exception &e) {
        cerr << "\nError: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
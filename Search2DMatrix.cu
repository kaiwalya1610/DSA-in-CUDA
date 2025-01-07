#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cassert>

using namespace std;

class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size();
        if (m == 0)
            return false;

        int n = matrix[0].size();
        int left = 0, right = m * n - 1;
        int pivotIdx, pivotElement;

        while (left <= right) {
            pivotIdx = (left + right) / 2;
            pivotElement = matrix[pivotIdx / n][pivotIdx % n];
            if (target == pivotElement)
                return true;
            else {
                if (target < pivotElement)
                    right = pivotIdx - 1;
                else
                    left = pivotIdx + 1;
            }
        }
        return false;
    }
};

__global__ void searchMatrixCUDA(int* d_matrix, int m, int n, int target, bool* d_result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= m * n) return;

    int row = idx / n;
    int col = idx % n;

    if (d_matrix[row * n + col] == target) {
        *d_result = true;
    }
}

int main() {
    // Generate a large matrix
    const int m = 10000; // number of rows
    const int n = 10000;// number of columns
    const int target = 123456;
    vector<vector<int>> matrix(m, vector<int>(n));
    
    int counter = 1;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = counter++;
        }
    }

    // Insert target in the matrix
    matrix[m / 2][n / 2] = target;

    // Measure time for CPU implementation
    Solution solution;
    auto startCPU = chrono::high_resolution_clock::now();
    bool resultCPU = solution.searchMatrix(matrix, target);
    auto endCPU = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsedCPU = endCPU - startCPU;
    cout << "CPU Result: " << resultCPU << "\n";
    cout << "CPU Time: " << elapsedCPU.count() << " seconds\n";

    // Flatten the matrix for CUDA
    int* h_matrix = new int[m * n];
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_matrix[i * n + j] = matrix[i][j];
        }
    }

    int* d_matrix;
    bool* d_result;
    bool h_result = false;

    cudaMalloc(&d_matrix, m * n * sizeof(int));
    cudaMalloc(&d_result, sizeof(bool));

    cudaMemcpy(d_matrix, h_matrix, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(bool), cudaMemcpyHostToDevice);

    // Measure time for CUDA implementation
    auto startCUDA = chrono::high_resolution_clock::now();

    searchMatrixCUDA<<<(m * n + 255) / 256, 256>>>(d_matrix, m, n, target, d_result);
    cudaDeviceSynchronize();

    auto endCUDA = chrono::high_resolution_clock::now();

    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    chrono::duration<double> elapsedCUDA = endCUDA - startCUDA;
    cout << "CUDA Result: " << h_result << "\n";
    cout << "CUDA Time: " << elapsedCUDA.count() << " seconds\n";

    // Check if results are consistent
    assert(resultCPU == h_result);

    // Cleanup
    delete[] h_matrix;
    cudaFree(d_matrix);
    cudaFree(d_result);

    return 0;
}

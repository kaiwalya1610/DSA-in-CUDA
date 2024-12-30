#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

__global__ void rotateKernel(int* d_matrix, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Only process the upper-left quadrant
    if (i < (n + 1) / 2 && j < n / 2) {
        // Perform the rotation swaps
        int temp = d_matrix[(n - 1 - j) * n + i];
        d_matrix[(n - 1 - j) * n + i] = d_matrix[(n - 1 - i) * n + (n - j - 1)];
        d_matrix[(n - 1 - i) * n + (n - j - 1)] = d_matrix[j * n + (n - 1 - i)];
        d_matrix[j * n + (n - 1 - i)] = d_matrix[i * n + j];
        d_matrix[i * n + j] = temp;
    }
}

void rotateMatrixCPU(std::vector<std::vector<int>>& matrix) {
    int n = matrix.size();
    for (int i = 0; i < (n + 1) / 2; i++) {
        for (int j = 0; j < n / 2; j++) {
            int temp = matrix[n - 1 - j][i];
            matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1];
            matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 - i];
            matrix[j][n - 1 - i] = matrix[i][j];
            matrix[i][j] = temp;
        }
    }
}

void rotateMatrixGPU(std::vector<std::vector<int>>& matrix) {
    int n = matrix.size();

    // Flatten the 2D matrix into a 1D array for CUDA
    int* h_matrix = new int[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_matrix[i * n + j] = matrix[i][j];
        }
    }

    // Allocate device memory
    int* d_matrix;
    cudaMalloc(&d_matrix, n * n * sizeof(int));

    // Copy the matrix from host to device
    cudaMemcpy(d_matrix, h_matrix, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    rotateKernel<<<gridDim, blockDim>>>(d_matrix, n);
    cudaDeviceSynchronize();

    // Copy the rotated matrix back to the host
    cudaMemcpy(h_matrix, d_matrix, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix);

    // Convert the 1D array back to the 2D matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = h_matrix[i * n + j];
        }
    }

    // Free host memory
    delete[] h_matrix;
}

int main() {
    const int n = 10000;

    // Generate a random 10000 x 10000 matrix
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = rand() % 100;
        }
    }

    // Copy matrix for GPU processing
    std::vector<std::vector<int>> matrixGPU = matrix;

    // Measure CPU rotation time
    auto startCPU = std::chrono::high_resolution_clock::now();
    rotateMatrixCPU(matrix);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedCPU = endCPU - startCPU;
    std::cout << "CPU Rotation Time: " << elapsedCPU.count() << " seconds" << std::endl;

    // Measure GPU rotation time
    auto startGPU = std::chrono::high_resolution_clock::now();
    rotateMatrixGPU(matrixGPU);
    auto endGPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedGPU = endGPU - startGPU;
    std::cout << "GPU Rotation Time: " << elapsedGPU.count() << " seconds" << std::endl;

    return 0;
}

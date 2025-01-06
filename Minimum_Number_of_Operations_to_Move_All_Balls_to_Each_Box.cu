#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace std;

// CPU Implementation (O(n))
vector<int> minOperationsCPU(const string &boxes) {
    int n = boxes.size();
    vector<int> answer(n, 0);

    int ballsToLeft = 0, movesToLeft = 0;
    int ballsToRight = 0, movesToRight = 0;

    for (int i = 0; i < n; i++) {
        // Left pass
        answer[i] += movesToLeft;
        ballsToLeft += boxes[i] - '0';
        movesToLeft += ballsToLeft;

        // Right pass
        int j = n - 1 - i;
        answer[j] += movesToRight;
        ballsToRight += boxes[j] - '0';
        movesToRight += ballsToRight;
    }

    return answer;
}

/**
 * GPU Implementation Outline:
 *
 * 1) Convert boxes to device array of ints (0 or 1).
 * 2) Build prefix sums from the left:
 *     prefixBalls[i] = sum of boxes[0..i]
 *     prefixMoves[i] = prefixMoves[i-1] + prefixBalls[i]
 *
 * 3) Build prefix sums from the right in reverse:
 *     suffixBalls[i] = sum of boxes[i..n-1]
 *     suffixMoves[i] = suffixMoves[i+1] + suffixBalls[i]
 *
 * 4) Combine:
 *     answer[i] = prefixMoves[i-1] + suffixMoves[i+1]
 *                 (with boundary checks)
 *     Because prefixMoves[i] is the total moves from [0..i] to i,
 *     suffixMoves[i] is the total moves from [i..n-1] to i.
 */

// Kernel to convert '0'/'1' chars to int (0 or 1)
__global__ void charToInt(const char *boxes, int *values, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        values[idx] = (boxes[idx] == '1') ? 1 : 0;
    }
}

// Kernel to combine prefix and suffix into final answer
__global__ void combineResults(const long long *prefixMoves,
                               const long long *suffixMoves,
                               int *answer, 
                               int n) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        long long leftPart  = (idx == 0)     ? 0 : prefixMoves[idx - 1];
        long long rightPart = (idx == n - 1) ? 0 : suffixMoves[idx + 1];
        answer[idx] = (int)(leftPart + rightPart);
    }
}

int main() {
    // --------------------------------------------------
    // 1) Generate a large test input
    // --------------------------------------------------
    string boxes(1'000'000, '0');
    for (int i = 0; i < (int)boxes.size(); i++) {
        if (i % 10 == 0) {
            boxes[i] = '1';  // Add some 1s
        }
    }
    int n = boxes.size();

    // --------------------------------------------------
    // 2) CPU Execution
    // --------------------------------------------------
    auto cpu_start = chrono::high_resolution_clock::now();
    vector<int> cpu_result = minOperationsCPU(boxes);
    auto cpu_end = chrono::high_resolution_clock::now();
    double cpu_time = chrono::duration<double, milli>(cpu_end - cpu_start).count();

    // --------------------------------------------------
    // 3) GPU Execution
    // --------------------------------------------------
    // Host memory allocations
    vector<int> gpu_result(n, 0);

    // Device memory allocations
    char *d_boxes;
    cudaMalloc(&d_boxes, n * sizeof(char));
    cudaMemcpy(d_boxes, boxes.data(), n * sizeof(char), cudaMemcpyHostToDevice);

    int *d_values;       // 0 or 1 for each char
    cudaMalloc(&d_values, n * sizeof(int));

    int *d_answer;
    cudaMalloc(&d_answer, n * sizeof(int));

    // Prefix sums and suffix sums each need an array of length n
    long long *d_prefixBalls, *d_prefixMoves;
    cudaMalloc(&d_prefixBalls, n * sizeof(long long));
    cudaMalloc(&d_prefixMoves, n * sizeof(long long));

    long long *d_suffixBalls, *d_suffixMoves;
    cudaMalloc(&d_suffixBalls, n * sizeof(long long));
    cudaMalloc(&d_suffixMoves, n * sizeof(long long));

    // Convert to int array
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    charToInt<<<blocks, threads>>>(d_boxes, d_values, n);
    cudaDeviceSynchronize();

    // Use Thrust for prefix sums (inclusive scan) on device
    // (1) prefixBalls[i] = sum of d_values[0..i]
    {
        thrust::device_ptr<int> dev_ptr_in(d_values);
        thrust::device_ptr<long long> dev_ptr_out(d_prefixBalls);

        // inclusive_scan: out[i] = in[0] + ... + in[i]
        thrust::inclusive_scan(dev_ptr_in, dev_ptr_in + n, dev_ptr_out);
    }
    // prefixMoves[i] = prefixMoves[i-1] + prefixBalls[i]
    // => prefixMoves[i] = sum_{k=0..i} prefixBalls[k]
    {
        thrust::device_ptr<long long> dev_ptr_in(d_prefixBalls);
        thrust::device_ptr<long long> dev_ptr_out(d_prefixMoves);

        thrust::inclusive_scan(dev_ptr_in, dev_ptr_in + n, dev_ptr_out);
    }

    // (2) suffixBalls[i] = sum of d_values[i..n-1]
    // We'll reuse thrust by reversing the data or by scanning from the back.
    // Easiest trick: copy data into a reversed array, do inclusive_scan, then reverse back.
    {
        thrust::device_vector<long long> temp(n);
        thrust::copy_n(thrust::device_pointer_cast(d_values), n, temp.begin());

        // Reverse temp so that temp[0] = d_values[n-1], etc.
        thrust::reverse(temp.begin(), temp.end());

        // Now do an inclusive_scan on reversed data
        thrust::inclusive_scan(temp.begin(), temp.end(), temp.begin());

        // Reverse it back to get suffixBalls
        thrust::reverse(temp.begin(), temp.end());

        // Copy into d_suffixBalls
        cudaMemcpy(d_suffixBalls, thrust::raw_pointer_cast(temp.data()),
                   n * sizeof(long long), cudaMemcpyDeviceToDevice);
    }
    // suffixMoves[i] = suffixMoves[i+1] + suffixBalls[i]
    // => suffixMoves[i] = sum_{k=i..n-1} suffixBalls[k]
    {
        thrust::device_vector<long long> temp(n);
        cudaMemcpy(thrust::raw_pointer_cast(temp.data()),
                   d_suffixBalls, 
                   n * sizeof(long long),
                   cudaMemcpyDeviceToDevice);

        // Reverse, scan, reverse
        thrust::reverse(temp.begin(), temp.end());
        thrust::inclusive_scan(temp.begin(), temp.end(), temp.begin());
        thrust::reverse(temp.begin(), temp.end());

        cudaMemcpy(d_suffixMoves, thrust::raw_pointer_cast(temp.data()),
                   n * sizeof(long long), cudaMemcpyDeviceToDevice);
    }

    auto gpu_start = chrono::high_resolution_clock::now();

    // Combine prefixMoves + suffixMoves into final answer
    combineResults<<<blocks, threads>>>(d_prefixMoves, d_suffixMoves, d_answer, n);
    cudaDeviceSynchronize();

    auto gpu_end = chrono::high_resolution_clock::now();
    double gpu_time = chrono::duration<double, milli>(gpu_end - gpu_start).count();

    // Copy result back
    cudaMemcpy(gpu_result.data(), d_answer, n * sizeof(int), cudaMemcpyDeviceToHost);

    // --------------------------------------------------
    // 4) Compare Results
    // --------------------------------------------------
    bool is_same = true;
    for (int i = 0; i < n; i++) {
        if (cpu_result[i] != gpu_result[i]) {
            is_same = false;
            break;
        }
    }

    // --------------------------------------------------
    // 5) Print Timing and Validation
    // --------------------------------------------------
    cout << "CPU Time: " << cpu_time << " ms\n";
    cout << "GPU Time (Combine Kernel Only): " << gpu_time << " ms\n";
    cout << "Results match: " << (is_same ? "Yes" : "No") << endl;

    // --------------------------------------------------
    // Cleanup
    // --------------------------------------------------
    cudaFree(d_boxes);
    cudaFree(d_values);
    cudaFree(d_answer);
    cudaFree(d_prefixBalls);
    cudaFree(d_prefixMoves);
    cudaFree(d_suffixBalls);
    cudaFree(d_suffixMoves);

    return 0;
}

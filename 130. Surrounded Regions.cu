#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;

// -------------------------------------------------------------------
// CPU SOLUTION
// -------------------------------------------------------------------
class SolutionCPU {
public:
    void solve(vector<vector<char>>& board) {
        if (board.empty() || board[0].empty()) return;
        ROWS = board.size(); 
        COLS = board[0].size();

        // Mark from boundary
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                if ((i == 0 || j == 0 || i == ROWS-1 || j == COLS-1) && board[i][j] == 'O') {
                    markBoundary(board, i, j);
                }
            }
        }

        // Flip
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                if (board[i][j] == 'O')      board[i][j] = 'X';
                else if (board[i][j] == 'E') board[i][j] = 'O';
            }
        }
    }

private:
    int ROWS, COLS;
    void markBoundary(vector<vector<char>>& board, int r, int c) {
        stack<pair<int,int>> stk;
        stk.push({r, c});
        board[r][c] = 'E';

        static int dr[4] = {1, -1, 0, 0};
        static int dc[4] = {0, 0, 1, -1};

        while (!stk.empty()) {
            auto topPair = stk.top(); 
            int rr = topPair.first; 
            int cc = topPair.second; 

 
            stk.pop();
            for (int i = 0; i < 4; i++) {
                int nr = rr + dr[i], nc = cc + dc[i];
                if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && board[nr][nc] == 'O') {
                    board[nr][nc] = 'E';
                    stk.push({nr, nc});
                }
            }
        }
    }
};

// -------------------------------------------------------------------
// GPU SOLUTION (NAIVE FLOOD-FILL)
// -------------------------------------------------------------------
__global__ void markBoundaryKernel(char* d_board, int rows, int cols) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int r = idx / cols, c = idx % cols;
    // Mark boundary 'O' -> 'E'
    if ((r == 0 || r == rows - 1 || c == 0 || c == cols - 1) && d_board[idx] == 'O') {
        d_board[idx] = 'E';
    }
}

__global__ void expandEKernel(char* d_board, char* d_boardNext, bool* changed, int rows, int cols) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int r = idx / cols, c = idx % cols;

    d_boardNext[idx] = d_board[idx]; // copy old state
    if (d_board[idx] == 'O') {
        // check up
        if (r > 0 && d_board[(r-1)*cols + c] == 'E') {
            d_boardNext[idx] = 'E';
            *changed = true;
        }
        // down
        else if (r < rows-1 && d_board[(r+1)*cols + c] == 'E') {
            d_boardNext[idx] = 'E';
            *changed = true;
        }
        // left
        else if (c > 0 && d_board[r*cols + (c-1)] == 'E') {
            d_boardNext[idx] = 'E';
            *changed = true;
        }
        // right
        else if (c < cols-1 && d_board[r*cols + (c+1)] == 'E') {
            d_boardNext[idx] = 'E';
            *changed = true;
        }
    }
}

__global__ void finalFlipKernel(char* d_board, int rows, int cols) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= rows * cols) return;
    if (d_board[idx] == 'O') d_board[idx] = 'X';
    else if (d_board[idx] == 'E') d_board[idx] = 'O';
}

void solveGPU(vector<vector<char>>& board) {
    if (board.empty() || board[0].empty()) return;
    int rows = board.size(), cols = board[0].size();
    int total = rows * cols;

    // Flatten
    vector<char> flat(total);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat[i * cols + j] = board[i][j];
        }
    }

    // Device alloc
    char *d_board, *d_boardNext;
    bool *d_changed;
    cudaMalloc(&d_board,     total * sizeof(char));
    cudaMalloc(&d_boardNext, total * sizeof(char));
    cudaMalloc(&d_changed,   sizeof(bool));

    cudaMemcpy(d_board, flat.data(), total*sizeof(char), cudaMemcpyHostToDevice);

    // Kernel configs
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;

    // 1) Mark boundary
    markBoundaryKernel<<<blocks, threads>>>(d_board, rows, cols);
    cudaDeviceSynchronize();

    // 2) Iteratively expand 'E'
    bool changed = true;
    while (changed) {
        changed = false;
        cudaMemcpy(d_changed, &changed, sizeof(bool), cudaMemcpyHostToDevice);

        expandEKernel<<<blocks, threads>>>(d_board, d_boardNext, d_changed, rows, cols);
        cudaDeviceSynchronize();

        // Swap
        char* temp = d_board;
        d_board = d_boardNext;
        d_boardNext = temp;

        cudaMemcpy(&changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    // 3) Flip
    finalFlipKernel<<<blocks, threads>>>(d_board, rows, cols);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(flat.data(), d_board, total*sizeof(char), cudaMemcpyDeviceToHost);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            board[i][j] = flat[i*cols + j];
        }
    }

    cudaFree(d_board);
    cudaFree(d_boardNext);
    cudaFree(d_changed);
}

// -------------------------------------------------------------------
// MAIN: Compare CPU & GPU
// -------------------------------------------------------------------
int main() {
    // Generate a large random board
    int rows = 20000, cols = 20000;
    vector<vector<char>> board(rows, vector<char>(cols, 'X'));
    srand(0);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (rand() % 7 == 0) {
                board[i][j] = 'O';
            }
        }
    }

    // Make copies
    auto boardCPU = board;
    auto boardGPU = board;

    // CPU time
    auto startCPU = std::chrono::high_resolution_clock::now();
    SolutionCPU().solve(boardCPU);
    auto endCPU = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(endCPU - startCPU).count();

    // GPU time
    auto startGPU = std::chrono::high_resolution_clock::now();
    solveGPU(boardGPU);
    auto endGPU = std::chrono::high_resolution_clock::now();
    double gpuTime = std::chrono::duration<double, std::milli>(endGPU - startGPU).count();

    // Compare results
    bool same = true;
    for(int i = 0; i < rows && same; i++){
        for(int j = 0; j < cols && same; j++){
            if(boardCPU[i][j] != boardGPU[i][j]) {
                same = false;
            }
        }
    }

    // Print
    cout << "CPU Time: " << cpuTime << " ms\n";
    cout << "GPU Time: " << gpuTime << " ms\n";
    cout << "Results match: " << (same ? "Yes" : "No") << endl;

    return 0;
}

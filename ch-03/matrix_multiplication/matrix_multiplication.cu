
#include <cuda_runtime.h>
#include <stdio.h>
#include "../../lib/timer.h"
#include "../../lib/testing.h"

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    // M x K
    // M = height
    // K = width
    if (row < M && col < K) {
        unsigned int idx = row * K + col;
        float sum = 0;

        for (int i = 0; i < N; i++) {
            sum += A[row * N + i]*B[i * K + col];
        }
        C[idx] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x, // 4 + 15 / 16 = 1
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y); // 3 + 15 / 16 = 1

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

void matrix_manipulation_gpu(const float* A, const float* B, float* C, int M, int N, int K) {
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, M*N*sizeof(float));
    cudaMalloc((void**)&B_d, N*K*sizeof(float));
    cudaMalloc((void**)&C_d, M*K*sizeof(float));
    cudaDeviceSynchronize();

    cudaMemcpy(A_d, A, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    solve(A_d, B_d, C_d, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(C, C_d, M*K*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    const int M = 3, N = 2, K = 4;
    float A[M*N] = {1, 2,
                    3, 4,
                    5, 6};
    float B[N*K] = {7, 8, 9, 10,
                    11, 12, 13, 14};
    float C[M*K];
    
    float expected[M*K] = {
        29, 32, 35, 38,
        65, 72, 79, 86,
        101, 112, 123, 134
    };
    matrix_manipulation_gpu(A, B, C, M, N, K);
    print_matrix(A, M, N, "A");
    print_matrix(B, N, K, "B");
    print_matrix(C, M, K, "C (output)");
    print_matrix(expected, M, K, "Expected");
    printf("Test 1: %s\n\n", arrays_equal(C, expected, M*K) ? "PASS" : "FAIL");

    return 0;
}
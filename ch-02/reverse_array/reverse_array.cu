#include <cuda_runtime.h>
#include <stdio.h>

__global__ void reverse_array(float* input, int N) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        float temp = input[i];
        input[i] = input[N - i - 1];
        input[N - i - 1] = temp;
    }
}

extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int numThreads = N / 2;
    int blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}

void print_array(float* arr, int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%.1f%s", arr[i], i < n - 1 ? ", " : "");
    }
    printf("]\n");
}

int arrays_equal(float* a, float* b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

int main() {
    float *d_arr;
    float h_arr[6];
    
    // Test 1: odd length
    float test1[] = {1, 2, 3, 4, 5};
    float expected1[] = {5, 4, 3, 2, 1};
    cudaMalloc(&d_arr, 5 * sizeof(float));
    cudaMemcpy(d_arr, test1, 5 * sizeof(float), cudaMemcpyHostToDevice);
    solve(d_arr, 5);
    cudaMemcpy(h_arr, d_arr, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Test 1 input:    "); print_array(test1, 6);
    printf("Test 1 output:   "); print_array(h_arr, 6);
    printf("Test 1 expected: "); print_array(expected1, 6);
    printf("Test 1: %s\n\n", arrays_equal(h_arr, expected1, 6) ? "PASS" : "FAIL");
    cudaFree(d_arr);

    // Test 2: even length
    float test2[] = {1, 2, 3, 4, 5, 6};
    float expected2[] = {6, 5, 4, 3, 2, 1};
    cudaMalloc(&d_arr, 6 * sizeof(float));
    cudaMemcpy(d_arr, test2, 6 * sizeof(float), cudaMemcpyHostToDevice);
    solve(d_arr, 6);
    cudaMemcpy(h_arr, d_arr, 6 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Test 2 input:    "); print_array(test2, 6);
    printf("Test 2 output:   "); print_array(h_arr, 6);
    printf("Test 2 expected: "); print_array(expected2, 6);
    printf("Test 2: %s\n\n", arrays_equal(h_arr, expected2, 6) ? "PASS" : "FAIL");
    cudaFree(d_arr);

    // Test 3: single element
    float test3[] = {42};
    float expected3[] = {42};
    cudaMalloc(&d_arr, 1 * sizeof(float));
    cudaMemcpy(d_arr, test3, 1 * sizeof(float), cudaMemcpyHostToDevice);
    solve(d_arr, 1);
    cudaMemcpy(h_arr, d_arr, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Test 3 (single): %s\n\n", arrays_equal(h_arr, expected3, 1) ? "PASS" : "FAIL");
    cudaFree(d_arr);

    // Test 4: two elements
    float test4[] = {1, 2};
    float expected4[] = {2, 1};
    cudaMalloc(&d_arr, 2 * sizeof(float));
    cudaMemcpy(d_arr, test4, 2 * sizeof(float), cudaMemcpyHostToDevice);
    solve(d_arr, 2);
    cudaMemcpy(h_arr, d_arr, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Test 4 (two):    %s\n\n", arrays_equal(h_arr, expected4, 2) ? "PASS" : "FAIL");
    cudaFree(d_arr);

    return 0;
}

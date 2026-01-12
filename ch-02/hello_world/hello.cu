#include <stdio.h>
#include <stdlib.h> // For rand() and srand()
#include <unistd.h> // For usleep()
#include <time.h>   // For time()

__global__ void hello() {
    long microseconds_to_wait = (rand() % 2000001);
    printf("waiting for %ld microseconds on thread %d\n", microseconds_to_wait, threadIdx.x);
    usleep(microseconds_to_wait);
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    hello<<<1, 10>>>();
    usleep(microseconds_to_wait);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Sync error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    return 0;
}
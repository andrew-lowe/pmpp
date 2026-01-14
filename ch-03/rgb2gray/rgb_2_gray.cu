#include <cuda_runtime.h>
#include <stdio.h>
#include "../../lib/timer.h"

__global__ void rgb2gray_kernel(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height) {
    // identify which row / column in image it's responsible for 
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // linear index of pixel
    if (row < height && col < width) {
        unsigned int i = row * width + col;
        gray[i] = red[i]*3/10 + green[i]*6/10 + blue[i]*1/10;
    }
}

void rgb2gray_gpu(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height) {
    Timer timer;

    // Allocate GPU memory
    startTimer(&timer);
    unsigned char *red_d, *green_d, *blue_d, *gray_d;
    cudaMalloc((void**)&red_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&green_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&blue_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&gray_d, width * height * sizeof(unsigned char));
    cudaDeviceSynchronize();
    stopTimer(&timer);
    printElapsedTime(&timer, "GPU memory allocation");

    // Copy to the GPU
    startTimer(&timer);
    cudaMemcpy(red_d, red, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTimer(&timer);
    printElapsedTime(&timer, "GPU memory copy");

    // Call kernel
    startTimer(&timer);
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, 
                   (height + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

    rgb2gray_kernel<<< numBlocks, numThreadsPerBlock >>>(red_d, green_d, blue_d, gray_d, width, height);
    cudaDeviceSynchronize();
    stopTimer(&timer);
    printElapsedTime(&timer, "GPU kernel time");

    // Copy from the GPU
    startTimer(&timer);
    cudaMemcpy(gray, gray_d, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTimer(&timer);
    printElapsedTime(&timer, "GPU memory copy");

    // Free GPU memory
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);
}

void print_image(unsigned char* img, int width, int height, const char* name) {
    printf("%s (%dx%d):\n", name, width, height);
    for (int row = 0; row < height; row++) {
        printf("  ");
        for (int col = 0; col < width; col++) {
            printf("%3d ", img[row * width + col]);
        }
        printf("\n");
    }
}

int arrays_equal(unsigned char* a, unsigned char* b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

int main() {
    // Test 1: 3x2 image
    unsigned int width1 = 3, height1 = 2;
    unsigned char red1[]   = {255,   0,   0, 100, 100, 100};
    unsigned char green1[] = {  0, 255,   0, 100, 100, 100};
    unsigned char blue1[]  = {  0,   0, 255, 100, 100, 100};
    unsigned char gray1[6];
    // Expected: R*0.3 + G*0.6 + B*0.1 (integer math: R*3/10 + G*6/10 + B*1/10)
    // (255,0,0) -> 255*3/10 = 76
    // (0,255,0) -> 255*6/10 = 153
    // (0,0,255) -> 255*1/10 = 25
    // (100,100,100) -> 30+60+10 = 100
    unsigned char expected1[] = {76, 153, 25, 100, 100, 100};

    rgb2gray_gpu(red1, green1, blue1, gray1, width1, height1);
    print_image(gray1, width1, height1, "Test 1 output");
    print_image(expected1, width1, height1, "Test 1 expected");
    printf("Test 1: %s\n\n", arrays_equal(gray1, expected1, 6) ? "PASS" : "FAIL");

    // Test 2: 2x2 white image
    unsigned int width2 = 2, height2 = 2;
    unsigned char red2[]   = {255, 255, 255, 255};
    unsigned char green2[] = {255, 255, 255, 255};
    unsigned char blue2[]  = {255, 255, 255, 255};
    unsigned char gray2[4];
    // White: 255*3/10 + 255*6/10 + 255*1/10 = 76+153+25 = 254
    unsigned char expected2[] = {254, 254, 254, 254};

    rgb2gray_gpu(red2, green2, blue2, gray2, width2, height2);
    print_image(gray2, width2, height2, "Test 2 output");
    print_image(expected2, width2, height2, "Test 2 expected");
    printf("Test 2: %s\n\n", arrays_equal(gray2, expected2, 4) ? "PASS" : "FAIL");

    // Test 3: 1x1 single pixel
    unsigned int width3 = 1, height3 = 1;
    unsigned char red3[]   = {128};
    unsigned char green3[] = {64};
    unsigned char blue3[]  = {32};
    unsigned char gray3[1];
    // 128*3/10 + 64*6/10 + 32*1/10 = 38+38+3 = 79
    unsigned char expected3[] = {79};

    rgb2gray_gpu(red3, green3, blue3, gray3, width3, height3);
    printf("Test 3 (single pixel): output=%d, expected=%d: %s\n\n",
           gray3[0], expected3[0], arrays_equal(gray3, expected3, 1) ? "PASS" : "FAIL");

    return 0;
}
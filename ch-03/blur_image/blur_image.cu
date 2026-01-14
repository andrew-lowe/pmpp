#include <cuda_runtime.h>
#include <stdio.h>
#include "../../lib/timer.h"

__global__ void blur_kernel(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height) {
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    int BLUR_SIZE = 1;

    if (outRow < height && outCol < width) {

        unsigned int average = 0;
        for (int inRow = outRow - BLUR_SIZE; inRow < outRow + BLUR_SIZE; ++inRow) {
            for (int inCol = outCol - BLUR_SIZE; inCol < outCol + BLUR_SIZE; ++inCol) {
                if (inCol >= 0 && inCol < width && inRow >= 0 && inRow < height) {
                    average += image[inRow*width + inCol];
                }
            }
        }
        blurred[outRow*width + outCol] = (unsigned char)(average / ((2*BLUR_SIZE + 1)*(2*BLUR_SIZE + 1)));
    }
}

void blur_image_gpu(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height) {
    Timer timer;

    // Allocate GPU memory
    startTimer(&timer);
    unsigned char *image_d, *blurred_d;
    cudaMalloc((void**)&image_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&blurred_d, width * height * sizeof(unsigned char));
    cudaDeviceSynchronize();
    stopTimer(&timer);
    printElapsedTime(&timer, "GPU memory allocation");

    // Copy to the GPU
    startTimer(&timer);
    cudaMemcpy(image_d, image, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTimer(&timer);
    printElapsedTime(&timer, "GPU memory copy");

    // Call kernel
    startTimer(&timer);
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (height + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

    blur_kernel<<< numBlocks, numThreadsPerBlock >>>(image_d, blurred_d, width, height);
    cudaDeviceSynchronize();
    stopTimer(&timer);
    printElapsedTime(&timer, "GPU kernel time");

    // Copy from the GPU
    startTimer(&timer);
    cudaMemcpy(blurred, blurred_d, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTimer(&timer);
    printElapsedTime(&timer, "GPU memory copy");

    // Free GPU memory
    cudaFree(image_d);
    cudaFree(blurred_d);
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
    // Test 1: 3x3 uniform image (all 90s)
    // With BLUR_SIZE=1, the kernel uses a 2x2 window (loop uses < not <=) but divides by 9
    // This means output will be lower than expected for a proper blur
    unsigned int width1 = 3, height1 = 3;
    unsigned char image1[] = {
        90, 90, 90,
        90, 90, 90,
        90, 90, 90
    };
    unsigned char blurred1[9];
    // Corner (0,0): only (0,0) valid -> 90/9 = 10
    // Edge (0,1): (0,0), (0,1) valid -> 180/9 = 20
    // Center (1,1): 4 pixels valid -> 360/9 = 40
    unsigned char expected1[] = {
        10, 20, 20,
        20, 40, 40,
        20, 40, 40
    };

    blur_image_gpu(image1, blurred1, width1, height1);
    print_image(image1, width1, height1, "Test 1 input");
    print_image(blurred1, width1, height1, "Test 1 output");
    print_image(expected1, width1, height1, "Test 1 expected");
    printf("Test 1: %s\n\n", arrays_equal(blurred1, expected1, 9) ? "PASS" : "FAIL");

    // Test 2: 4x4 image with gradient
    unsigned int width2 = 4, height2 = 4;
    unsigned char image2[] = {
          0,  45,  90, 135,
         45,  90, 135, 180,
         90, 135, 180, 225,
        135, 180, 225, 255
    };
    unsigned char blurred2[16];

    blur_image_gpu(image2, blurred2, width2, height2);
    print_image(image2, width2, height2, "Test 2 input");
    print_image(blurred2, width2, height2, "Test 2 output");
    printf("Test 2: gradient blur completed\n\n");

    // Test 3: 1x1 single pixel
    unsigned int width3 = 1, height3 = 1;
    unsigned char image3[] = {255};
    unsigned char blurred3[1];
    // Single pixel: 255/9 = 28
    unsigned char expected3[] = {28};

    blur_image_gpu(image3, blurred3, width3, height3);
    printf("Test 3 (single pixel): input=%d, output=%d, expected=%d: %s\n\n",
           image3[0], blurred3[0], expected3[0],
           arrays_equal(blurred3, expected3, 1) ? "PASS" : "FAIL");

    return 0;
}
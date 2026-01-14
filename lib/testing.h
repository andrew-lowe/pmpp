#ifndef TESTING_H
#define TESTING_H

#include <stdio.h>
#include <stdlib.h>

void print_matrix(float* mat, int rows, int cols, const char* name) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int row = 0; row < rows; row++) {
        printf("  ");
        for (int col = 0; col < cols; col++) {
            printf("%6.1f ", mat[row * cols + col]);
        }
        printf("\n");
    }
}

int arrays_equal(float* a, float* b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

#endif
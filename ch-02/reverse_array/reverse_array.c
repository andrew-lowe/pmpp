#include <stdio.h>
#include <stdlib.h>

void solve(float* input, int N) {
    for (int i = 0; i < N / 2; i++) {
        int temp = input[i];
        input[i] = input[N-i-1];
        input[N-i-1] = temp;
    }
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
    // Test 1: odd length
    float test1[] = {1, 2, 3, 4, 5};
    float expected1[] = {5, 4, 3, 2, 1};
    printf("Test 1 input:    "); print_array(test1, 5);
    solve(test1, 5);
    printf("Test 1 output:   "); print_array(test1, 5);
    printf("Test 1 expected: "); print_array(expected1, 5);
    printf("Test 1: %s\n\n", arrays_equal(test1, expected1, 5) ? "PASS" : "FAIL");

    // Test 2: even length
    float test2[] = {1, 2, 3, 4, 5, 6};
    float expected2[] = {6, 5, 4, 3, 2, 1};
    printf("Test 2 input:    "); print_array(test2, 6);
    solve(test2, 6);
    printf("Test 2 output:   "); print_array(test2, 6);
    printf("Test 2 expected: "); print_array(expected2, 6);
    printf("Test 2: %s\n\n", arrays_equal(test2, expected2, 6) ? "PASS" : "FAIL");

    // Test 3: single element
    float test3[] = {42};
    float expected3[] = {42};
    solve(test3, 1);
    printf("Test 3 (single): %s\n\n", arrays_equal(test3, expected3, 1) ? "PASS" : "FAIL");

    // Test 4: two elements
    float test4[] = {1, 2};
    float expected4[] = {2, 1};
    solve(test4, 2);
    printf("Test 4 (two):    %s\n\n", arrays_equal(test4, expected4, 2) ? "PASS" : "FAIL");

    return 0;
}
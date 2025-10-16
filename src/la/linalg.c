#include "linalg.h"
#include <string.h>

// Matrix multiplication: C = A * B
// A: m x n, B: n x p, C: m x p
void matrix_multiply(const double* A, const double* B, double* C, 
                     size_t m, size_t n, size_t p) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
            C[i * p + j] = 0.0;
            for (size_t k = 0; k < n; k++) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

// Matrix addition: C = A + B
void matrix_add(const double* A, const double* B, double* C, 
                size_t rows, size_t cols) {
    size_t size = rows * cols;
    for (size_t i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}

// Matrix transpose: AT = A^T
void matrix_transpose(const double* A, double* AT, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            AT[j * rows + i] = A[i * cols + j];
        }
    }
}

// Hadamard (element-wise) multiplication: C = A ⊙ B
void matrix_hadamard(const double* A, const double* B, double* C, 
                     size_t rows, size_t cols) {
    size_t size = rows * cols;
    for (size_t i = 0; i < size; i++) {
        C[i] = A[i] * B[i];
    }
}

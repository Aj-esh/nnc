#ifndef LINALG_H
#define LINALG_H

#include <stddef.h>

// Matrix operations
void matrix_multiply(const double* A, const double* B, double* C, 
                     size_t m, size_t n, size_t p);
void matrix_add(const double* A, const double* B, double* C, 
                size_t rows, size_t cols);
void matrix_transpose(const double* A, double* AT, size_t rows, size_t cols);
void matrix_hadamard(const double* A, const double* B, double* C, 
                     size_t rows, size_t cols);

#endif // LINALG_H

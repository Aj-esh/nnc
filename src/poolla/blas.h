#ifndef BLAS_H
#define BLAS_H

#include <stddef.h>

// Basic Linear Algebra Subprograms (BLAS) like operations
void vector_add(const double* a, const double* b, double* result, size_t n);
void vector_scale(const double* a, double scalar, double* result, size_t n);
double vector_dot(const double* a, const double* b, size_t n);

#endif // BLAS_H

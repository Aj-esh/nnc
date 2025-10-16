#include "blas.h"

void vector_add(const double* a, const double* b, double* result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

void vector_scale(const double* a, double scalar, double* result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] * scalar;
    }
}

double vector_dot(const double* a, const double* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

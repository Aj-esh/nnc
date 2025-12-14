#ifndef LA_BLAS_H
#define LA_BLAS_H

#include "thread_pool.h" // Include the thread pool header
#include "la/linalg.h"
// Note: All functions now take a ThreadPool pointer as the first argument.

void dsv(ThreadPool *pool, double a, Matrix *x, double b);
void dvv(ThreadPool *pool, double a, const Matrix *A, double b, Matrix *B);
void dmv(ThreadPool *pool, double a, const Matrix *A, const Matrix *B, double b, Matrix *C);
void* dmm(ThreadPool *pool, double a, const Matrix *A, const Matrix *B, double b, Matrix *C);

#endif // LA_BLAS_H
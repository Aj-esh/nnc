#ifndef LA_BLAS_H
#define LA_BLAS_H

#include "thread_pool.h" // Include the thread pool header

// Note: All functions now take a ThreadPool pointer as the first argument.


void dsv(ThreadPool *pool, double a, Matrix *x, double b);
void dvv(ThreadPool *pool, double a, Matrix *A, double b, Matrix *B);
void dmv(ThreadPool *pool, double a, Matrix *A, Matrix *B, double b, Matrix *C);
void* dmm(ThreadPool *pool, double a, Matrix *A, Matrix *B, double b, Matrix *C);

#endif // LA_BLAS_H
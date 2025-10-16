#ifndef LA_BLAS_H
#define LA_BLAS_H

#include "thread_pool.h" // Include the thread pool header

typedef struct {
    int row, col;
    double *data;
} Matrix;

// Note: All functions now take a ThreadPool pointer as the first argument.


void dsv(ThreadPool *pool, int a, Matrix *x, int b);
void dvv(ThreadPool *pool, int a, Matrix *x, int b, Matrix *y);
void dmv(ThreadPool *pool, int a, Matrix *A, Matrix *x, int b, Matrix *y);

void dmm_task(void *args);
void* dmm(ThreadPool *pool, int a, Matrix *A, Matrix *B, int b, Matrix *C);

#endif // LA_BLAS_H
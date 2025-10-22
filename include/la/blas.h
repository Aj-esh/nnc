#ifndef LA_BLAS_H
#define LA_BLAS_H

typedef struct {
    int row, col;
    double *data;
} Matrix;

/**
    vector addition : y := a * x + b
    where a, b is scalar, x (n, 1)
    @return void (y)
*/
void dsv(int a, Matrix *x, int b);

/**
    vector-vector addition : y := a * x + b * y
    where a, b is scalar, x (n, 1) and y (n, 1)
    @return void (y)
*/
void dvv(int a, Matrix *x, int b, Matrix *y);

/**
    matrix-vector addition : y := a * A * x + b * y
    where a is scalar, x (m, n) and y (m, 1)
    @return void (y)
*/
void dmv(int a, Matrix *A, Matrix *x, int b, Matrix *y);

/**
    matrix-matrix multiplication : C := a * A * B + b * C
    where a, b is scalar, A (m, n), B (n, p) and C (m, p)
    @return C (m, p)
*/
Matrix* dmm(int a, Matrix *A, Matrix *B, int b, Matrix *C);
#endif // LA_BLAS_H

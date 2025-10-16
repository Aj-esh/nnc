#ifndef LA_LINALG_H
#define LA_LINALG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Matrix structure
typedef struct {
    int row, col;
    double* data;
} Matrix;

/**
 * Create Matrix (row, col)
 * @param row number of rows
 * @param col number of columns
 * @return pointer to created Matrix, or NULL on failure
 */
Matrix* create_matrix(int row, int col);

/**
 * Free matrix
 * @param matrix pointer to Matrix to free
 */
void free_matrix(Matrix* matrix);

/**
 * Create Matrix (row, col) from array (n,)
 * @param row number of rows
 * @param col number of columns
 * @param n size of array
 * @param array pointer to array of size n
 * @return pointer to created Matrix, or NULL on failure
 */
Matrix* matrix_from_array(int row, int col, int n, double* array);

/**
 * Print matrix to stdout
 * @param matrix pointer to Matrix to print
 */
void print_matrix(const Matrix* matrix);

/**
 * matrix multiplication C = A * B
 * @param A pointer to Matrix A
 * @param B pointer to Matrix B
 * @return pointer to Matrix C, or NULL on failure
 */
Matrix* matmul(const Matrix* A, const Matrix* B);

/**
 * matrix addition C = A + B
 * @param A pointer to Matrix A
 * @param B pointer to Matrix B
 * @return pointer to Matrix C, or NULL on failure
 */
Matrix* matadd(const Matrix* A, const Matrix* B);

/**
 * scaling matrix C = A .* B (element-wise multiplication)
 * @param A pointer to Matrix A
 * @param B pointer to Matrix B
 * @return pointer to Matrix C, or NULL on failure
 */
Matrix* matscale(const Matrix* A, const Matrix* B);

/**
 * transpose matrix At = A^T
 * @param A pointer to Matrix A
 * @return pointer to transposed Matrix At, or NULL on failure
 */
Matrix* transpose(const Matrix* A);

/**
 * Hadamard product C = A ⊙ B (element-wise multiplication)
 * @param A pointer to Matrix A
 * @param B pointer to Matrix B
 * @return pointer to Matrix C, or NULL on failure
 */
Matrix* hadamard(const Matrix* A, const Matrix* B);


#endif // LA_LINALG_H
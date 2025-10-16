#include "linalg.h"
#include "blas.h"
#include "thread_pool.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static ThreadPool* pool = NULL;

void la_init() {
    if(pool == NULL) {
        int num_threads = 4;
        const char* thread_env = getenv("NNC_NUM_THREADS");
        if(thread_env) {
            num_threads = atoi(thread_env);
        }
        pool = threadpool_init(num_threads > 0 ? num_threads : 1);
    }
}

void la_destroy() {
    if(pool) {
        threadpool_destroy(pool);
        pool = NULL;
    }
}

Matrix* create_matrix(int row, int col) {
    Matrix* matrix = (Matrix*) malloc(sizeof(Matrix));
    if(!matrix) return NULL;
    matrix->row = row;
    matrix->col = col;
    matrix->data = (double*) calloc(row * col, sizeof(double));
    if(!matrix->data) {
        free(matrix);
        return NULL;
    }
    return matrix;
}

void free_matrix(Matrix* matrix) {
    if(matrix) {
        free(matrix->data);
        free(matrix);
    }
}

Matrix* matrix_from_array(int row, int col, int n, double* array) {
    if(!array) return NULL;
    if(row * col <= 0) return NULL;
    // if array (n) != row * col, return NULL
    if(n != row * col) return NULL;
    Matrix* matrix = create_matrix(row, col);
    if(matrix) {
        memcpy(matrix->data, array, row * col * sizeof(double));
    }
    return matrix;
}

void print_matrix(const Matrix* matrix) {
    if(!matrix) {
        printf("NULL matrix\n");
        return;
    }

    for(int i = 0; i < matrix->row; i++) {
        for(int j = 0; j < matrix->col; j++) {
            printf("%8.3f ", matrix->data[i * matrix->col + j]);
        }
        printf("\n");
    }
}

Matrix* matmul(const Matrix* A, const Matrix* B) {
    assert(A->col == B->row && "matrix dim A.col != B.row");

    if(!pool) 
        la_init();
    
    Matrix* C = create_matrix(A->row, B->col);
    if(!C) {
        return NULL;
    }
    if (A->row == 0 || A->col == 0 || B->col == 0) {
        return C; // return zero matrix
    }
    if(A->row == 1 && A->col == 1 && B->row == 1 && B->col == 1) {
        C->data[0] = A->data[0] * B->data[0];
        return C;
    }
    if (B->col == 1) {
        dmv(pool, 1.0, A, B, 0.0, C);
    } else {
        dmm(pool, 1.0, A, B, 0.0, C);
    }
    return C;
}

Matrix* matadd(const Matrix* A, const Matrix* B) {
    assert(A->row == B->row && A->col == B->col && "matrix dim A != B");
    Matrix* C = create_matrix(A->row, A->col);
    if(!C) {
        return NULL;
    }
    
    if(A->col == 1) {
        if(!pool) 
            la_init();
        
        memcpy(C->data, B->data, A->row * A->col * sizeof(double));
        dvv(pool, 1.0, A, 1.0, C);
    }

    int size = A->row * A->col;
    for(int i = 0; i < size; i++) {
        C->data[i] = A->data[i] + B->data[i];
    }
    return C;
}

Matrix* matscale(const Matrix* A, const Matrix* B) {
    assert(A->row == B->row && A->col == B->col && "matrix dim A != B");
    Matrix* C = create_matrix(A->row, A->col);
    if(!C) {
        return NULL;
    }
    int size = A->row * A->col;
    for(int i = 0; i < size; i++) {
        C->data[i] = A->data[i] * B->data[i];
    }
    return C;
}

Matrix* transpose(const Matrix* A) {
    Matrix* At = create_matrix(A->col, A->row);
    if(!At) {
        return NULL;
    }
    for(int i = 0; i < A->row; i++) {
        for(int j = 0; j < A->col; j++) {
            At->data[j * At->col + i] = A->data[i * A->col + j];
        }
    }
    return At;
}

Matrix* hadamard(const Matrix* A, const Matrix* B) {
    assert(A->row == B->row && A->col == B->col && "matrix dim A != B");
    Matrix* C = create_matrix(A->row, A->col);
    if(!C) {
        return NULL;
    }
    int size = A->row * A->col;
    for(int i = 0; i < size; i++) {
        C->data[i] = A->data[i] * B->data[i];
    }
    return C;
}

// test 
/*
int main(void) {
    int row = 4, col = 3;
    double arrayA[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    };
    double arrayB[] = {
        1, 2,
        3, 4,
        5, 6
    };
    Matrix* A = matrix_from_array(row, col, arrayA);
    Matrix* B = matrix_from_array(col, 2, arrayB);
    printf("Matrix A:\n");
    print_matrix(A);
    printf("Matrix B:\n");
    print_matrix(B);
    Matrix* C = matmul(A, B);
    printf("Matrix C = A * B:\n");
    print_matrix(C);

    Matrix* At = transpose(A);
    printf("Matrix At (transpose of A):\n");
    print_matrix(At);

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    free_matrix(At);
    // free_matrix(D); // Uncomment if D is created

    return 0;
} 
*/
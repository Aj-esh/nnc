#include "la/linalg.h"
#include "poolla/blas.h"
#include "poolla/thread_pool.h"
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

ThreadPool* get_la_pool() {
    if(!pool) la_init();
    return pool;
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

// Helper for parallel tasks
typedef struct {
    const Matrix *A, *B;
    Matrix *C;
    int start, end;
} MatOpArgs;

static void matadd_task(void *arg) {
    MatOpArgs *args = (MatOpArgs*)arg;
    // Simple split by linear index for element-wise ops
    for(int i = args->start; i < args->end; i++) {
        args->C->data[i] = args->A->data[i] + args->B->data[i];
    }
    free(args);
}

Matrix* matadd(const Matrix* A, const Matrix* B) {
    assert(A->row == B->row && A->col == B->col && "matrix dim A != B");
    Matrix* C = create_matrix(A->row, A->col);
    if(!C) return NULL;
    
    ThreadPool *tp = get_la_pool();
    int total = A->row * A->col;
    int num_threads = tp->tcount;
    int chunk = (total + num_threads - 1) / num_threads;

    for(int i=0; i<num_threads; i++) {
        int start = i * chunk;
        int end = (start + chunk > total) ? total : start + chunk;
        if(start >= end) break;

        MatOpArgs *args = malloc(sizeof(MatOpArgs));
        args->A = A; args->B = B; args->C = C;
        args->start = start; args->end = end;
        threadpool_submit(tp, matadd_task, args);
    }
    threadpool_wait(tp);
    return C;
}

static void matscale_task(void *arg) {
    MatOpArgs *args = (MatOpArgs*)arg;
    for(int i = args->start; i < args->end; i++) {
        args->C->data[i] = args->A->data[i] * args->B->data[i];
    }
    free(args);
}

Matrix* matscale(const Matrix* A, const Matrix* B) {
    assert(A->row == B->row && A->col == B->col && "matrix dim A != B");
    Matrix* C = create_matrix(A->row, A->col);
    if(!C) return NULL;

    ThreadPool *tp = get_la_pool();
    int total = A->row * A->col;
    int num_threads = tp->tcount;
    int chunk = (total + num_threads - 1) / num_threads;

    for(int i=0; i<num_threads; i++) {
        int start = i * chunk;
        int end = (start + chunk > total) ? total : start + chunk;
        if(start >= end) break;

        MatOpArgs *args = malloc(sizeof(MatOpArgs));
        args->A = A; args->B = B; args->C = C;
        args->start = start; args->end = end;
        threadpool_submit(tp, matscale_task, args);
    }
    threadpool_wait(tp);
    return C;
}

static void transpose_task(void *arg) {
    MatOpArgs *args = (MatOpArgs*)arg;
    // Split by rows of A
    for(int i = args->start; i < args->end; i++) {
        for(int j = 0; j < args->A->col; j++) {
            args->C->data[j * args->C->col + i] = args->A->data[i * args->A->col + j];
        }
    }
    free(args);
}

Matrix* transpose(const Matrix* A) {
    Matrix* At = create_matrix(A->col, A->row);
    if(!At) return NULL;

    ThreadPool *tp = get_la_pool();
    int num_threads = tp->tcount;
    int chunk = (A->row + num_threads - 1) / num_threads;

    for(int i=0; i<num_threads; i++) {
        int start = i * chunk;
        int end = (start + chunk > A->row) ? A->row : start + chunk;
        if(start >= end) break;

        MatOpArgs *args = malloc(sizeof(MatOpArgs));
        args->A = A; args->C = At;
        args->start = start; args->end = end;
        threadpool_submit(tp, transpose_task, args);
    }
    threadpool_wait(tp);
    return At;
}

Matrix* hadamard(const Matrix* A, const Matrix* B) {
    // matscale logic
    return matscale(A, B);
}

static void add_bias_task(void *arg) {
    MatOpArgs *args = (MatOpArgs*)arg;
    // Split by rows of Z
    for(int i = args->start; i < args->end; i++) {
        for(int j = 0; j < args->A->col; j++) {
            args->A->data[i * args->A->col + j] += args->B->data[j];
        }
    }
    free(args);
}

void mat_add_bias(Matrix *Z, const Matrix *b) {
    ThreadPool *tp = get_la_pool();
    int num_threads = tp->tcount;
    int chunk = (Z->row + num_threads - 1) / num_threads;

    for(int i=0; i<num_threads; i++) {
        int start = i * chunk;
        int end = (start + chunk > Z->row) ? Z->row : start + chunk;
        if(start >= end) break;

        MatOpArgs *args = malloc(sizeof(MatOpArgs));
        args->A = Z; args->B = b;
        args->start = start; args->end = end;
        threadpool_submit(tp, add_bias_task, args);
    }
    threadpool_wait(tp);
}

static void sum_rows_task(void *arg) {
    MatOpArgs *args = (MatOpArgs*)arg;
    // Split by columns (output size)
    for(int j = args->start; j < args->end; j++) {
        double sum = 0.0;
        for(int i = 0; i < args->A->row; i++) {
            sum += args->A->data[i * args->A->col + j];
        }
        args->C->data[j] += sum;
    }
    free(args);
}

Matrix* mat_sum_rows(const Matrix *dA) {
    Matrix *db = create_matrix(1, dA->col);
    ThreadPool *tp = get_la_pool();
    int num_threads = tp->tcount;
    int chunk = (dA->col + num_threads - 1) / num_threads;

    for(int i=0; i<num_threads; i++) {
        int start = i * chunk;
        int end = (start + chunk > dA->col) ? dA->col : start + chunk;
        if(start >= end) break;

        MatOpArgs *args = malloc(sizeof(MatOpArgs));
        args->A = dA; args->C = db;
        args->start = start; args->end = end;
        threadpool_submit(tp, sum_rows_task, args);
    }
    threadpool_wait(tp);
    return db;
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
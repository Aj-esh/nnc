#include <stdlib.h>
#include <stdio.h>
#include "poolla/blas.h"
#include "la/linalg.h"

typedef struct {
    int start_row, end_row;
    double a, b;
    Matrix *x; //  x (n, 1)
} dsv_args;

void dsv_task(void *args) {
    dsv_args *data = (dsv_args*) args;
    
    for(int i=data->start_row; i<data->end_row; i++) {
        data->x->data[i] = data->a * data->x->data[i] + data->b;
    }
    free(data);
}

void dsv(ThreadPool *pool, double a, Matrix *x, double b) {
    /**
     * x := a * x + b
     * @param pool ThreadPool to use for parallelism
     * @param a Scalar multiplier
     * @param x Vector to be scaled and shifted
     * @param b Scalar to be added
     */
    if(x->col != 1) {
        fprintf(stderr, "dsv expects x to be a column vector (n,1)\n");
        exit(EXIT_FAILURE);
    }
    
    int total = x->row;
    int num_threads = pool->tcount > 0 ? pool->tcount : 1;
    int chunk = (total + num_threads - 1) / num_threads;

    for(int i=0; i<num_threads; i++) {
        int start = i * chunk;
        int end = imin(start + chunk, total);
        if(start >= end) 
            break;

        dsv_args *args = malloc(sizeof(dsv_args));
        if(!args) {
            perror("Failed to allocate memory for dsv_args");
            exit(EXIT_FAILURE);
        }

        args->start_row = start;
        args->end_row = end;
        args->a = a;
        args->x = x;
        args->b = b;

        threadpool_submit(pool, dsv_task, args);
    }
    threadpool_wait(pool);
}

typedef struct {
    int start_row, end_row;
    double a, b;
    Matrix *A, *B; // A (n, 1), B (n, 1)
} dvv_args;

void dvv_task(void *args) {
    dvv_args *data = (dvv_args*) args;
    
    for(int i=data->start_row; i<data->end_row; i++) {
        data->B->data[i] = data->a * data->A->data[i] + data->b * data->B->data[i];
    }
    free(data);
}

void dvv(ThreadPool *pool, double a, Matrix *A, double b, Matrix *B) {
    /**
     * B := a * A + b * B
     * @param pool ThreadPool to use for parallelism
     * @param a Scalar multiplier for A
     * @param A First vector (n, 1)
     * @param b Scalar multiplier for B
     * @param B Second vector to be updated (n, 1)
     */
    if(A->col != 1 || B->col != 1 || A->row != B->row) {
        fprintf(stderr, "Matrix dimensions do not match for vector-vector operation\n");
        exit(EXIT_FAILURE);
    }

    int total = A->row;
    int num_threads = pool->tcount > 0 ? pool->tcount : 1;
    int chunk = (total + num_threads - 1) / num_threads;

    for(int i=0; i<num_threads; i++) {
        int start = i * chunk;
        int end = imin(start + chunk, total);
        if(start >= end) 
            break;

        dvv_args *args = malloc(sizeof(dvv_args));
        if(!args) {
            perror("Failed to allocate memory for dvv_args");
            exit(EXIT_FAILURE);
        }

        args->start_row = start;
        args->end_row = end;

        args->a = a;
        args->A = A;
        args->b = b;
        args->B = B;

        threadpool_submit(pool, dvv_task, args);
    }
    threadpool_wait(pool);
}

typedef struct {
    int start_row, end_row;
    double a, b;
    Matrix *A, *B, *C; // A (n, m), B (m, 1), C (n, 1)
} dmv_args;

void dmv_task(void *args) {
    dmv_args *data = (dmv_args*) args;

    for(int i=data->start_row; i<data->end_row; i++) {
        double sum = 0.0;
        for(int j=0; j<data->A->col; j++) {
            sum += data->A->data[i * data->A->col + j] * data->B->data[j];
        }
        data->C->data[i] = data->a * sum + data->b * data->C->data[i];
    }
    free(data);
}

void dmv(ThreadPool *pool, double a, Matrix *A, Matrix *B, double b, Matrix *C) {
    /**
     * C := a * A * B + b * C
     *
     * @param pool ThreadPool to use for parallelism
     * @param a Scalar multiplier for A*B
     * @param A Left matrix (n, m)
     * @param B Right vector (m, 1) and result vector (n, 1)
     * @param b Scalar multiplier for C (result vector)
     * @param C Result vector (n, 1)    
     */
    if(A->col != B->row || B->col != 1 || C->col != 1 || C->row != A->row) {
        fprintf(stderr, "Matrix dimensions do not match for matrix-vector multiplication\n");
        exit(EXIT_FAILURE);
    }

    int total = A->row;
    int num_threads = pool->tcount > 0 ? pool->tcount : 1;
    int chunk = (total + num_threads - 1) / num_threads;

    for(int i=0; i<num_threads; i++) {
        int start = i * chunk;
        int end = imin(start + chunk, total);
        if(start >= end) 
            break;

        dmv_args *args = malloc(sizeof(dmv_args));
        if(!args) {
            perror("Failed to allocate memory for dmv_args");
            exit(EXIT_FAILURE);
        }

        args->start_row = start;
        args->end_row = end;

        args->a = a;
        args->A = A;
        args->b = b;
        args->B = B;
        args->C = C;
        
        threadpool_submit(pool, dmv_task, args);
    }
    threadpool_wait(pool);
}

typedef struct {
    int start_row, end_row;
    double a, b;
    Matrix *A, *B, *C;
} dmm_args;

void dmm_task(void *args) {
    dmm_args *data = (dmm_args*) args;

    for(int i=data->start_row; i<data->end_row; i++) {
        for(int j=0; j<data->B->col; j++) {
            double sum = 0.0;
            for(int k=0; k<data->A->col; k++) {
                sum += data->A->data[i * data->A->col + k] * data->B->data[k * data->B->col + j];
            }
            size_t idx = (size_t) i * data->C->col + j;
            data->C->data[idx] = data->a * sum + data->b * data->C->data[idx];
        }
    }
    free(data);
}

void *dmm(ThreadPool *pool, double a, Matrix *A, Matrix *B, double b, Matrix *C) {
    /**
     * C := a * A * B + b * C
     *
     * @param pool ThreadPool to use for parallelism
     * @param a Scalar multiplier for A*B
     * @param A Left matrix
     * @param B Right matrix
     * @param b Scalar multiplier for C
     * @param C Result matrix
     */
    // check dimensions
    if(A->col != B->row || A->row != C->row || B->col != C->col) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication\n");
        exit(EXIT_FAILURE);
    }
    
    int total = A->row;
    int num_threads = pool->tcount > 0 ? pool->tcount : 1;
    int chunk = (total + num_threads - 1) / num_threads;

    for(int i=0; i<num_threads; i++) {
        int start = i * chunk;
        int end = imin(start + chunk, total);
        if(start >= end) 
            break;
        dmm_args *args = malloc(sizeof(dmm_args));
        if (!args) {
            perror("Failed to allocate memory for dmm_args");
            exit(EXIT_FAILURE);
        }

        args->start_row = start;
        args->end_row = end;

        args->a = a;
        args->b = b;
        args->A = A;
        args->B = B;
        args->C = C;
        threadpool_submit(pool, dmm_task, args);
    }
    threadpool_wait(pool);
    return NULL;
}


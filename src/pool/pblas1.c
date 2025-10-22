// Basic Linear Algebra subprograms - BLAS
#include "blas.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h> // For sysconf
#include <pthread.h>

// Global constant for number of threads
/**
 * @brief Number of threads to use for parallel computations
 * Default is set to the number of CPU cores available
 * @return int (num_threads)
 */
static int get_nthreads () {
    static int nthreads = 0;
    if(nthreads == 0) {
        nthreads = sysconf(_SC_NPROCESSORS_ONLN);
        if(nthreads < 1) nthreads = 1; // Fallback to 1 thread if detection fails
    }
    return nthreads;
}

/**
    vector addition : x := a * x + b
    @param a : scalar
    @param x : vector (n, 1)
    @param b : scalar
    @return void (x)
*/
typedef struct {
    int start_idx, end_idx;
    double a, b;
    Matrix *X; // (n, 1)
} dsv_args;

void* dsv_worker(void *args) {
    dsv_args* data = (dsv_args*)args;
    for(int i = data->start_idx; i<data->end_idx; i++) {
        data->X->data[i] = data->a * data->X->data[i] + data->b;
    }
    pthread_exit(NULL);
}

void dsv(int a, Matrix *x, int b) {
    const int nthreads = get_nthreads();
    pthread_t threads[nthreads];
    dsv_args thread_args[nthreads];

    int N = x->row * x->col;
    int chunk = N / nthreads;

    for(int i=0; i<nthreads; i++) {
        thread_args[i].a = a;
        thread_args[i].X = x;
        thread_args[i].b = b;
        thread_args[i].start_idx = i * chunk;
        thread_args[i].end_idx = (i == nthreads - 1) ? N : (i + 1) * chunk;

        pthread_create(&threads[i], NULL, dsv_worker, &thread_args[i]);
    }

    for(int i=0; i<nthreads; i++) {
        pthread_join(threads[i], NULL);
    }
}

/**
 * @brief vector vector addititon : y := a * x + b * y
 * @param a : scalar
 * @param x : vector (n, 1)
 * @param b : scalar
 * @param y : vector (n, 1)
 * @return void (y)
 */
typedef struct {
    int start_idx, end_idx;
    double a, b;
    Matrix *X; // (n, 1)
    Matrix *Y; // (n, 1)
} dvv_args;

void *dvv_worker(void *args) {
    dvv_args* data = (dvv_args*) args;
    for(int i=data->start_idx; i<data->end_idx; i++) {
        data->Y->data[i] = data->a * data->X->data[i] + data->b * data->Y->data[i];
    } 
    pthread_exit(NULL);
}

void dvv(double a, Matrix *x, double b, Matrix *y) {
    // Basic checks, vector-vector addition
    if((x->row != y->row) || (x->col != y->col) || (x->col !=-1)) {
        fprintf(stderr, "Error : dvv : incompatible matrix dimensions\n");
        return;
    }

    const int nthreads = get_nthreads();
    pthread_t threads[nthreads];
    dvv_args threads_args[nthreads];

    int N = y->col * y->row;
    int chunk = N / nthreads;

    for(int i=0; i<nthreads; i++) {
        threads_args[i].a = a;
        threads_args[i].b = b;
        threads_args[i].X = x;
        threads_args[i].Y = y;
        threads_args[i].start_idx = i * chunk;
        threads_args[i].end_idx = (i == nthreads - 1) ? N : (i + 1) * chunk;

        pthread_create(&threads[i], NULL, dvv_worker, &threads_args[i]);
    }

    for(int i=0; i<nthreads; i++) {
        pthread_join(threads[i], NULL);
    }
}

/**
 * @brief matrix-vector multiplication : y := s1 * A * x + s2 * y
 * @param s1 : scalar
 * @param A : matrix (m, n)
 * @param x : vector (n, 1)
 * @param s2 : scalar
 * @param y : vector (m, 1)
 * @return void (y)
 */
typedef struct {
    int start_row, end_row;
    double s1, s2;
    Matrix *A, *x, *y; // A (m, n), x (n, 1), y (m, 1)
} dmv_args;

void *dmv_worker(void *args) {
    dmv_args* data = (dmv_args*) args;
    for(int i=data->start_row; i<data->end_row; i++) {
        double sum = 0.0;
        for(int j=0; j<data->A->col; j++) {
            size_t idx = i * data->A->col + j;
            sum += data->A->data[idx] * data->x->data[j];
        }
        data->y->data[i] = data->s1 * sum + data->s2 * data->y->data[i];
    } 
    pthread_exit(NULL);
}

void dmv(double s1, Matrix *A, Matrix *x, double s2, Matrix *y) {
    if(A->col != x-> row) {
        fprintf(stderr, "Error : dmv : mv, %d != %d\n", A->col, x->row);
    }
    if(A->row != y->row) {
        fprintf(stderr, "Error : dmv : my, %d != %d\n", A->row, y->row);
    }

    if(x->col != 1 || y->col != 1) {
        fprintf(stderr, "Error : dmv : x or y not a vector\n");
        return;
    }

    const int nthreads = get_nthreads();
    pthread_t threads[nthreads];
    dmv_args targs[nthreads];

    int chunk = A->row / nthreads;
    for(int i=0; i<nthreads; i++) {
        targs[i] = (dmv_args) {
            .start_row = i * chunk,
            .end_row = (i == nthreads - 1) ? A->row : (i + 1) * chunk,
            .s1 = s1,
            .A = A,
            .x = x,
            .s2 = s2,
            .y = y
        };
        pthread_create(&threads[i], NULL, dmv_worker, &targs[i]);
    }

    for(int i=0; i<nthreads; i++) {
        pthread_join(threads[i], NULL);
    }
}

/**
 * @brief matrix-matrix multiplication : C := s1 * A * B + s2 * C
 * @param s1 : scalar
 * @param A : matrix (m, k)
 * @param B : matrix (k, n)
 * @param s2 : scalar
 * @param C : matrix (m, n)
 * @
 * @return void (C)
 */
typedef struct {
    int start_row, end_row;
    double s1, s2;
    Matrix *A, *B, *C; // A (m, k), B (k, n), C (m, n)
} dmm_args;

void *dmm_worker(void *args) {
    dmm_args* data = (dmm_args*) args;
    const TILE = 32;

    for(int i=data->start_row; i<data->end_row; i+=TILE) {
        for(int j=0; j<data->C->col; j+=TILE) {
            for(int k=0; k<data->A->col; k+=TILE) {
                //process tiles
                for(int ii=i; ii < (i + TILE) && ii < data->end_row; ii++) {
                    for(int jj=j; jj<(j+TILE) && jj < data->C->col; jj++) {
                        double sum = 0.0;
                        for(int kk=k; kk<(k + TILE) && kk < data->A->col; kk++) {
                            sum += data->A->data[ii * data->A->col + kk] * data->B->data[kk*data->B->col+jj];
                        }
                        data->C->data[ii*data->C->col+jj] = data->s1 * sum + data->s2 * data->C->data[ii*data->C->col+jj];
                    }
                }
            }
        }
    }
    pthread_exit(NULL);
}
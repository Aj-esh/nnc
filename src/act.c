#include <math.h>
#include <stdlib.h>
#include "act.h"

typedef struct {
    const Matrix* Z;
    const Matrix* dZ; // For backward
    Matrix* A;
    int s, e;
} act_args;

static void relu_task(void *args) {
    act_args *a = (act_args*) args;
    for(int i=a->s; i < a->e; i++) {
        a->A->data[i] = fmax(0.0, a->Z->data[i]);
    }
    free(a);
}

Matrix *relu(const Matrix *Z) {
    Matrix *A = create_matrix(Z->row, Z->col);
    if(!A) return NULL;    

    ThreadPool *tp = get_la_pool();
    int total = Z->row * Z->col;
    int num_threads = tp->tcount;
    int chunk = (total + num_threads - 1) / num_threads;

    for(int i=0; i<num_threads; i++) {
        int start = i * chunk;
        int end = (start + chunk > total) ? total : start + chunk;
        if(start >= end) break;

        act_args *args = malloc(sizeof(act_args));
        args->Z = Z; args->A = A;
        args->s = start; args->e = end;
        threadpool_submit(tp, relu_task, args);
    }
    threadpool_wait(tp);
    return A;
}

static void drelu_task(void *args) {
    act_args *a = (act_args*) args;
    for(int i=a->s; i < a->e; i++) {
        a->A->data[i] = (a->Z->data[i] > 0.0) ? a->dZ->data[i] : 0.0;
    }
    free(a);
}

Matrix* drelu(const Matrix *Z, const Matrix *dZ) {
    Matrix *dA = create_matrix(Z->row, Z->col);
    if(!dA) return NULL;

    ThreadPool *tp = get_la_pool();
    int total = Z->row * Z->col;
    int num_threads = tp->tcount;
    int chunk = (total + num_threads - 1) / num_threads;

    for(int i=0; i<num_threads; i++) {
        int start = i * chunk;
        int end = (start + chunk > total) ? total : start + chunk;
        if(start >= end) break;

        act_args *args = malloc(sizeof(act_args));
        args->Z = Z; args->dZ = dZ; args->A = dA;
        args->s = start; args->e = end;
        threadpool_submit(tp, drelu_task, args);
    }
    threadpool_wait(tp);
    return dA;
}
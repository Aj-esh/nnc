#include <math.h>
#include <stdlib.h>
#include "optax.h"

typedef struct {
    Matrix *W, *dW;
    AdamState *st;
    double lr;
    int start, end;
} OptArgs;

static void sgd_task(void *arg) {
    OptArgs *a = (OptArgs*)arg;
    for(int i=a->start; i<a->end; i++) {
        a->W->data[i] -= a->lr * a->dW->data[i];
    }
    free(a);
}

void sgd(Matrix *W, Matrix *dW, double lr) {
    ThreadPool *tp = get_la_pool();
    int total = W->row * W->col;
    int num_threads = tp->tcount;
    int chunk = (total + num_threads - 1) / num_threads;

    for(int i=0; i<num_threads; i++) {
        int start = i * chunk;
        int end = (start + chunk > total) ? total : start + chunk;
        if(start >= end) break;

        OptArgs *args = malloc(sizeof(OptArgs));
        args->W = W; args->dW = dW; args->lr = lr;
        args->start = start; args->end = end;
        threadpool_submit(tp, sgd_task, args);
    }
    threadpool_wait(tp);
}

AdamState* adam_init (const Matrix *W, double b1, double b2, double eps) {
    /**
     * Initialize Adam optimizer state
     * @param W Weights matrix
     * @param b1 Decay rate for first moment
     * @param b2 Decay rate for second moment
     * @param eps Small constant for numerical stability
     * @return Pointer to initialized AdamState
     */
    AdamState *st = malloc(sizeof(AdamState));
    if(!st) 
        return NULL;

    st->m = create_matrix(W->row, W->col);
    st->v = create_matrix(W->row, W->col);
    if(!st->m || !st->v) {
        adam_free(st);
        return NULL;
    }

    st->b1 = b1;
    st->b2 = b2;
    st->eps = eps;
    st->t = 0;
    return st;
}

static void adam_task(void *arg) {
    OptArgs *a = (OptArgs*)arg;
    AdamState *st = a->st;
    double corr1 = 1.0 - pow(st->b1, st->t);
    double corr2 = 1.0 - pow(st->b2, st->t);

    for(int i=a->start; i<a->end; i++) {
        double g = a->dW->data[i];
        st->m->data[i] = st->b1 * st->m->data[i] + (1.0 - st->b1) * g;     
        st->v->data[i] = st->b2 * st->v->data[i] + (1.0 - st->b2) * g * g; 

        double mh = st->m->data[i] / corr1;
        double vh = st->v->data[i] / corr2;

        a->W->data[i] -= a->lr * mh / (sqrt(vh) + st->eps);
    }
    free(a);
}

void adam (Matrix *W, Matrix *dW, AdamState *st, double lr) {
    st->t++;
    ThreadPool *tp = get_la_pool();
    int total = W->row * W->col;
    int num_threads = tp->tcount;
    int chunk = (total + num_threads - 1) / num_threads;

    for(int i=0; i<num_threads; i++) {
        int start = i * chunk;
        int end = (start + chunk > total) ? total : start + chunk;
        if(start >= end) break;

        OptArgs *args = malloc(sizeof(OptArgs));
        args->W = W; args->dW = dW; args->st = st; args->lr = lr;
        args->start = start; args->end = end;
        threadpool_submit(tp, adam_task, args);
    }
    threadpool_wait(tp);
}

void adam_free(AdamState *st) {
    /**
     * Free Adam optimizer state
     * @param st Pointer to AdamState to be freed
     */
    if(st) {
        free_matrix(st->m);
        free_matrix(st->v);
        free(st);
    }
}
#ifndef OPTAX_H
#define OPTAX_H

#include "la/linalg.h"

typedef struct {
    Matrix *m, *v; // first and second moment vectors
    double b1, b2, eps; // decay rates and epsilon
    int t; // time step
} AdamState;

void sgd(Matrix *W, Matrix *dW, double lr);

AdamState* adam_init (const Matrix *W, double b1, double b2, double eps);

void adam (Matrix *W, Matrix *dW, AdamState *state, double lr);
void adam_free(AdamState *state);

#endif // OPTAX_H
#ifndef NN_H
#define NN_H

#include "la/linalg.h"
#include "optax.h"
#include "act.h"

typedef struct {
    Matrix *W1, *b1;
    Matrix *W2, *b2;
    Matrix *W3, *b3;
    Matrix *W4, *b4;
} NN;

typedef struct {
    Matrix *Z1, *A1;
    Matrix *Z2, *A2;
    Matrix *Z3, *A3;
    Matrix *Z4, *A4;
} Cache;

typedef struct {
    Matrix *dW1, *db1;
    Matrix *dW2, *db2;
    Matrix *dW3, *db3;
    Matrix *dW4, *db4;
} Grad;

// Initialization
NN* net_create(int input, int hidden1, int hidden2, int hidden3, int output);
void net_free(NN *net);

// Forward Pass
Cache* forward(NN *net, const Matrix *X);
void cache_free(Cache *cache);

// Loss
double mse(const Matrix *Y_pred, const Matrix *Y_true);

// Backward Pass
Grad* backward(NN *net, const Matrix *X, const Matrix *Y_true, Cache *cache);
void grad_free(Grad *grads);

// Update
void sgd_update(NN *net, Grad *grads, double lr);

#endif // NN_H
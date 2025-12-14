#include "nn.h"
#include "la/normal.h"
#include <stdlib.h>
#include <stdio.h>

NN* net_create(int input, int hidden1, int hidden2, int hidden3, int output) {
    NN *net = malloc(sizeof(NN));
    la_init();

    net->W1 = He_init(input, hidden1, input);
    net->b1 = create_matrix(1, hidden1);
    
    net->W2 = He_init(hidden1, hidden2, hidden1);
    net->b2 = create_matrix(1, hidden2);

    net->W3 = He_init(hidden2, hidden3, hidden2);
    net->b3 = create_matrix(1, hidden3);

    net->W4 = He_init(hidden3, output, hidden3);
    net->b4 = create_matrix(1, output);

    return net;
}

void net_free(NN *net) {
    if(!net) return;
    free_matrix(net->W1); free_matrix(net->b1);
    free_matrix(net->W2); free_matrix(net->b2);
    free_matrix(net->W3); free_matrix(net->b3);
    free_matrix(net->W4); free_matrix(net->b4);
    free(net);
}

Cache* forward(NN *net, const Matrix *X) {
    Cache *c = malloc(sizeof(Cache));
    
    // Layer 1
    c->Z1 = matmul(X, net->W1);
    mat_add_bias(c->Z1, net->b1);
    c->A1 = relu(c->Z1);

    // Layer 2
    c->Z2 = matmul(c->A1, net->W2);
    mat_add_bias(c->Z2, net->b2);
    c->A2 = relu(c->Z2);

    // Layer 3
    c->Z3 = matmul(c->A2, net->W3);
    mat_add_bias(c->Z3, net->b3);
    c->A3 = relu(c->Z3);

    // Layer 4 (Output) - Linear activation
    c->Z4 = matmul(c->A3, net->W4);
    mat_add_bias(c->Z4, net->b4);
    c->A4 = create_matrix(c->Z4->row, c->Z4->col);
    for(int i = 0; i < c->Z4->row * c->Z4->col; i++) {
        c->A4->data[i] = c->Z4->data[i];
    }

    return c;
}

void cache_free(Cache *c) {
    if(!c) return;
    free_matrix(c->Z1); free_matrix(c->A1);
    free_matrix(c->Z2); free_matrix(c->A2);
    free_matrix(c->Z3); free_matrix(c->A3);
    free_matrix(c->Z4); free_matrix(c->A4);
    free(c);
}

double mse(const Matrix *Y_pred, const Matrix *Y_true) {
    double loss = 0.0;
    int size = Y_pred->row * Y_pred->col;
    for(int i=0; i<size; i++) {
        double diff = Y_pred->data[i] - Y_true->data[i];
        loss += diff * diff;
    }
    return loss / size;
}

Grad* backward(NN *net, const Matrix *X, const Matrix *Y_true, Cache *c) {
    Grad *g = malloc(sizeof(Grad));
    int batch_size = X->row;
    double scale = 2.0 / (batch_size * Y_true->col);

    // 1. Output Layer Gradients (Linear activation: dZ4 = dL/dA4)
    Matrix *dZ4 = create_matrix(c->A4->row, c->A4->col);
    for(int i=0; i<dZ4->row * dZ4->col; i++) {
        dZ4->data[i] = scale * (c->A4->data[i] - Y_true->data[i]);
    }

    Matrix *A3_T = transpose(c->A3);
    g->dW4 = matmul(A3_T, dZ4);
    g->db4 = mat_sum_rows(dZ4);
    free_matrix(A3_T);

    // 2. Hidden Layer 3 Gradients
    Matrix *W4_T = transpose(net->W4);
    Matrix *dA3 = matmul(dZ4, W4_T);
    free_matrix(W4_T);
    free_matrix(dZ4);

    Matrix *dZ3 = drelu(c->Z3, dA3);
    free_matrix(dA3);

    Matrix *A2_T = transpose(c->A2);
    g->dW3 = matmul(A2_T, dZ3);
    g->db3 = mat_sum_rows(dZ3);
    free_matrix(A2_T);

    // 3. Hidden Layer 2 Gradients
    Matrix *W3_T = transpose(net->W3);
    Matrix *dA2 = matmul(dZ3, W3_T);
    free_matrix(W3_T);
    free_matrix(dZ3);

    Matrix *dZ2 = drelu(c->Z2, dA2);
    free_matrix(dA2);

    Matrix *A1_T = transpose(c->A1);
    g->dW2 = matmul(A1_T, dZ2);
    g->db2 = mat_sum_rows(dZ2);
    free_matrix(A1_T);

    // 4. Hidden Layer 1 Gradients
    Matrix *W2_T = transpose(net->W2);
    Matrix *dA1 = matmul(dZ2, W2_T);
    free_matrix(W2_T);
    free_matrix(dZ2);

    Matrix *dZ1 = drelu(c->Z1, dA1);
    free_matrix(dA1);

    Matrix *X_T = transpose(X);
    g->dW1 = matmul(X_T, dZ1);
    g->db1 = mat_sum_rows(dZ1);
    
    free_matrix(X_T);
    free_matrix(dZ1);

    return g;
}

void grad_free(Grad *g) {
    if(!g) return;
    free_matrix(g->dW1); free_matrix(g->db1);
    free_matrix(g->dW2); free_matrix(g->db2);
    free_matrix(g->dW3); free_matrix(g->db3);
    free_matrix(g->dW4); free_matrix(g->db4);
    free(g);
}

void sgd_update(NN *net, Grad *g, double lr) {
    sgd(net->W1, g->dW1, lr);
    sgd(net->b1, g->db1, lr);
    sgd(net->W2, g->dW2, lr);
    sgd(net->b2, g->db2, lr);
    sgd(net->W3, g->dW3, lr);
    sgd(net->b3, g->db3, lr);
    sgd(net->W4, g->dW4, lr);
    sgd(net->b4, g->db4, lr);
}


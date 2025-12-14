#include "val.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

ValResult validate(NN *net, const Matrix *X_val, const Matrix *Y_val) {
    ValResult result;
    
    // Forward pass
    Cache *cache = forward(net, X_val);
    
    // Compute metrics
    result.loss = mse(cache->A4, Y_val);
    result.rmse = compute_rmse(result.loss);
    result.r_squared = compute_r_squared(cache->A4, Y_val);
    
    // Cleanup
    cache_free(cache);
    
    return result;
}

void train_val_split(const Matrix *X, const Matrix *Y,
                     Matrix **X_train, Matrix **Y_train,
                     Matrix **X_val, Matrix **Y_val,
                     double val_ratio) {
    int n_samples = X->row;
    int n_features = X->col;
    int n_outputs = Y->col;
    
    int n_val = (int)(n_samples * val_ratio);
    int n_train = n_samples - n_val;
    
    // Allocate matrices
    *X_train = create_matrix(n_train, n_features);
    *Y_train = create_matrix(n_train, n_outputs);
    *X_val = create_matrix(n_val, n_features);
    *Y_val = create_matrix(n_val, n_outputs);
    
    // Copy training data
    memcpy((*X_train)->data, X->data, n_train * n_features * sizeof(double));
    memcpy((*Y_train)->data, Y->data, n_train * n_outputs * sizeof(double));
    
    // Copy validation data
    memcpy((*X_val)->data, X->data + n_train * n_features, n_val * n_features * sizeof(double));
    memcpy((*Y_val)->data, Y->data + n_train * n_outputs, n_val * n_outputs * sizeof(double));
}

void print_val_result(const ValResult *result, int epoch) {
    printf("  Val   - Loss: %.6f, RMSE: %.6f, R²: %.6f\n", 
           result->loss, result->rmse, result->r_squared);
}

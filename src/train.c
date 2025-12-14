#include "train.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

MetricList* metrics_init(void) {
    MetricList *list = malloc(sizeof(MetricList));
    if (!list) return NULL;
    list->head = NULL;
    list->tail = NULL;
    list->count = 0;
    return list;
}

void metrics_append(MetricList *list, int epoch, double loss, double rmse, double r_squared) {
    if (!list) return;
    
    MetricNode *node = malloc(sizeof(MetricNode));
    if (!node) return;
    
    node->epoch = epoch;
    node->loss = loss;
    node->rmse = rmse;
    node->r_squared = r_squared;
    node->next = NULL;
    
    if (list->tail == NULL) {
        list->head = node;
        list->tail = node;
    } else {
        list->tail->next = node;
        list->tail = node;
    }
    list->count++;
}

void metrics_free(MetricList *list) {
    if (!list) return;
    
    MetricNode *current = list->head;
    while (current) {
        MetricNode *next = current->next;
        free(current);
        current = next;
    }
    free(list);
}

void metrics_print(const MetricList *list, const char *label) {
    if (!list) return;
    
    printf("\n=== %s Metrics History ===\n", label);
    printf("%-8s %-12s %-12s %-12s\n", "Epoch", "Loss", "RMSE", "R²");
    printf("----------------------------------------\n");
    
    MetricNode *node = list->head;
    while (node) {
        printf("%-8d %-12.6f %-12.6f %-12.6f\n", 
               node->epoch, node->loss, node->rmse, node->r_squared);
        node = node->next;
    }
    printf("\n");
}

double compute_rmse(double mse_loss) {
    return sqrt(mse_loss);
}

double compute_r_squared(const Matrix *y_pred, const Matrix *y_true) {
    int n = y_true->row * y_true->col;
    
    // Compute mean of y_true
    double mean = 0.0;
    for (int i = 0; i < n; i++) {
        mean += y_true->data[i];
    }
    mean /= n;
    
    // Compute residual sum of squares and total sum of squares
    double ss_res = 0.0;
    double ss_tot = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = y_pred->data[i] - y_true->data[i];
        ss_res += diff * diff;
        double diff_mean = y_true->data[i] - mean;
        ss_tot += diff_mean * diff_mean;
    }
    
    // R2 = 1 - SS_res / SS_tot
    if (ss_tot == 0.0) return 0.0;
    return 1.0 - (ss_res / ss_tot);
}

TrainResult train_epoch(NN *net, const Matrix *X_train, const Matrix *y_train, double lr) {
    TrainResult result;
    
    // Forward pass
    Cache *cache = forward(net, X_train);
    
    // Compute loss and metrics
    result.loss = mse(cache->A4, y_train);
    result.rmse = compute_rmse(result.loss);
    result.r_squared = compute_r_squared(cache->A4, y_train);
    
    // Backward pass
    Grad *grads = backward(net, X_train, y_train, cache);
    
    // Update weights
    sgd_update(net, grads, lr);
    
    // Cleanup
    cache_free(cache);
    grad_free(grads);
    
    return result;
}

void generate_synthetic_data(Matrix **X, Matrix **Y, int n_samples, int n_features) {
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    
    *X = create_matrix(n_samples, n_features);
    *Y = create_matrix(n_samples, 1);
    
    // Generate random weights for the true function
    double *true_weights = malloc(n_features * sizeof(double));
    for (int j = 0; j < n_features; j++) {
        true_weights[j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    double bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    
    // Generate X and Y = X @ true_weights + bias + noise
    for (int i = 0; i < n_samples; i++) {
        double y_val = bias;
        for (int j = 0; j < n_features; j++) {
            double x_val = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            (*X)->data[i * n_features + j] = x_val;
            y_val += x_val * true_weights[j];
        }
        // Add small noise
        double noise = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        (*Y)->data[i] = y_val + noise;
    }
    
    free(true_weights);
}

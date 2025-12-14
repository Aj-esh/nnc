#ifndef TRAIN_H
#define TRAIN_H

#include "nn.h"

// Linked list node for storing metrics
typedef struct MetricNode {
    int epoch;
    double loss;
    double rmse;
    double r_squared;
    struct MetricNode *next;
} MetricNode;

// Linked list for metrics history
typedef struct {
    MetricNode *head;
    MetricNode *tail;
    int count;
} MetricList;

// Training result for one epoch
typedef struct {
    double loss;
    double rmse;
    double r_squared;
} TrainResult;

/**
 * Initialize a new metrics list
 * @return pointer to MetricList
 */
MetricList* metrics_init(void);

/**
 * Append a metric to the list
 * @param list pointer to MetricList
 * @param epoch epoch number
 * @param loss loss value
 * @param rmse RMSE value
 * @param r_squared R-squared value
 */
void metrics_append(MetricList *list, int epoch, double loss, double rmse, double r_squared);

/**
 * Free all nodes in the metrics list
 * @param list pointer to MetricList
 */
void metrics_free(MetricList *list);

/**
 * Print all metrics in the list
 * @param list pointer to MetricList
 * @param label label for the list (e.g., "Train" or "Val")
 */
void metrics_print(const MetricList *list, const char *label);

/**
 * Train for one epoch
 * @param net pointer to neural network
 * @param X_train training input data
 * @param Y_train training target data
 * @param lr learning rate
 * @return TrainResult with loss and metrics
 */
TrainResult train_epoch(NN *net, const Matrix *X_train, const Matrix *Y_train, double lr);

/**
 * Compute RMSE from MSE loss
 * @param mse_loss MSE loss value
 * @return RMSE value
 */
double compute_rmse(double mse_loss);

/**
 * Compute R-squared (coefficient of determination)
 * @param Y_pred predicted values
 * @param Y_true true values
 * @return R-squared value
 */
double compute_r_squared(const Matrix *Y_pred, const Matrix *Y_true);

/**
 * Generate synthetic regression data for testing
 * @param X output pointer for input matrix
 * @param Y output pointer for target matrix
 * @param n_samples number of samples
 * @param n_features number of features
 */
void generate_synthetic_data(Matrix **X, Matrix **Y, int n_samples, int n_features);

#endif // TRAIN_H

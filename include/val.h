#ifndef VAL_H
#define VAL_H

#include "nn.h"
#include "train.h"

// Validation result structure
typedef struct {
    double loss;
    double rmse;
    double r_squared;
} ValResult;

/**
 * Validate the model on validation data
 * @param net pointer to neural network
 * @param X_val validation input data
 * @param Y_val validation target data
 * @return ValResult with loss and metrics
 */
ValResult validate(NN *net, const Matrix *X_val, const Matrix *Y_val);

/**
 * Split data into train and validation sets
 * @param X input data
 * @param Y target data
 * @param X_train output pointer for training input
 * @param Y_train output pointer for training target
 * @param X_val output pointer for validation input
 * @param Y_val output pointer for validation target
 * @param val_ratio ratio of data to use for validation (0.0 to 1.0)
 */
void train_val_split(const Matrix *X, const Matrix *Y,
                     Matrix **X_train, Matrix **Y_train,
                     Matrix **X_val, Matrix **Y_val,
                     double val_ratio);

/**
 * Print validation results
 * @param result validation result
 * @param epoch current epoch
 */
void print_val_result(const ValResult *result, int epoch);

#endif // VAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn.h"
#include "train.h"
#include "val.h"
#include "data.h"

#define HIDDEN1_DIM  144
#define HIDDEN2_DIM  144
#define HIDDEN3_DIM  144
#define OUTPUT_DIM   1

#define EPOCHS       100
#define LEARNING_RATE 0.001

#define MAX_PATH_LEN 512

static void trim_newline(char *str) {
    int len = strlen(str);
    while (len > 0 && (str[len-1] == '\n' || str[len-1] == '\r')) {
        str[--len] = '\0';
    }
}

int main(void) {
    printf("=== Neural Network Training ===\n\n");

    char train_path[MAX_PATH_LEN];
    char test_path[MAX_PATH_LEN];

    // Prompt for training data path
    printf("Enter training data path: ");
    if (!fgets(train_path, sizeof(train_path), stdin)) {
        fprintf(stderr, "Error reading input\n");
        return 1;
    }
    trim_newline(train_path);

    // Prompt for test/validation data path
    printf("Enter test data path: ");
    if (!fgets(test_path, sizeof(test_path), stdin)) {
        fprintf(stderr, "Error reading input\n");
        return 1;
    }
    trim_newline(test_path);

    printf("\n");

    // Load training data
    printf("Loading training data from: %s\n", train_path);
    Dataset *train_data = load_csv(train_path, OUTPUT_DIM, 1);
    
    if (!train_data) {
        fprintf(stderr, "Error: Failed to load training data\n");
        return 1;
    }
    dataset_info(train_data, "Train");

    // Load test data
    printf("\nLoading test data from: %s\n", test_path);
    Dataset *test_data = load_csv(test_path, OUTPUT_DIM, 1);
    
    if (!test_data) {
        fprintf(stderr, "Error: Failed to load test data\n");
        dataset_free(train_data);
        return 1;
    }
    dataset_info(test_data, "Test");

    // Normalize both datasets
    normalize_zscore(train_data);
    normalize_zscore(test_data);
    printf("\nData normalized (z-score)\n");

    // Get input dimension from data
    int input_dim = train_data->n_features;
    
    printf("\nArchitecture: %d -> %d -> %d -> %d -> %d\n\n", 
           input_dim, HIDDEN1_DIM, HIDDEN2_DIM, HIDDEN3_DIM, OUTPUT_DIM);

    // Create network
    NN *net = net_create(input_dim, HIDDEN1_DIM, HIDDEN2_DIM, HIDDEN3_DIM, OUTPUT_DIM);

    // Initialize metric lists
    MetricList *train_metrics = metrics_init();
    MetricList *test_metrics = metrics_init();

    // Training loop
    printf("Training for %d epochs (lr=%.4f)\n\n", EPOCHS, LEARNING_RATE);

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        // Train
        TrainResult train_result = train_epoch(net, train_data->X, train_data->Y, LEARNING_RATE);
        
        // Evaluate on test set
        ValResult test_result = validate(net, test_data->X, test_data->Y);
        
        // Store metrics
        metrics_append(train_metrics, epoch, train_result.loss, 
                      train_result.rmse, train_result.r_squared);
        metrics_append(test_metrics, epoch, test_result.loss, 
                      test_result.rmse, test_result.r_squared);
        
        if (epoch % 10 == 0 || epoch == 1) {
            printf("Epoch %3d/%d:\n", epoch, EPOCHS);
            printf("  Train - Loss: %.6f, RMSE: %.6f, R²: %.6f\n",
                   train_result.loss, train_result.rmse, train_result.r_squared);
            printf("  Test  - Loss: %.6f, RMSE: %.6f, R²: %.6f\n\n",
                   test_result.loss, test_result.rmse, test_result.r_squared);
        }
    }

    // Print history
    metrics_print(train_metrics, "Training");
    metrics_print(test_metrics, "Test");

    // Cleanup
    metrics_free(train_metrics);
    metrics_free(test_metrics);
    dataset_free(train_data);
    dataset_free(test_data);
    net_free(net);
    la_destroy();

    printf("Done.\n");
    return 0;
}

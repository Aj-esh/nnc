#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "optax/net.h"

// Calculate Mean Squared Error
double calculate_mse(double *output, double *target, int size) {
    double mse = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = output[i] - target[i];
        mse += diff * diff;
    }
    return mse / size;
}

int main() {
    printf("Neural Network - Forward and Backward Propagation\n");
    printf("==================================================\n\n");
    
    // Create neural network with specified architecture
    // Input: 24, Hidden: [128, 128, 128], Output: 1
    int input_size = 24;
    int hidden_size = 128;
    int num_hidden = 3;
    int output_size = 1;
    
    printf("Creating neural network...\n");
    printf("Architecture: %d -> %d -> %d -> %d -> %d\n", 
           input_size, hidden_size, hidden_size, hidden_size, output_size);
    
    NeuralNetwork *nn = create_network(input_size, hidden_size, num_hidden, output_size);
    printf("Network created successfully!\n\n");
    
    // Create sample training data
    double input[24];
    double target[1] = {0.8};  // Target output
    
    // Initialize input with some sample values
    printf("Initializing sample input data...\n");
    for (int i = 0; i < input_size; i++) {
        input[i] = (double)i / input_size;  // Values from 0 to ~1
    }
    
    // Training parameters
    double learning_rate = 0.01;
    int epochs = 1000;
    
    printf("\nTraining Parameters:\n");
    printf("  Learning Rate: %.4f\n", learning_rate);
    printf("  Epochs: %d\n\n", epochs);
    
    printf("Starting training...\n");
    printf("Epoch\t\tMSE\t\tOutput\n");
    printf("-----\t\t---\t\t------\n");
    
    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward propagation
        forward_propagation(nn, input);
        
        // Calculate error
        double mse = calculate_mse(nn->output, target, output_size);
        
        // Print progress every 100 epochs
        if (epoch % 100 == 0) {
            printf("%d\t\t%.6f\t%.6f\n", epoch, mse, nn->output[0]);
        }
        
        // Backward propagation
        backward_propagation(nn, target, learning_rate);
    }
    
    // Final evaluation
    printf("\nFinal Results:\n");
    printf("==============\n");
    forward_propagation(nn, input);
    double final_mse = calculate_mse(nn->output, target, output_size);
    printf("Final MSE: %.6f\n", final_mse);
    printf("Final Output: %.6f\n", nn->output[0]);
    printf("Target: %.6f\n", target[0]);
    printf("Error: %.6f\n", fabs(nn->output[0] - target[0]));
    
    // Test with different input
    printf("\n\nTesting with different input:\n");
    printf("=============================\n");
    for (int i = 0; i < input_size; i++) {
        input[i] = (double)(input_size - i) / input_size;  // Reverse pattern
    }
    
    forward_propagation(nn, input);
    printf("New Output: %.6f\n", nn->output[0]);
    
    // Clean up
    printf("\nCleaning up...\n");
    free_network(nn);
    printf("Network freed successfully!\n");
    
    return 0;
}

#include <stdio.h>
#include "src/optax/net.h"

int main() {
    // Network configuration: input=24, hidden=(128, 128, 128), output=1
    size_t input_size = 24;
    size_t hidden_sizes[] = {128, 128, 128};
    size_t num_hidden_layers = 3;
    size_t output_size = 1;
    
    printf("Creating neural network...\n");
    printf("Architecture: Input(%zu) -> Hidden(%zu, %zu, %zu) -> Output(%zu)\n",
           input_size, hidden_sizes[0], hidden_sizes[1], hidden_sizes[2], output_size);
    
    // Create network
    NeuralNetwork* net = create_network(input_size, num_hidden_layers, 
                                        hidden_sizes, output_size);
    if (!net) {
        printf("Failed to create network\n");
        return 1;
    }
    
    // Initialize weights
    printf("Initializing weights...\n");
    initialize_weights(net, 42);
    
    // Create dummy input and target
    double input[24];
    double target[1] = {1.0};
    
    printf("Generating sample input data...\n");
    for (size_t i = 0; i < input_size; i++) {
        input[i] = (double)i / input_size;
    }
    
    // Forward propagation
    printf("\n=== Forward Propagation ===\n");
    forward_propagation(net, input);
    printf("Output: %f\n", net->activations[num_hidden_layers + 1][0]);
    
    // Backward propagation (training)
    printf("\n=== Backward Propagation (Training) ===\n");
    double learning_rate = 0.01;
    printf("Learning rate: %f\n", learning_rate);
    
    printf("\nTraining for 10 iterations...\n");
    for (int i = 0; i < 10; i++) {
        backward_propagation(net, input, target, learning_rate);
        printf("Iteration %d: Output = %f, Target = %f, Error = %f\n", 
               i + 1, 
               net->activations[num_hidden_layers + 1][0],
               target[0],
               target[0] - net->activations[num_hidden_layers + 1][0]);
    }
    
    printf("\n=== Network Summary ===\n");
    printf("Total parameters:\n");
    size_t total_params = 0;
    size_t prev_size = input_size;
    for (size_t i = 0; i < num_hidden_layers; i++) {
        size_t params = prev_size * hidden_sizes[i] + hidden_sizes[i];
        printf("  Layer %zu: %zu weights + %zu biases = %zu parameters\n",
               i + 1, prev_size * hidden_sizes[i], hidden_sizes[i], params);
        total_params += params;
        prev_size = hidden_sizes[i];
    }
    size_t output_params = prev_size * output_size + output_size;
    printf("  Output layer: %zu weights + %zu biases = %zu parameters\n",
           prev_size * output_size, output_size, output_params);
    total_params += output_params;
    printf("Total: %zu parameters\n", total_params);
    
    // Clean up
    printf("\nCleaning up...\n");
    destroy_network(net);
    printf("Done!\n");
    
    return 0;
}

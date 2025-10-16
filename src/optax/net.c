#include "net.h"
#include "../la/linalg.h"
#include "../la/normal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Create neural network with specified architecture
NeuralNetwork* create_network(size_t input_size, size_t num_hidden_layers, 
                               const size_t* hidden_sizes, size_t output_size) {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!net) return NULL;
    
    net->input_size = input_size;
    net->num_hidden_layers = num_hidden_layers;
    net->output_size = output_size;
    
    // Copy hidden layer sizes
    net->hidden_sizes = (size_t*)malloc(num_hidden_layers * sizeof(size_t));
    for (size_t i = 0; i < num_hidden_layers; i++) {
        net->hidden_sizes[i] = hidden_sizes[i];
    }
    
    // Total number of layers (hidden + output)
    size_t total_layers = num_hidden_layers + 1;
    
    // Allocate arrays for weights, biases, activations, etc.
    net->weights = (double**)malloc(total_layers * sizeof(double*));
    net->biases = (double**)malloc(total_layers * sizeof(double*));
    net->activations = (double**)malloc((total_layers + 1) * sizeof(double*));
    net->z_values = (double**)malloc(total_layers * sizeof(double*));
    net->weight_gradients = (double**)malloc(total_layers * sizeof(double*));
    net->bias_gradients = (double**)malloc(total_layers * sizeof(double*));
    net->delta = (double**)malloc(total_layers * sizeof(double*));
    
    // Allocate memory for each layer
    size_t prev_size = input_size;
    for (size_t i = 0; i < num_hidden_layers; i++) {
        size_t curr_size = hidden_sizes[i];
        net->weights[i] = (double*)malloc(prev_size * curr_size * sizeof(double));
        net->biases[i] = (double*)malloc(curr_size * sizeof(double));
        net->activations[i] = (double*)malloc(prev_size * sizeof(double));
        net->z_values[i] = (double*)malloc(curr_size * sizeof(double));
        net->weight_gradients[i] = (double*)malloc(prev_size * curr_size * sizeof(double));
        net->bias_gradients[i] = (double*)malloc(curr_size * sizeof(double));
        net->delta[i] = (double*)malloc(curr_size * sizeof(double));
        prev_size = curr_size;
    }
    
    // Output layer
    size_t output_idx = num_hidden_layers;
    net->weights[output_idx] = (double*)malloc(prev_size * output_size * sizeof(double));
    net->biases[output_idx] = (double*)malloc(output_size * sizeof(double));
    net->activations[output_idx] = (double*)malloc(prev_size * sizeof(double));
    net->z_values[output_idx] = (double*)malloc(output_size * sizeof(double));
    net->weight_gradients[output_idx] = (double*)malloc(prev_size * output_size * sizeof(double));
    net->bias_gradients[output_idx] = (double*)malloc(output_size * sizeof(double));
    net->delta[output_idx] = (double*)malloc(output_size * sizeof(double));
    net->activations[output_idx + 1] = (double*)malloc(output_size * sizeof(double));
    
    return net;
}

// Destroy neural network and free memory
void destroy_network(NeuralNetwork* net) {
    if (!net) return;
    
    size_t total_layers = net->num_hidden_layers + 1;
    
    for (size_t i = 0; i < total_layers; i++) {
        free(net->weights[i]);
        free(net->biases[i]);
        free(net->activations[i]);
        free(net->z_values[i]);
        free(net->weight_gradients[i]);
        free(net->bias_gradients[i]);
        free(net->delta[i]);
    }
    free(net->activations[total_layers]);
    
    free(net->weights);
    free(net->biases);
    free(net->activations);
    free(net->z_values);
    free(net->weight_gradients);
    free(net->bias_gradients);
    free(net->delta);
    free(net->hidden_sizes);
    free(net);
}

// Initialize weights using Xavier initialization
void initialize_weights(NeuralNetwork* net, unsigned int seed) {
    init_random(seed);
    
    size_t prev_size = net->input_size;
    
    for (size_t i = 0; i < net->num_hidden_layers; i++) {
        size_t curr_size = net->hidden_sizes[i];
        double stddev = sqrt(2.0 / prev_size);  // Xavier initialization
        random_normal(net->weights[i], prev_size * curr_size, 0.0, stddev);
        
        // Initialize biases to zero
        for (size_t j = 0; j < curr_size; j++) {
            net->biases[i][j] = 0.0;
        }
        prev_size = curr_size;
    }
    
    // Output layer
    size_t output_idx = net->num_hidden_layers;
    double stddev = sqrt(2.0 / prev_size);
    random_normal(net->weights[output_idx], prev_size * net->output_size, 0.0, stddev);
    for (size_t j = 0; j < net->output_size; j++) {
        net->biases[output_idx][j] = 0.0;
    }
}

// ReLU activation function
void relu(const double* input, double* output, size_t size) {
    for (size_t i = 0; i < size; i++) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

// ReLU derivative
void relu_derivative(const double* input, double* output, size_t size) {
    for (size_t i = 0; i < size; i++) {
        output[i] = input[i] > 0 ? 1.0 : 0.0;
    }
}

// Sigmoid activation function
void sigmoid(const double* input, double* output, size_t size) {
    for (size_t i = 0; i < size; i++) {
        output[i] = 1.0 / (1.0 + exp(-input[i]));
    }
}

// Sigmoid derivative
void sigmoid_derivative(const double* input, double* output, size_t size) {
    for (size_t i = 0; i < size; i++) {
        double sig = 1.0 / (1.0 + exp(-input[i]));
        output[i] = sig * (1.0 - sig);
    }
}

// Forward propagation through the network
void forward_propagation(NeuralNetwork* net, const double* input) {
    // Copy input to first activation
    memcpy(net->activations[0], input, net->input_size * sizeof(double));
    
    // Forward through hidden layers
    for (size_t i = 0; i < net->num_hidden_layers; i++) {
        size_t prev_size = (i == 0) ? net->input_size : net->hidden_sizes[i - 1];
        size_t curr_size = net->hidden_sizes[i];
        
        // Compute z = W * a + b
        matrix_multiply(net->activations[i], net->weights[i], net->z_values[i],
                        1, prev_size, curr_size);
        matrix_add(net->z_values[i], net->biases[i], net->z_values[i],
                   1, curr_size);
        
        // Apply ReLU activation
        relu(net->z_values[i], net->activations[i + 1], curr_size);
    }
    
    // Forward through output layer
    size_t output_idx = net->num_hidden_layers;
    size_t prev_size = net->hidden_sizes[net->num_hidden_layers - 1];
    
    // Compute z = W * a + b
    matrix_multiply(net->activations[output_idx], net->weights[output_idx], 
                    net->z_values[output_idx], 1, prev_size, net->output_size);
    matrix_add(net->z_values[output_idx], net->biases[output_idx], 
               net->z_values[output_idx], 1, net->output_size);
    
    // Apply sigmoid activation for output
    sigmoid(net->z_values[output_idx], net->activations[output_idx + 1], 
            net->output_size);
}

// Backward propagation with gradient descent
void backward_propagation(NeuralNetwork* net, const double* input, 
                          const double* target, double learning_rate) {
    // First do forward propagation to compute activations
    forward_propagation(net, input);
    
    size_t total_layers = net->num_hidden_layers + 1;
    
    // Compute output layer error
    // delta = (a - y) * sigmoid'(z)
    size_t output_idx = net->num_hidden_layers;
    double* sigmoid_deriv = (double*)malloc(net->output_size * sizeof(double));
    sigmoid_derivative(net->z_values[output_idx], sigmoid_deriv, net->output_size);
    
    for (size_t i = 0; i < net->output_size; i++) {
        net->delta[output_idx][i] = (net->activations[output_idx + 1][i] - target[i]) 
                                     * sigmoid_deriv[i];
    }
    free(sigmoid_deriv);
    
    // Backpropagate through hidden layers
    for (int i = (int)net->num_hidden_layers - 1; i >= 0; i--) {
        size_t curr_size = net->hidden_sizes[i];
        size_t next_size = ((size_t)i == net->num_hidden_layers - 1) ? 
                           net->output_size : net->hidden_sizes[i + 1];
        
        // Compute delta for this layer
        // delta[i] = (W[i+1]^T * delta[i+1]) ⊙ relu'(z[i])
        double* weight_transpose = (double*)malloc(curr_size * next_size * sizeof(double));
        double* temp_delta = (double*)malloc(curr_size * sizeof(double));
        double* relu_deriv = (double*)malloc(curr_size * sizeof(double));
        
        matrix_transpose(net->weights[i + 1], weight_transpose, next_size, curr_size);
        matrix_multiply(net->delta[i + 1], weight_transpose, temp_delta,
                        1, next_size, curr_size);
        relu_derivative(net->z_values[i], relu_deriv, curr_size);
        matrix_hadamard(temp_delta, relu_deriv, net->delta[i], 1, curr_size);
        
        free(weight_transpose);
        free(temp_delta);
        free(relu_deriv);
    }
    
    // Compute gradients and update weights
    for (size_t i = 0; i < total_layers; i++) {
        size_t prev_size = (i == 0) ? net->input_size : 
                          (i <= net->num_hidden_layers ? net->hidden_sizes[i - 1] : 0);
        size_t curr_size = (i < net->num_hidden_layers) ? 
                          net->hidden_sizes[i] : net->output_size;
        
        // Compute weight gradients: dW = a[i]^T * delta[i]
        double* activation_transpose = (double*)malloc(prev_size * sizeof(double));
        memcpy(activation_transpose, net->activations[i], prev_size * sizeof(double));
        
        for (size_t j = 0; j < curr_size; j++) {
            for (size_t k = 0; k < prev_size; k++) {
                net->weight_gradients[i][k * curr_size + j] = 
                    activation_transpose[k] * net->delta[i][j];
            }
        }
        
        // Update weights: W = W - learning_rate * dW
        for (size_t j = 0; j < prev_size * curr_size; j++) {
            net->weights[i][j] -= learning_rate * net->weight_gradients[i][j];
        }
        
        // Update biases: b = b - learning_rate * delta
        for (size_t j = 0; j < curr_size; j++) {
            net->biases[i][j] -= learning_rate * net->delta[i][j];
        }
        
        free(activation_transpose);
    }
}

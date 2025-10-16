#ifndef NET_H
#define NET_H

#include <stddef.h>

// Neural Network structure
typedef struct {
    size_t input_size;
    size_t num_hidden_layers;
    size_t* hidden_sizes;
    size_t output_size;
    
    // Weights and biases
    double** weights;  // weights[i]: matrix for layer i
    double** biases;   // biases[i]: bias vector for layer i
    
    // Activations and intermediate values (for forward/backward pass)
    double** activations;  // activations[i]: activation output of layer i
    double** z_values;     // z_values[i]: pre-activation values of layer i
    
    // Gradients (for backward pass)
    double** weight_gradients;
    double** bias_gradients;
    double** delta;  // error terms for each layer
    
} NeuralNetwork;

// Create and destroy network
NeuralNetwork* create_network(size_t input_size, size_t num_hidden_layers, 
                               const size_t* hidden_sizes, size_t output_size);
void destroy_network(NeuralNetwork* net);

// Initialize weights and biases
void initialize_weights(NeuralNetwork* net, unsigned int seed);

// Activation functions
void relu(const double* input, double* output, size_t size);
void relu_derivative(const double* input, double* output, size_t size);
void sigmoid(const double* input, double* output, size_t size);
void sigmoid_derivative(const double* input, double* output, size_t size);

// Forward propagation
void forward_propagation(NeuralNetwork* net, const double* input);

// Backward propagation
void backward_propagation(NeuralNetwork* net, const double* input, 
                          const double* target, double learning_rate);

#endif // NET_H

#ifndef NET_H
#define NET_H

#include <stddef.h>

// Neural network structure
typedef struct {
    // Network architecture
    int input_size;      // 24
    int hidden_size;     // 128
    int num_hidden;      // 3 layers
    int output_size;     // 1
    
    // Weights and biases
    double **W1;         // Input to hidden1: [input_size x hidden_size]
    double *b1;          // Bias for hidden1: [hidden_size]
    double **W2;         // Hidden1 to hidden2: [hidden_size x hidden_size]
    double *b2;          // Bias for hidden2: [hidden_size]
    double **W3;         // Hidden2 to hidden3: [hidden_size x hidden_size]
    double *b3;          // Bias for hidden3: [hidden_size]
    double **W4;         // Hidden3 to output: [hidden_size x output_size]
    double *b4;          // Bias for output: [output_size]
    
    // Activations (stored during forward pass for backward pass)
    double *input;       // [input_size]
    double *hidden1;     // [hidden_size]
    double *hidden2;     // [hidden_size]
    double *hidden3;     // [hidden_size]
    double *output;      // [output_size]
    
    // Pre-activation values (z = Wx + b, before activation function)
    double *z1;          // [hidden_size]
    double *z2;          // [hidden_size]
    double *z3;          // [hidden_size]
    double *z4;          // [output_size]
    
    // Gradients
    double **dW1;
    double *db1;
    double **dW2;
    double *db2;
    double **dW3;
    double *db3;
    double **dW4;
    double *db4;
} NeuralNetwork;

// Function declarations
NeuralNetwork* create_network(int input_size, int hidden_size, int num_hidden, int output_size);
void free_network(NeuralNetwork *nn);
void forward_propagation(NeuralNetwork *nn, double *input);
void backward_propagation(NeuralNetwork *nn, double *target, double learning_rate);
double sigmoid(double x);
double sigmoid_derivative(double x);
void initialize_weights(NeuralNetwork *nn);

#endif // NET_H

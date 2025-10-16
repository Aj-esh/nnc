#include "net.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid
double sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

// Allocate 2D array
double** allocate_2d(int rows, int cols) {
    double **arr = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        arr[i] = (double*)calloc(cols, sizeof(double));
    }
    return arr;
}

// Free 2D array
void free_2d(double **arr, int rows) {
    for (int i = 0; i < rows; i++) {
        free(arr[i]);
    }
    free(arr);
}

// Initialize weights with random values (Xavier initialization)
void initialize_weights(NeuralNetwork *nn) {
    srand(time(NULL));
    
    // Initialize W1 (input to hidden1)
    double scale1 = sqrt(2.0 / nn->input_size);
    for (int i = 0; i < nn->input_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->W1[i][j] = ((double)rand() / RAND_MAX - 0.5) * 2 * scale1;
        }
    }
    
    // Initialize W2 (hidden1 to hidden2)
    double scale2 = sqrt(2.0 / nn->hidden_size);
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->W2[i][j] = ((double)rand() / RAND_MAX - 0.5) * 2 * scale2;
        }
    }
    
    // Initialize W3 (hidden2 to hidden3)
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->W3[i][j] = ((double)rand() / RAND_MAX - 0.5) * 2 * scale2;
        }
    }
    
    // Initialize W4 (hidden3 to output)
    double scale4 = sqrt(2.0 / nn->hidden_size);
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->output_size; j++) {
            nn->W4[i][j] = ((double)rand() / RAND_MAX - 0.5) * 2 * scale4;
        }
    }
    
    // Biases are already initialized to 0 by calloc
}

// Create and initialize neural network
NeuralNetwork* create_network(int input_size, int hidden_size, int num_hidden, int output_size) {
    NeuralNetwork *nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->num_hidden = num_hidden;
    nn->output_size = output_size;
    
    // Allocate weights
    nn->W1 = allocate_2d(input_size, hidden_size);
    nn->W2 = allocate_2d(hidden_size, hidden_size);
    nn->W3 = allocate_2d(hidden_size, hidden_size);
    nn->W4 = allocate_2d(hidden_size, output_size);
    
    // Allocate biases
    nn->b1 = (double*)calloc(hidden_size, sizeof(double));
    nn->b2 = (double*)calloc(hidden_size, sizeof(double));
    nn->b3 = (double*)calloc(hidden_size, sizeof(double));
    nn->b4 = (double*)calloc(output_size, sizeof(double));
    
    // Allocate activations
    nn->input = (double*)calloc(input_size, sizeof(double));
    nn->hidden1 = (double*)calloc(hidden_size, sizeof(double));
    nn->hidden2 = (double*)calloc(hidden_size, sizeof(double));
    nn->hidden3 = (double*)calloc(hidden_size, sizeof(double));
    nn->output = (double*)calloc(output_size, sizeof(double));
    
    // Allocate pre-activations
    nn->z1 = (double*)calloc(hidden_size, sizeof(double));
    nn->z2 = (double*)calloc(hidden_size, sizeof(double));
    nn->z3 = (double*)calloc(hidden_size, sizeof(double));
    nn->z4 = (double*)calloc(output_size, sizeof(double));
    
    // Allocate gradients
    nn->dW1 = allocate_2d(input_size, hidden_size);
    nn->dW2 = allocate_2d(hidden_size, hidden_size);
    nn->dW3 = allocate_2d(hidden_size, hidden_size);
    nn->dW4 = allocate_2d(hidden_size, output_size);
    
    nn->db1 = (double*)calloc(hidden_size, sizeof(double));
    nn->db2 = (double*)calloc(hidden_size, sizeof(double));
    nn->db3 = (double*)calloc(hidden_size, sizeof(double));
    nn->db4 = (double*)calloc(output_size, sizeof(double));
    
    // Initialize weights
    initialize_weights(nn);
    
    return nn;
}

// Free neural network memory
void free_network(NeuralNetwork *nn) {
    free_2d(nn->W1, nn->input_size);
    free_2d(nn->W2, nn->hidden_size);
    free_2d(nn->W3, nn->hidden_size);
    free_2d(nn->W4, nn->hidden_size);
    
    free(nn->b1);
    free(nn->b2);
    free(nn->b3);
    free(nn->b4);
    
    free(nn->input);
    free(nn->hidden1);
    free(nn->hidden2);
    free(nn->hidden3);
    free(nn->output);
    
    free(nn->z1);
    free(nn->z2);
    free(nn->z3);
    free(nn->z4);
    
    free_2d(nn->dW1, nn->input_size);
    free_2d(nn->dW2, nn->hidden_size);
    free_2d(nn->dW3, nn->hidden_size);
    free_2d(nn->dW4, nn->hidden_size);
    
    free(nn->db1);
    free(nn->db2);
    free(nn->db3);
    free(nn->db4);
    
    free(nn);
}

// Forward propagation algorithm
void forward_propagation(NeuralNetwork *nn, double *input) {
    // Store input
    for (int i = 0; i < nn->input_size; i++) {
        nn->input[i] = input[i];
    }
    
    // Layer 1: Input -> Hidden1
    // z1 = W1^T * input + b1
    for (int j = 0; j < nn->hidden_size; j++) {
        nn->z1[j] = nn->b1[j];
        for (int i = 0; i < nn->input_size; i++) {
            nn->z1[j] += nn->W1[i][j] * nn->input[i];
        }
        nn->hidden1[j] = sigmoid(nn->z1[j]);
    }
    
    // Layer 2: Hidden1 -> Hidden2
    // z2 = W2^T * hidden1 + b2
    for (int j = 0; j < nn->hidden_size; j++) {
        nn->z2[j] = nn->b2[j];
        for (int i = 0; i < nn->hidden_size; i++) {
            nn->z2[j] += nn->W2[i][j] * nn->hidden1[i];
        }
        nn->hidden2[j] = sigmoid(nn->z2[j]);
    }
    
    // Layer 3: Hidden2 -> Hidden3
    // z3 = W3^T * hidden2 + b3
    for (int j = 0; j < nn->hidden_size; j++) {
        nn->z3[j] = nn->b3[j];
        for (int i = 0; i < nn->hidden_size; i++) {
            nn->z3[j] += nn->W3[i][j] * nn->hidden2[i];
        }
        nn->hidden3[j] = sigmoid(nn->z3[j]);
    }
    
    // Layer 4: Hidden3 -> Output
    // z4 = W4^T * hidden3 + b4
    for (int j = 0; j < nn->output_size; j++) {
        nn->z4[j] = nn->b4[j];
        for (int i = 0; i < nn->hidden_size; i++) {
            nn->z4[j] += nn->W4[i][j] * nn->hidden3[i];
        }
        nn->output[j] = sigmoid(nn->z4[j]);
    }
}

// Backward propagation algorithm
void backward_propagation(NeuralNetwork *nn, double *target, double learning_rate) {
    // Allocate delta arrays for each layer
    double *delta4 = (double*)calloc(nn->output_size, sizeof(double));
    double *delta3 = (double*)calloc(nn->hidden_size, sizeof(double));
    double *delta2 = (double*)calloc(nn->hidden_size, sizeof(double));
    double *delta1 = (double*)calloc(nn->hidden_size, sizeof(double));
    
    // Output layer delta: delta4 = (output - target) * sigmoid'(z4)
    for (int i = 0; i < nn->output_size; i++) {
        delta4[i] = (nn->output[i] - target[i]) * sigmoid_derivative(nn->z4[i]);
    }
    
    // Hidden3 layer delta: delta3 = (W4 * delta4) * sigmoid'(z3)
    for (int i = 0; i < nn->hidden_size; i++) {
        delta3[i] = 0.0;
        for (int j = 0; j < nn->output_size; j++) {
            delta3[i] += nn->W4[i][j] * delta4[j];
        }
        delta3[i] *= sigmoid_derivative(nn->z3[i]);
    }
    
    // Hidden2 layer delta: delta2 = (W3 * delta3) * sigmoid'(z2)
    for (int i = 0; i < nn->hidden_size; i++) {
        delta2[i] = 0.0;
        for (int j = 0; j < nn->hidden_size; j++) {
            delta2[i] += nn->W3[i][j] * delta3[j];
        }
        delta2[i] *= sigmoid_derivative(nn->z2[i]);
    }
    
    // Hidden1 layer delta: delta1 = (W2 * delta2) * sigmoid'(z1)
    for (int i = 0; i < nn->hidden_size; i++) {
        delta1[i] = 0.0;
        for (int j = 0; j < nn->hidden_size; j++) {
            delta1[i] += nn->W2[i][j] * delta2[j];
        }
        delta1[i] *= sigmoid_derivative(nn->z1[i]);
    }
    
    // Compute gradients and update weights/biases
    
    // Update W4 and b4
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->output_size; j++) {
            nn->dW4[i][j] = nn->hidden3[i] * delta4[j];
            nn->W4[i][j] -= learning_rate * nn->dW4[i][j];
        }
    }
    for (int j = 0; j < nn->output_size; j++) {
        nn->db4[j] = delta4[j];
        nn->b4[j] -= learning_rate * nn->db4[j];
    }
    
    // Update W3 and b3
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->dW3[i][j] = nn->hidden2[i] * delta3[j];
            nn->W3[i][j] -= learning_rate * nn->dW3[i][j];
        }
    }
    for (int j = 0; j < nn->hidden_size; j++) {
        nn->db3[j] = delta3[j];
        nn->b3[j] -= learning_rate * nn->db3[j];
    }
    
    // Update W2 and b2
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->dW2[i][j] = nn->hidden1[i] * delta2[j];
            nn->W2[i][j] -= learning_rate * nn->dW2[i][j];
        }
    }
    for (int j = 0; j < nn->hidden_size; j++) {
        nn->db2[j] = delta2[j];
        nn->b2[j] -= learning_rate * nn->db2[j];
    }
    
    // Update W1 and b1
    for (int i = 0; i < nn->input_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->dW1[i][j] = nn->input[i] * delta1[j];
            nn->W1[i][j] -= learning_rate * nn->dW1[i][j];
        }
    }
    for (int j = 0; j < nn->hidden_size; j++) {
        nn->db1[j] = delta1[j];
        nn->b1[j] -= learning_rate * nn->db1[j];
    }
    
    // Free delta arrays
    free(delta4);
    free(delta3);
    free(delta2);
    free(delta1);
}

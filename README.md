# nnc
Neural network from scratch written in C language

## Overview
A neural network implementation in C featuring forward and backward propagation algorithms.

## Architecture
The current implementation supports:
- Input layer: 24 neurons
- Hidden layers: 3 layers with 128 neurons each (ReLU activation)
- Output layer: 1 neuron (Sigmoid activation)
- Total parameters: 36,353

## Features
- **Forward Propagation**: Computes network output for given input
- **Backward Propagation**: Computes gradients and updates weights using gradient descent
- **Xavier Initialization**: Smart weight initialization for better convergence
- **Activation Functions**: ReLU for hidden layers, Sigmoid for output
- **Memory Management**: Proper allocation and deallocation of network resources

## Building
```bash
make
```

## Running
```bash
make run
# or
./bin/nnc
```

## Cleaning
```bash
make clean
```

## Project Structure
```
nnc/
├── src/
│   ├── la/          # Linear algebra operations
│   │   ├── linalg.c/h   # Matrix operations
│   │   └── normal.c/h   # Random number generation
│   ├── poolla/      # BLAS-like operations
│   │   └── blas.c/h     # Vector operations
│   └── optax/       # Neural network implementation
│       └── net.c/h      # Network structure and algorithms
├── main.c           # Example program
└── Makefile         # Build configuration
```

## Usage Example
```c
// Create network with input=24, hidden=(128,128,128), output=1
size_t hidden_sizes[] = {128, 128, 128};
NeuralNetwork* net = create_network(24, 3, hidden_sizes, 1);

// Initialize weights
initialize_weights(net, 42);

// Forward propagation
double input[24] = {...};
forward_propagation(net, input);

// Backward propagation (training)
double target[1] = {1.0};
double learning_rate = 0.01;
backward_propagation(net, input, target, learning_rate);

// Clean up
destroy_network(net);
```

## Algorithm Details

### Forward Propagation
1. Compute pre-activation: `z = W * a + b`
2. Apply activation function: `a = activation(z)`
3. Repeat for all layers

### Backward Propagation
1. Compute output layer error: `δ = (a - y) ⊙ σ'(z)`
2. Backpropagate error: `δ[l] = (W[l+1]^T * δ[l+1]) ⊙ f'(z[l])`
3. Compute gradients: `∇W = a^T * δ`, `∇b = δ`
4. Update weights: `W = W - η * ∇W`, `b = b - η * ∇b`

Where:
- `W` = weights, `b` = biases
- `a` = activations, `z` = pre-activations
- `δ` = error terms
- `η` = learning rate
- `⊙` = element-wise multiplication
- `σ` = sigmoid, `f` = activation function


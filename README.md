# nnc
Neural Network from Scratch Written in C

## Overview
This project implements a fully-connected neural network in C with forward and backward propagation algorithms.

## Architecture
- **Input Layer**: 24 neurons
- **Hidden Layers**: 3 layers with 128 neurons each
- **Output Layer**: 1 neuron
- **Activation Function**: Sigmoid

## Features
- Forward propagation algorithm
- Backward propagation algorithm (backpropagation)
- Xavier weight initialization
- Gradient descent optimization
- Mean Squared Error (MSE) loss calculation

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

## Clean

```bash
make clean
```

## Project Structure
```
nnc/
├── src/
│   ├── main.c           # Example usage and training loop
│   └── optax/
│       ├── net.h        # Neural network header
│       └── net.c        # Neural network implementation
├── Makefile             # Build configuration
└── README.md
```

## Implementation Details

### Forward Propagation
The forward propagation algorithm computes the output of the neural network by passing input through all layers:
1. Input → Hidden Layer 1
2. Hidden Layer 1 → Hidden Layer 2
3. Hidden Layer 2 → Hidden Layer 3
4. Hidden Layer 3 → Output

Each layer applies: `activation(W * input + bias)`

### Backward Propagation
The backward propagation algorithm computes gradients and updates weights using:
1. Calculate output layer error
2. Propagate error backward through all hidden layers
3. Compute gradients for weights and biases
4. Update parameters using gradient descent

### Training
The example demonstrates training the network to approximate a target output of 0.8, showing convergence over 1000 epochs with decreasing MSE.

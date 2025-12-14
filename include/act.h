#ifndef ACT_H
#define ACT_H

#include "la/linalg.h"

typedef enum {
    ACT_NONE = 0,
    ACT_RELU,
} Activation;

// Forward pass
Matrix *relu(const Matrix *Z);
// Backward pass
Matrix* drelu(const Matrix *Z, const Matrix *dZ);

#endif // ACT_H
#ifndef NORMAL_H
#define NORMAL_H

#include <stddef.h>

// Initialize random number generator
void init_random(unsigned int seed);

// Generate random normal distribution values
void random_normal(double* array, size_t size, double mean, double stddev);

#endif // NORMAL_H

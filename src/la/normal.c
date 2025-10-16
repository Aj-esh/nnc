#include "normal.h"
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void init_random(unsigned int seed) {
    srand(seed);
}

// Box-Muller transform for generating normal distribution
void random_normal(double* array, size_t size, double mean, double stddev) {
    for (size_t i = 0; i < size; i += 2) {
        double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        array[i] = mean + stddev * z0;
        
        if (i + 1 < size) {
            double z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
            array[i + 1] = mean + stddev * z1;
        }
    }
}

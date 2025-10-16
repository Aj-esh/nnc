#ifndef LA_NORMAL_H
#define LA_NORMAL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#include "linalg.h"

/**
 * Create Matrix (row, col) with normal (mean, std) distribution
 * @param row number of rows
 * @param col number of columns
 * @param mean mean of normal distribution
 * @param std standard deviation of normal distribution
 * @return pointer to created Matrix, or NULL on failure
 */
Matrix* matrix_randn(size_t row, size_t col, double mean, double std);

/**
 * He initialization for weights matrix (row, col)
 * @param row number of rows
 * @param col number of columns
 * @param fan_in number of input units, if -1, set to row
 * @return pointer to created Matrix, or NULL on failure
 */
Matrix* He_init(size_t row, size_t col, int fan_in);

#endif // LA_NORMAL_H
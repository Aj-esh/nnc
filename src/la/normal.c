#include "blas.h"
#include "linalg.h"
#include "normal.h"

static void ensure_seeded(void) {
    static bool seeded = false;
    if(!seeded) {
        const unsigned int seed = (unsigned int) time(NULL) ^ (uintptr_t) &seeded;
        srand(seed);
        seeded = true;
    }
}

static double uniform01(void) {
    return (double) rand() / ((double) RAND_MAX + 1);
}

static double randn(double mean, double std) {
    static bool have_spare = false;
    static double spare;

    ensure_seeded();

    if(have_spare) {
        have_spare = false;
        return spare * std + mean;
    }

    have_spare = true;
    double u, v, r2;
    do {
        u = 2.0 * uniform01() - 1.0;
        v = 2.0 * uniform01() - 1.0;
        r2 = u * u + v * v;
    } while (r2 >= 1.0 || r2 == 0.0);

    const double mul = sqrt(-2.0 * log(r2) / r2);
    spare = v * mul;
    return mean + std * u * mul;
}

Matrix* matrix_randn(size_t row, size_t col, double mean, double std) {
    Matrix *m = create_matrix(row, col);
    if(!m) {
        return NULL;
    }

    for(size_t i=0; i<row*col; i++) {
        m->data[i] = randn(mean, std);
    }
    return m;
}

Matrix* He_init(size_t row, size_t col, int fan_in){
    if(fan_in == -1) fan_in = row;
    double std = sqrt(2.0 / fan_in);
    return matrix_randn(row, col, 0.0, std);
} 
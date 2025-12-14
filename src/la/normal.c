#include "la/linalg.h"
#include "la/normal.h"

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

static double randn_r(double mean, double std, unsigned int *seed) {
    double u, v, r2;
    do {
        u = 2.0 * ((double)rand_r(seed) / RAND_MAX) - 1.0;
        v = 2.0 * ((double)rand_r(seed) / RAND_MAX) - 1.0;
        r2 = u * u + v * v;
    } while (r2 >= 1.0 || r2 == 0.0);

    const double mul = sqrt(-2.0 * log(r2) / r2);
    return mean + std * u * mul; // Only return one to keep state simple
}

typedef struct {
    Matrix *m;
    double mean, std;
    int start, end;
    unsigned int seed;
} RandArgs;

static void randn_task(void *arg) {
    RandArgs *a = (RandArgs*)arg;
    unsigned int seed = a->seed;
    for(int i = a->start; i < a->end; i++) {
        a->m->data[i] = randn_r(a->mean, a->std, &seed);
    }
    free(a);
}

Matrix* matrix_randn(size_t row, size_t col, double mean, double std) {
    Matrix *m = create_matrix(row, col);
    if(!m) return NULL;

    ThreadPool *tp = get_la_pool();
    int total = row * col;
    int num_threads = tp->tcount;
    int chunk = (total + num_threads - 1) / num_threads;

    for(int i=0; i<num_threads; i++) {
        int start = i * chunk;
        int end = (start + chunk > total) ? total : start + chunk;
        if(start >= end) break;

        RandArgs *args = malloc(sizeof(RandArgs));
        args->m = m; args->mean = mean; args->std = std;
        args->start = start; args->end = end;
        args->seed = rand() + i; // Initial seed from global rand
        threadpool_submit(tp, randn_task, args);
    }
    threadpool_wait(tp);
    return m;
}

Matrix* He_init(size_t row, size_t col, int fan_in){
    if(fan_in == -1) fan_in = (int)row;
    double std = sqrt(2.0 / (double)fan_in);
    return matrix_randn(row, col, 0.0, std);
}

Matrix* Xavier_init(size_t row, size_t col, int fan_in, int fan_out) {
    if(fan_in == -1) fan_in = (int)row;
    if(fan_out == -1) fan_out = (int)col;

    const double denom = (double)fan_in + (double)fan_out;
    if(denom <= 0.0) return NULL;

    const double std = sqrt(2.0 / denom);
    return matrix_randn(row, col, 0.0, std);
}
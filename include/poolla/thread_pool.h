#ifndef THREADPOOLLA_BLAS_H
#define THREADPOOLLA_BLAS_H

#include <pthread.h>

typedef struct Task {
    void (*function) (void *arg);
    void *args;
    struct Task *next;
} Task;

typedef struct ThreadPool {
    pthread_mutex_t lock; // Mutex for synchronizing access to the task queue
    pthread_cond_t notify; // Condition variable for notifying worker threads of new tasks
    pthread_cond_t working; // Condition variable for notifying when all tasks are done
    pthread_t *threads; // Array of worker threads
    Task *head, *tail; // Head and tail of the task queue
    int tcount, active, shutdown; // Thread count, active task count, and shutdown flag
} ThreadPool;

/**
 * @brief Initializes a thread pool with a specified number of threads.
 * @param nthreads Number of threads to create in the pool.
 * @return pointer to the initialized ThreadPool structure, or NULL on failure.
 */
ThreadPool* threadpool_init(int nthreads);

/**
 * @brief submits a new task to the thread pool.
 * @param pool Pointer to the ThreadPool structure.
 * @param function Function pointer representing the task to be executed.
 * @param args Arguments to be passed to the task function.
 */
void threadpool_submit(ThreadPool *pool, void (*function)(void *), void *args);

/** 
 * @brief waits for all tasks in the thread pool to complete.
 * @param pool Pointer to the ThreadPool structure.
 * @return void   
*/
void threadpool_wait(ThreadPool *pool);

/**
 * @brief Destroys the thread pool, freeing all associated resources.
 * @param pool Pointer to the ThreadPool structure.
 * @return void
 */
void threadpool_destroy(ThreadPool *pool);

#endif //THREADPOOLLA_BLAS_H
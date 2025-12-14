#include "poolla/thread_pool.h"
#include <stdlib.h>
#include <stdio.h> 

// function for each worker thread
static void* tp_worker (void* arg) {
    ThreadPool *pool = (ThreadPool*) arg;
    Task* task;

    while(1) {
        pthread_mutex_lock(&(pool->lock)); // lock the pool
        // wait for new task or shutdown signal
        while(pool->active == 0 && !pool->shutdown) {
            pthread_cond_wait(&(pool->notify), &(pool->lock));
        }

        if(pool->shutdown) {
            pthread_mutex_unlock(&(pool->lock));
            pthread_exit(NULL);
        }

        // dequeue a task
        task = pool->head;
        pool->head = task->next;
        if(pool->head == NULL) {
            pool->tail = NULL;
        }

        pthread_mutex_unlock(&(pool->lock)); // unlock the pool

        // Execute the task
        (*(task->function))(task->args);
        free(task);

        // signal task completion
        pthread_mutex_lock(&(pool->lock));
        pool->active--;
        if(pool->active == 0) {
            pthread_cond_signal(&(pool->working));
        }
        pthread_mutex_unlock(&(pool->lock));
    }
    return NULL;
}

ThreadPool* threadpool_init(int nthreads) {
    ThreadPool *pool = (ThreadPool*) malloc(sizeof(ThreadPool));
    if(pool == NULL) {
        return NULL;
    }

    pool->tcount = nthreads;
    pool->active = 0;
    pool->shutdown = 0;
    pool->head = NULL;
    pool->tail = NULL;

    pthread_mutex_init(&(pool->lock), NULL);
    pthread_cond_init(&(pool->notify), NULL);
    pthread_cond_init(&(pool->working), NULL);

    pool->threads = (pthread_t*) malloc(sizeof(pthread_t) * nthreads);
    if(pool->threads == NULL) {
        free(pool);
        return NULL;
    }

    for(int i=0; i<nthreads; i++) {
        pthread_create(&(pool->threads[i]), NULL, tp_worker, (void*) pool);
    }
    return pool;
}

void threadpool_submit(ThreadPool *pool, void (*function)(void*), void *arg) {
    Task *task = (Task*) malloc(sizeof(Task));
    if(task == NULL) return;

    task->function = function;
    task->args = arg;
    task->next = NULL;
    
    pthread_mutex_lock(&(pool->lock));
    
    if(pool->tail == NULL) {
        pool->head = task;
        pool->tail = task;
    } else {
        pool->tail->next = task;
        pool->tail = task;
    }

    pool->active++;
    pthread_cond_signal(&(pool->notify));
    pthread_mutex_unlock(&(pool->lock));
}

void threadpool_wait(ThreadPool *pool) {
    pthread_mutex_lock(&(pool->lock));
    while(pool->active > 0) {
        pthread_cond_wait(&(pool->working), &(pool->lock));
    }
    pthread_mutex_unlock(&(pool->lock));
}

void threadpool_destroy(ThreadPool *pool) {
    pthread_mutex_lock(&(pool->lock));
    pool->shutdown = 1;
    pthread_cond_broadcast(&(pool->notify));
    pthread_mutex_unlock(&(pool->lock));

    for(int i=0; i<pool->tcount; i++) 
        pthread_join(pool->threads[i], NULL);
    
    // free reamianing tasks
    while(pool->head) {
        Task *tmp = pool->head;
        pool->head = pool->head->next;
        free(tmp);
    }

    pthread_mutex_destroy(&(pool->lock));
    pthread_cond_destroy(&(pool->notify));
    pthread_cond_destroy(&(pool->working));
    free(pool->threads);
    free(pool);
}
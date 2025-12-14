#ifndef DATA_H
#define DATA_H

#include "la/linalg.h"

// Dataset structure
typedef struct {
    Matrix *X;      // Features matrix (n_samples, n_features)
    Matrix *Y;      // Target matrix (n_samples, n_outputs)
    int n_samples;
    int n_features;
    int n_outputs;
} Dataset;

/**
 * Load CSV file into a Dataset
 * @param filepath path to CSV file
 * @param n_outputs number of output columns (from the end)
 * @param has_header 1 if CSV has header row, 0 otherwise
 * @return pointer to Dataset, or NULL on failure
 */
Dataset* load_csv(const char *filepath, int n_outputs, int has_header);

/**
 * Free dataset memory
 * @param data pointer to Dataset
 */
void dataset_free(Dataset *data);

/**
 * Print dataset info
 * @param data pointer to Dataset
 * @param name name/label for the dataset
 */
void dataset_info(const Dataset *data, const char *name);

/**
 * Normalize features using min-max scaling to [0, 1]
 * @param data pointer to Dataset (modified in place)
 */
void normalize_minmax(Dataset *data);

/**
 * Normalize features using z-score (mean=0, std=1)
 * @param data pointer to Dataset (modified in place)
 */
void normalize_zscore(Dataset *data);

/**
 * Split dataset into train and validation sets
 * @param data source dataset
 * @param train output pointer for training dataset
 * @param val output pointer for validation dataset
 * @param val_ratio ratio of data for validation (0.0 to 1.0)
 */
void dataset_split(const Dataset *data, Dataset **train, Dataset **val, double val_ratio);

/**
 * Shuffle dataset rows randomly
 * @param data pointer to Dataset (modified in place)
 */
void dataset_shuffle(Dataset *data);

/**
 * Count lines in a file (for pre-allocation)
 * @param filepath path to file
 * @return number of lines, or -1 on error
 */
int count_lines(const char *filepath);

/**
 * Count columns in CSV (comma-separated)
 * @param filepath path to CSV file
 * @return number of columns, or -1 on error
 */
int count_columns(const char *filepath);

/**
 * Create a sample CSV for testing
 * @param filepath path to create CSV
 * @param n_samples number of samples
 * @param n_features number of features
 */
void create_sample_csv(const char *filepath, int n_samples, int n_features);

#endif // DATA_H

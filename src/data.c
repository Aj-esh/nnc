#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_LINE_LENGTH 65536
#define MAX_FIELD_LENGTH 256

int count_lines(const char *filepath) {
    FILE *fp = fopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file '%s'\n", filepath);
        return -1;
    }
    
    int count = 0;
    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), fp)) {
        // Skip empty lines
        if (line[0] != '\n' && line[0] != '\r' && line[0] != '\0') {
            count++;
        }
    }
    
    fclose(fp);
    return count;
}

int count_columns(const char *filepath) {
    FILE *fp = fopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file '%s'\n", filepath);
        return -1;
    }
    
    char line[MAX_LINE_LENGTH];
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return -1;
    }
    
    int count = 1;
    for (int i = 0; line[i] != '\0' && line[i] != '\n' && line[i] != '\r'; i++) {
        if (line[i] == ',') {
            count++;
        }
    }
    
    fclose(fp);
    return count;
}

static double parse_field(const char *field) {
    // Handle empty fields
    if (field == NULL || field[0] == '\0') {
        return 0.0;
    }
    
    // Parse as double
    char *endptr;
    double val = strtod(field, &endptr);
    
    // Check if parsing was successful
    if (endptr == field) {
        // Not a number, return 0
        return 0.0;
    }
    
    return val;
}

static int parse_csv_line(char *line, double *values, int max_cols) {
    int col = 0;
    char *token = line;
    char *comma;
    
    while (token && col < max_cols) {
        // Find next comma or end of line
        comma = strchr(token, ',');
        
        if (comma) {
            *comma = '\0';
        }
        
        // Trim leading whitespace
        while (*token == ' ' || *token == '\t') token++;
        
        // Remove trailing whitespace/newline
        char *end = token + strlen(token) - 1;
        while (end > token && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) {
            *end = '\0';
            end--;
        }
        
        values[col] = parse_field(token);
        col++;
        
        if (comma) {
            token = comma + 1;
        } else {
            break;
        }
    }
    
    return col;
}

Dataset* load_csv(const char *filepath, int n_outputs, int has_header) {
    // Count lines and columns
    int total_lines = count_lines(filepath);
    int total_cols = count_columns(filepath);
    
    if (total_lines <= 0 || total_cols <= 0) {
        fprintf(stderr, "Error: Invalid CSV file '%s'\n", filepath);
        return NULL;
    }
    
    int n_samples = has_header ? total_lines - 1 : total_lines;
    int n_features = total_cols - n_outputs;
    
    if (n_features <= 0 || n_outputs <= 0) {
        fprintf(stderr, "Error: Invalid column configuration (features=%d, outputs=%d)\n", 
                n_features, n_outputs);
        return NULL;
    }
    
    // Allocate dataset
    Dataset *data = malloc(sizeof(Dataset));
    if (!data) return NULL;
    
    data->n_samples = n_samples;
    data->n_features = n_features;
    data->n_outputs = n_outputs;
    
    data->X = create_matrix(n_samples, n_features);
    data->Y = create_matrix(n_samples, n_outputs);
    
    if (!data->X || !data->Y) {
        dataset_free(data);
        return NULL;
    }
    
    // Open file and parse
    FILE *fp = fopen(filepath, "r");
    if (!fp) {
        dataset_free(data);
        return NULL;
    }
    
    char line[MAX_LINE_LENGTH];
    double *values = malloc(total_cols * sizeof(double));
    if (!values) {
        fclose(fp);
        dataset_free(data);
        return NULL;
    }
    
    // Skip header if present
    if (has_header) {
        if (!fgets(line, sizeof(line), fp)) {
            free(values);
            fclose(fp);
            dataset_free(data);
            return NULL;
        }
    }
    
    // Parse data rows
    int row = 0;
    while (fgets(line, sizeof(line), fp) && row < n_samples) {
        // Skip empty lines
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') {
            continue;
        }
        
        int cols_parsed = parse_csv_line(line, values, total_cols);
        
        if (cols_parsed < total_cols) {
            fprintf(stderr, "Warning: Row %d has fewer columns than expected (%d < %d)\n",
                    row + 1, cols_parsed, total_cols);
        }
        
        // Copy features (first n_features columns)
        for (int j = 0; j < n_features; j++) {
            data->X->data[row * n_features + j] = values[j];
        }
        
        // Copy outputs (last n_outputs columns)
        for (int j = 0; j < n_outputs; j++) {
            data->Y->data[row * n_outputs + j] = values[n_features + j];
        }
        
        row++;
    }
    
    free(values);
    fclose(fp);
    
    // Adjust if fewer rows were read
    if (row < n_samples) {
        fprintf(stderr, "Warning: Only read %d rows out of expected %d\n", row, n_samples);
        data->n_samples = row;
        // Note: matrices still have original size, but we track actual count
    }
    
    return data;
}

void dataset_free(Dataset *data) {
    if (!data) return;
    
    if (data->X) free_matrix(data->X);
    if (data->Y) free_matrix(data->Y);
    free(data);
}

void dataset_info(const Dataset *data, const char *name) {
    if (!data) {
        printf("Dataset '%s': NULL\n", name);
        return;
    }
    
    printf("Dataset '%s':\n", name);
    printf("  Samples:  %d\n", data->n_samples);
    printf("  Features: %d\n", data->n_features);
    printf("  Outputs:  %d\n", data->n_outputs);
    
    // Print first few samples
    int preview = (data->n_samples < 3) ? data->n_samples : 3;
    printf("  First %d samples:\n", preview);
    
    for (int i = 0; i < preview; i++) {
        printf("    X[%d]: [", i);
        int feat_preview = (data->n_features < 5) ? data->n_features : 5;
        for (int j = 0; j < feat_preview; j++) {
            printf("%.4f", data->X->data[i * data->n_features + j]);
            if (j < feat_preview - 1) printf(", ");
        }
        if (data->n_features > 5) printf(", ...");
        printf("]  Y[%d]: [", i);
        for (int j = 0; j < data->n_outputs; j++) {
            printf("%.4f", data->Y->data[i * data->n_outputs + j]);
            if (j < data->n_outputs - 1) printf(", ");
        }
        printf("]\n");
    }
}

void normalize_minmax(Dataset *data) {
    if (!data || !data->X) return;
    
    int n = data->n_samples;
    int f = data->n_features;
    
    for (int j = 0; j < f; j++) {
        // Find min and max for this feature
        double min_val = data->X->data[j];
        double max_val = data->X->data[j];
        
        for (int i = 1; i < n; i++) {
            double val = data->X->data[i * f + j];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        
        // Normalize
        double range = max_val - min_val;
        if (range > 0) {
            for (int i = 0; i < n; i++) {
                data->X->data[i * f + j] = (data->X->data[i * f + j] - min_val) / range;
            }
        }
    }
}

void normalize_zscore(Dataset *data) {
    if (!data || !data->X) return;
    
    int n = data->n_samples;
    int f = data->n_features;
    
    for (int j = 0; j < f; j++) {
        // Compute mean
        double mean = 0.0;
        for (int i = 0; i < n; i++) {
            mean += data->X->data[i * f + j];
        }
        mean /= n;
        
        // Compute std
        double var = 0.0;
        for (int i = 0; i < n; i++) {
            double diff = data->X->data[i * f + j] - mean;
            var += diff * diff;
        }
        double std = sqrt(var / n);
        
        // Normalize
        if (std > 0) {
            for (int i = 0; i < n; i++) {
                data->X->data[i * f + j] = (data->X->data[i * f + j] - mean) / std;
            }
        }
    }
}

void dataset_shuffle(Dataset *data) {
    if (!data || data->n_samples <= 1) return;
    
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    
    int n = data->n_samples;
    int f = data->n_features;
    int o = data->n_outputs;
    
    // Fisher-Yates shuffle
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        
        // Swap rows i and j in X
        for (int k = 0; k < f; k++) {
            double tmp = data->X->data[i * f + k];
            data->X->data[i * f + k] = data->X->data[j * f + k];
            data->X->data[j * f + k] = tmp;
        }
        
        // Swap rows i and j in Y
        for (int k = 0; k < o; k++) {
            double tmp = data->Y->data[i * o + k];
            data->Y->data[i * o + k] = data->Y->data[j * o + k];
            data->Y->data[j * o + k] = tmp;
        }
    }
}

void dataset_split(const Dataset *data, Dataset **train, Dataset **val, double val_ratio) {
    if (!data || !train || !val) return;
    
    int n = data->n_samples;
    int f = data->n_features;
    int o = data->n_outputs;
    
    int n_val = (int)(n * val_ratio);
    int n_train = n - n_val;
    
    // Allocate train dataset
    *train = malloc(sizeof(Dataset));
    (*train)->n_samples = n_train;
    (*train)->n_features = f;
    (*train)->n_outputs = o;
    (*train)->X = create_matrix(n_train, f);
    (*train)->Y = create_matrix(n_train, o);
    
    // Allocate val dataset
    *val = malloc(sizeof(Dataset));
    (*val)->n_samples = n_val;
    (*val)->n_features = f;
    (*val)->n_outputs = o;
    (*val)->X = create_matrix(n_val, f);
    (*val)->Y = create_matrix(n_val, o);
    
    // Copy training data
    memcpy((*train)->X->data, data->X->data, n_train * f * sizeof(double));
    memcpy((*train)->Y->data, data->Y->data, n_train * o * sizeof(double));
    
    // Copy validation data
    memcpy((*val)->X->data, data->X->data + n_train * f, n_val * f * sizeof(double));
    memcpy((*val)->Y->data, data->Y->data + n_train * o, n_val * o * sizeof(double));
}

void create_sample_csv(const char *filepath, int n_samples, int n_features) {
    FILE *fp = fopen(filepath, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot create file '%s'\n", filepath);
        return;
    }
    
    srand((unsigned int)time(NULL));
    
    // Write header
    for (int j = 0; j < n_features; j++) {
        fprintf(fp, "feature_%d,", j + 1);
    }
    fprintf(fp, "target\n");
    
    // Generate random weights for true function
    double *weights = malloc(n_features * sizeof(double));
    for (int j = 0; j < n_features; j++) {
        weights[j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    double bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    
    // Generate samples: y = sum(x_i * w_i) + bias + noise
    for (int i = 0; i < n_samples; i++) {
        double y = bias;
        for (int j = 0; j < n_features; j++) {
            double x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            fprintf(fp, "%.6f,", x);
            y += x * weights[j];
        }
        // Add small noise
        double noise = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        fprintf(fp, "%.6f\n", y + noise);
    }
    
    free(weights);
    fclose(fp);
    
    printf("Created sample CSV: %s (%d samples, %d features)\n", 
           filepath, n_samples, n_features);
}

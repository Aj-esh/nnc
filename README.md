# nnc

A neural network library for regression, written in C.

## Structure

```
src/
  main.c      - Entry point, training loop, CLI
  nn.c        - Neural network forward/backward logic
  train.c     - Training utilities, metrics, synthetic data
  val.c       - Validation and data splitting
  data.c      - CSV data loading and normalization
  act.c       - Activation functions (ReLU)
  optax.c     - Optimizers (SGD, Adam)
  la/         - Linear algebra routines
  poolla/     - Thread pool for parallelism
include/      - Header files
```

## How It Works

- **Architecture:** 4-layer fully connected neural network with ReLU activations and a linear output layer.
- **Training:** Uses mean squared error (MSE) loss and supports SGD (default) or Adam optimizers.
- **Parallelism:** Matrix operations are parallelized using a thread pool for performance.
- **Metrics:** Tracks loss, RMSE, and R² during training and validation.
- **Data:** Expects CSV files for input, with features first and target last.

## Data Format

- CSV files **must** have a header row.
- All feature columns come first; the target column is last.
- Example:
  ```
  feature_1,feature_2,feature_3,target
  0.5,0.3,0.1,1.2
  0.2,0.8,0.4,2.1
  ...
  ```
- Both training and test/validation files should follow this format.

## Usage

### 1. Build

```sh
make
```

### 2. Run

```sh
./build/nnc
```

You will be prompted for the paths to your training and test CSV files.

### 3. Example Session

```
Enter training data path: /path/to/train.csv
Enter test data path: /path/to/test.csv
```

The program will load, normalize, train, and print metrics for each epoch.

---

**Note:**  
- Data is normalized using z-score normalization.
- Adjust network dimensions, epochs, and learning rate in `src/main.c` as needed.

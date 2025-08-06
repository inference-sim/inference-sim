# Optimizer Documentation

This document provides an overview of how to use and configure the inference simulation optimizer.

## Installation

Before running the optimizer, you need to set up a Python virtual environment and install the required dependencies. You can use either `pip` or `uv`.

### Using `pip`

1.  **Create a virtual environment:**

    ```bash
    python -m venv .venv
    ```

2.  **Activate the virtual environment:**

    -   On macOS and Linux:
        ```bash
        source .venv/bin/activate
        ```
    -   On Windows:
        ```bash
        .venv\Scripts\activate
        ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

### Using `uv`

`uv` is a fast Python package installer and resolver.

1.  **Create and activate a virtual environment:**

    ```bash
    uv venv
    source .venv/bin/activate
    ```

2.  **Install the required packages:**

    ```bash
    uv sync
    ```

## Running the Optimizer

The main script for running the optimizer is `run_optimizer.py`. It takes several command-line arguments to control the optimization process.

### Usage

You can run the optimizer from your terminal using the following command:

```bash
python run_optimizer.py [OPTIONS]
```

### Command-Line Arguments

-   `--config_file`: The path to the YAML configuration file for the optimizer.
    -   Type: `string`
    -   Default: `optimizer_config.yaml`
-   `--n_trials`: The number of optimization trials to run.
    -   Type: `int`
    -   Default: `100`
-   `--save_study`: A flag to indicate whether to save the study results to a JSON file. Use `--save_study` to enable saving and `--no-save_study` to disable it.
    -   Type: `boolean`
    -   Default: `True`

### Example

To run the optimizer with a custom configuration file and 200 trials, you would use:

```bash
python run_optimizer.py --config_file my_config.yaml --n_trials 200
```

## Configuration (`optimizer_config.yaml`)

The behavior of the optimizer is controlled by a YAML configuration file. Here is a breakdown of the different sections in the configuration file:

-   `tokens_dir`: Path to the tokenized data file.
-   `sim_dir`: Directory where simulation outputs will be stored.
-   `vllm_dir`: Directory containing the vLLM reference log files for comparison.
-   `kv_blocks`: The number of KV cache blocks available.
-   `error_function`: The error metric to use for optimization. Can be `mape` (Mean Absolute Percentage Error) or `mse` (Mean Squared Error).
-   `seed`: An integer for the random seed for reproducibility.

### `pbounds`

This section defines the parameter bounds for the optimization variables. Each key is a parameter name, and the value is a list `[min, max]` representing the search space for that parameter during optimization. These are normalized bounds, typically between 0 and 1.

### `scaling`

This section provides scaling factors for each parameter. The values suggested by the optimizer within the `pbounds` range are multiplied by these scaling factors to get the actual parameter values used in the simulation.

### `train_config` and `test_config`

These sections define the configurations for the training (optimization) and testing (evaluation) phases. They specify the parameters for generating the workload for the simulation, including:

-   `num_prompts`: A list of the number of prompts to simulate.
-   `request_rate`: A list of request rates to simulate.
-   `temperature`: A list of sampling temperatures.
-   `max_num_batched_tokens`: A list of maximum number of tokens to batch.
-   `long_prefill_token_threshold`: A list of thresholds for long prefill tokens.
-   `datasets`: A list of datasets to use for generating requests.

## Optimizer Structure (`optimizer.py`)

The `optimizer.py` file contains the core logic for the optimization process.

### `parse_optimizer_config(config_file)`

This function reads the YAML configuration file and returns a dictionary of parameters that are then passed to the `InferenceSimOptimizer` during initialization.

### `InferenceSimOptimizer` Class

This class encapsulates all the functionality for the optimization task.

-   `__init__(...)`: The constructor initializes the optimizer with the parameters from the configuration file. It sets up default values if they are not provided.
-   `_create_objective_function(...)`: This method creates the objective function that Optuna will minimize. The objective function:
    1.  Takes a `trial` object from Optuna.
    2.  Suggests parameter values for the trial.
    3.  Constructs and runs a simulation command (`request_rate_sweep.py`) with the suggested parameters.
    4.  Parses the simulation results and the vLLM reference results.
    5.  Calculates the error between the simulation and reference results using the specified `error_function`.
    6.  Returns the calculated error.
-   `optimize(n_trials, ...)`: This method sets up and runs the Optuna study. It creates a study, calls the objective function for `n_trials`, and stores the best results.
-   `evaluate(...)`: After optimization, this method evaluates the best-found parameters on the `test_config` to see how well they generalize.
-   `save_study(...)`: This method saves the results of the study, including the best parameters, train error, and evaluation error, to a JSON file.

# inference-sim

Discrete event simulator for inference platforms.

## Getting started

> Go >= 1.21 is required.

## Testing Pipeline

#### Install and build inference-sim

```shell
git clone git@github.com:inference-sim/inference-sim.git
cd inference-sim
go build -o simulation_worker main.go
```

#### Run test data collection and collect results

Collect data from [vllm-data-collection](https://github.com/inference-sim/vllm-data-collection) and post process results to get vllm aggregate metrics. Remember to use `mode=test`. 

#### Tokenize prompts and run BLIS sweep

```shell
python offline_tokenizer.py --results_path <path to results_new/scenario in vllm-data-collection>
```

Next, modify `experiment_constants_test.py` as required to specify test configs to BLIS. Finally, run BLIS sweeps.

```shell
python request_rate_sweep.py --mode test
```

#### Analyze results for accuracy, MAE

```shell
python analyze_sim_results.py
```

## Inference Pipeline

#### Install inference-sim

```shell
pip install git+https://github.com/inference-sim/inference-sim.git
```

#### Import and utilize BLIS related modules

```shell
spec = importlib.util.find_spec("run_blis_sweep")
SIMULATION_BASE_DIR = Path(spec.origin).parent
CONSTANTS_PATH = SIMULATION_BASE_DIR / "experiment_constants_inference.py"
RESULTS_PATH = SIMULATION_BASE_DIR / "results/sweep_params/simulator_inference_results.csv"
```

#### Set the env variable expected by the BLIS script

```shell
os.environ["SIMULATION_BASE_DIR"] = str(SIMULATION_BASE_DIR)
runpy.run_module("run_blis_sweep", run_name="__main__")
```

#### Run BLIS sweep

Modify `experiment_constants_inference.py` as required to specify inference configs to BLIS. Next, run the simulator through:

```shell
python run_blis_sweep.py
```

<!-- ### Obtain best model for your model x GPU (blackbox docs - retaining in comments for future reference)

For this, you need the vllm profiling data from actual runs of the model x GPU you want to test. This is an example for `Qwen/Qwen-2-1.5b` on NVIDIA L4 GPU, but a similar approach would be followed for any other LLM. Let us say you keep the vllm profiling results in `results/qwen2-1.5b/sweep/`. 

* Go to `optimizer_qwen2-1.5b_config.yaml` and edit the `vllm_dir` field to `./results/qwen2-1.5b/sweep/`.
* In the same file, edit `tokens_dir` field to where you stored the output tokens for the model: e.g, `data/output_tokens_2025-06-30_tokenized.json`
* Now run the blackbox optimizer:

```shell
python run_optimizer.py --config_file optimizer_qwen2-1.5b_config.yaml
```
-->

## Help

```shell
go run main.go run --help
```

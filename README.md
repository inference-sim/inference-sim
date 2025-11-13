# inference-sim

Discrete event simulator for inference platforms.

## Getting started

> Go >= 1.21 is required.

## simple test

### before this PR
- requestgenconfig.yaml is as follows
```yaml
format: "GuideLLM"
seed: 42
rate:
  arrival_type: "Constant"
  rate: 10
  max-requests: 2000
data:
  prompt_tokens: 2048
  prompt_tokens_stdev: 1024
  prompt_tokens_min: 2
  prompt_tokens_max: 50000
  output_tokens: 256
  output_tokens_stdev: 128
  output_tokens_min: 1
  output_tokens_max: 50000
  prefix_tokens: 614
```

./simulation_worker run --max-num-running-reqs 8192 --total-kv-blocks 64173 --max-num-scheduled-tokens 4096 --block-size-in-tokens 16 --horizon 922337203685477580 --regression-coeffs 0.006857138483067575,3.82496271961672e-05,5.517393223466091e-05 --reqgen-config-path requestgenconfig.yaml --long-prefill-token-threshold 0 --queuing-coeffs 4.87273847,8956.34427 --finished-coeffs 0.0470950911,1297.25103 --log warn

=== Simulation Metrics ===
Completed Requests   : 1284
Request Rate(req/s)  : 10
Total Input Tokens   : 5282504
Total Output Tokens  : 327058
Simulation Duration(s): 7.991
vLLM estimated Duration(s): 199.921
Request throughput (req/s):  : 6.423
Mean E2E(ms)     : 30550.837
Median E2E(ms)   : 28328.736
P99 E2E(ms)      : 74076.961
Mean Active Steps     : 255.143
Sim Ended Time : 199920995
Avg KV Blocks Usage : 29241.280
Peak KV Usage       : 64008 blocks
=== Saturation Metrics ===
Throughput to arrival rate ratio:  : 0.642


## after this PR
WARN[0007] Not enough KV cache space to allocate 195 new blocks 
WARN[0007] [Preemption]                                 
=== Simulation Metrics ===
Completed Requests   : 1284
Request Rate(req/s)  : 0
Total Input Tokens   : 5282504
Total Output Tokens  : 327058
Simulation Duration(s): 7.802
vLLM estimated Duration(s): 0.000
Request throughput (req/s):  : +Inf
Mean E2E(ms)     : 30550.837
Median E2E(ms)   : 28328.736
P99 E2E(ms)      : 74076.961
Mean Active Steps     : 255.143
Sim Ended Time : 0
Avg KV Blocks Usage : +Inf
Peak KV Usage       : 64008 blocks
=== Saturation Metrics ===
Throughput to arrival rate ratio:  : +Inf
=== Simulation Metrics ===
Completed Requests   : 0
Request Rate(req/s)  : 10
Total Input Tokens   : 0
Total Output Tokens  : 0
Simulation Duration(s): 7.802
vLLM estimated Duration(s): 199.921

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
python run_blis_sweep.py --mode test
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

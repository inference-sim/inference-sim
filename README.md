# inference-sim

Discrete event simulator for inference platforms.

## Getting started

> Go >= 1.21 is required.

### Compile simulator code

```shell
git clone git@github.com:inference-sim/inference-sim.git
cd inference-sim
go build -o simulation_worker main.go
```

### Download and preprocess dataset

* Make a `data/` directory. Put the vLLM generated non-tokenized output JSON as `output_tokens_2025-06-30.json` under it.
* Now tokenize the output tokens by running the following:

`python offline_tokenizer.py --model_name <LLM_name> --input_filepath data/output_tokens_2025-06-30.json --output_filepath data/output_tokens_2025-06-30_tokenized.json`

### Obtain best model for your model x GPU

```shell
python blackbox_optimization.py
```

Copy the coefficients that print out in stdout into clipboard. The output is of the format:

```
{'sum_decode_tokens': a, 'sum_prefill_tokens': b, 'num_prefills': c, 'intercept': d, 'step_constant': e, 'vllm_overhead': f}
```

Now, run the following:

```shell
python request_rate_sweep.py --rates 32 --long_prefill_token_thresholds 16 32 64 --max_num_batched_tokens 256 512 --input_filename data/output_tokens_2025-06-30_tokenized.json --num_requests 400 --output_dir results/sweep --regression_coeffs a,b,c,d --schedule_time e --update_time 0 ----vllm_overhead_time f --queue_overhead_time 0
```
## Help

```shell
go run main.go run --help
```

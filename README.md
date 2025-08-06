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

**Input tokens**

Download the ShareGPT dataset locally:

```shell
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

**Output tokens**

* Make a `data/` directory. Put the vLLM generated non-tokenized output JSON as `output_tokens_2025-06-30.json` under it.
* Now tokenize the output tokens by running the following:

`python offline_tokenizer.py --model_name <LLM_name> --input_filepath data/output_tokens_2025-06-30.json --output_filepath data/output_tokens_2025-06-30_tokenized.json`

### Obtain best model for your model x GPU

For this, you need the vllm profiling data from actual runs of the model x GPU you want to test. This is an example for `Qwen/Qwen-2-1.5b` on NVIDIA L4 GPU, but a similar approach would be followed for any other LLM. Let us say you keep the vllm profiling results in `results/qwen2-1.5b/sweep/`. 

* Go to `optimizer_qwen2-1.5b_config.yaml` and edit the `vllm_dir` field to `./results/qwen2-1.5b/sweep/`.
* In the same file, edit `tokens_dir` field to where you stored the output tokens for the model: e.g, `data/output_tokens_2025-06-30_tokenized.json`
* Now run the blackbox optimizer:

```shell
python run_optimizer.py --config_file optimizer_qwen2-1.5b_config.yaml
```

Copy the coefficients that print out in stdout into clipboard. The output is of the format:

```
--regression_coeffs 5.541779348743728e-05,3.056158363878956e-05,0,3.908682147124097e-05,0,0,0,0,0,0,0,0,0,0,0 --schedule_time 13787 --update_time 0 --queue_overhead_time 0 --vllm_overhead_time 5999
```

Now, run the following:

```shell
python request_rate_sweep.py --rates 32 --long_prefill_token_thresholds 16 32 64 --max_num_batched_tokens 256 512 --input_filename data/output_tokens_2025-06-30_tokenized.json --num_requests 400 --output_dir results/sweep --regression_coeffs 5.541779348743728e-05,3.056158363878956e-05,0,3.908682147124097e-05,0,0,0,0,0,0,0,0,0,0,0 --schedule_time 13787 --update_time 0 --queue_overhead_time 0 --vllm_overhead_time 5999
```

## Help

```shell
go run main.go run --help
```

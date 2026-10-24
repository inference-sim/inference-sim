# Oracle

vLLM v0.32.0's own classes, driven through a scenario on CPU, with only the GPU replaced. See
[docs/testing.md](../docs/testing.md).

```sh
# A vLLM checkout at the pinned tag (for its test helpers) and a Python with vLLM 0.32.0:
python -I oracle/check_mocks.py oracle/mocks.json
python -I oracle/run_scheduler.py <vllm-checkout> oracle/scenarios/<scenario>.json > trace.json
```

## Scenario

```json
{
  "name": "...",
  "config": {"max_num_seqs": 2, "max_num_batched_tokens": 8, "block_size": 4, "num_blocks": 8,
             "max_model_len": 64, "long_prefill_token_threshold": 0,
             "enable_prefix_caching": false, "scheduling_policy": "fcfs"},
  "requests": [{"id": "A", "arrival_step": 0, "prompt": [1, 2, 3], "output": [7, 8],
                "max_tokens": 16, "priority": 0}]
}
```

`num_blocks` is the KV capacity, fixed so a cost model cannot change admission. `output` is the
assigned output (the GPU mock samples it, the last token as EOS). `max_tokens`, optional, is a
client budget.

## Trace

Per step: `scheduled` (tokens per request), `preempted`, `finished`, `free_blocks`,
`num_running`, `num_waiting`, `kv_usage`, and `blocks` (each running request's blocks as
`[hash or null, ref_cnt]`, never block ids), plus `provenance`.

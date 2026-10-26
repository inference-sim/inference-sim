# Oracle

vLLM v0.32.0's own classes, driven through a scenario on CPU, with only the GPU replaced. See
[docs/testing.md](../docs/testing.md).

```sh
# A vLLM checkout at the pinned tag (for its test helpers) and a Python with vLLM 0.32.0:
python -I oracle/check_mocks.py oracle/mocks.json
ORACLE_MODEL=<local snapshot of facebook/opt-125m@27dcfa74, config files only> \
ORACLE_MODEL_REVISION=27dcfa74d334bc871f3234de431e71c6eeba5dd6 HF_HUB_OFFLINE=1 \
  python -I oracle/run_scheduler.py <vllm-checkout> oracle/scenarios/<scenario>.json > trace.json
```

The scheduler helper builds a `ModelConfig`, which reads a model's `config.json` but loads no
weights; the scenario's fixed `num_blocks` and `block_size` govern every decision.

## Environment

`oracle/requirements.lock` is the whole environment, every package pinned (Linux x86_64, CPU):
`pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cpu -r
oracle/requirements.lock`, then `python -I oracle/check_lock.py oracle/requirements.lock <(pip
freeze)` confirms it. The runners also read:

| Variable | What |
|---|---|
| `ORACLE_MODEL` | a local snapshot of `facebook/opt-125m@27dcfa74` (config files only) |
| `ORACLE_MODEL_REVISION` | its revision, recorded in provenance |
| `ORACLE_LOCKFILE` | the `pip freeze` of the environment, whose sha256 provenance records |
| `HF_HUB_OFFLINE=1` | no Hub access while a scenario runs |
| `BLIS_VLLM_ALLOW_GOLDEN_UPDATE=1` | lets a runner write into a `testdata/golden*` directory (then add a `golden-reviewed:` trailer) |

## Runners and tests, by tier

Each runner takes `<vllm checkout> <scenario.json | dir> [out dir]` (a directory writes
`<name>.trace.json` beside each scenario, or into the out dir). The Go side reads committed
goldens; its fuzz tests read a corpus directory named by an environment variable and skip, saying
so, when it is unset.

| Tier | Runner | Scenarios | Fuzz generator | Go test (corpus variable) |
|---|---|---|---|---|
| T2 scheduler and KV | `run_scheduler.py` | `scenarios/` | `gen_fuzz.py <dir> <first> <count>` | `./core/sched` (`BLIS_VLLM_FUZZ_DIR`) |
| T3 EngineCore | `run_enginecore.py` | `scenarios-t3/` | `gen_fuzz_t3.py` | `./core/engine` (`BLIS_VLLM_T3_FUZZ_DIR`) |
| T3 data parallel | `run_dp.py` | `scenarios-dp/` | `gen_fuzz_dp.py` | `./core/engine` (`BLIS_VLLM_DP_FUZZ_DIR`) |
| T3 data parallel, async (a batch queue per rank) | `run_dp.py` | `scenarios-dp-async/` | | `./core/engine`, `./core/vllm` |
| T4 NIXL connector | `run_connector.py` | `scenarios/connector/` | `gen_connector_fuzz.py` (seeds 10000+ reach `max_model_len`) | `./core/connector/nixl` (`BLIS_VLLM_CONNECTOR_FUZZ_DIR`) |
| T5 metrics | `run_metrics.py` | `scenarios/` (tier-2 scenarios) | `gen_fuzz.py` | `./core/metrics` (`BLIS_VLLM_METRICS_FUZZ_DIR`) |

The guards: `check_mocks.py` (every write a runner makes into vLLM is listed in `mocks.json`),
`check_goldens.py <base> <head>` (a golden's decisions change only with a `golden-reviewed:`
trailer), `compare_goldens.py` / `compare_goldens_t3.py` (each golden is what the pinned release
produces now), `check_manifest.py` (tier 5's manifest against vLLM's live registry) and
`test_checks.py` (each guard fails on what it guards). The mutation sweeps: `mutate.py` (T2),
`mutate_t3.py`, `mutate_t4.py` (with the registry), `mutate_e24.py` (the engine interface) and
`mutate_metrics.py` (T5).

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

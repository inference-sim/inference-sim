# Oracle

vLLM v0.32.0's own classes, driven through a scenario on CPU, with only the GPU replaced. See
[docs/testing.md](../docs/testing.md).

```sh
# A vLLM checkout at the pinned tag (for its test helpers) and a Python with vLLM 0.32.0:
python -I oracle/check_mocks.py oracle/mocks.json
ORACLE_MODEL=<local snapshot of facebook/opt-125m@27dcfa74, config files only> \
ORACLE_MODEL_REVISION=27dcfa74d334bc871f3234de431e71c6eeba5dd6 HF_HUB_OFFLINE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  python -I oracle/run_scheduler.py <vllm-checkout> oracle/scenarios/<scenario>.json > trace.json
```

The scheduler helper builds a `ModelConfig`, which reads a model's `config.json` but loads no
weights; the scenario's fixed `num_blocks` and `block_size` govern every decision.

## Environment

`oracle/requirements.lock` is the whole environment, every package pinned (Linux x86_64, CPU).
`oracle/lock.sh` writes the same pins with every archive's sha256 into
`oracle/requirements-hashed.txt`, and pip's own release into `oracle/requirements-pip.txt`
(X21). Install with `python -m pip install --require-hashes -r oracle/requirements-pip.txt`,
then `pip install --require-hashes --no-deps --extra-index-url https://download.pytorch.org/whl/cpu
-r oracle/requirements-hashed.txt`; `python -I oracle/check_lock.py oracle/requirements.lock <(pip
freeze)` confirms it. After changing the lock, run `oracle/lock.sh` and commit all three files;
`python -I oracle/check_lock.py --hashed oracle/requirements.lock oracle/requirements-hashed.txt`,
which CI runs first, fails while they disagree. The runners also read:

| Variable | What |
|---|---|
| `ORACLE_MODEL` | a local snapshot of `facebook/opt-125m@27dcfa74` (config files only) |
| `ORACLE_MODEL_REVISION` | its revision, recorded in provenance |
| `ORACLE_LOCKFILE` | the `pip freeze` of the environment, whose sha256 provenance records |
| `HF_HUB_OFFLINE=1` | no Hub access while a scenario runs |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` | lets a scenario state a `max_model_len` above the helper model's 2048 positions (its `config.json`), as a dashboard deployment's start-up check needs; the window reaches only the scheduler's length checks and the KV start-up check, never a weight |
| `BLIS_VLLM_ALLOW_GOLDEN_UPDATE=1` | lets a runner write into a `testdata/golden*` directory (then add a `golden-reviewed: <trace>: <reason>` trailer for each trace the commit changes) |

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
| T4 NIXL connector | `run_connector.py` | `scenarios/connector/` | `gen_connector_fuzz.py` (seeds 10000+ reach `max_model_len`; 20000+ add hybrid groups and drafters; 30000+ the batch queue) | `./core/connector/nixl` (`BLIS_VLLM_CONNECTOR_FUZZ_DIR`) |
| T4 OffloadingConnector (CPU tier, LRU) | `run_offloading.py` | `scenarios/offloading/` | `gen_offloading_fuzz.py` (seeds 20000+ add sliding-window groups) | `./core/connector/offloading` (`BLIS_VLLM_OFFLOAD_FUZZ_DIR`) |
| T5 metrics | `run_metrics.py` | `scenarios/` (tier-2 scenarios) | `gen_fuzz.py` | `./core/metrics` (`BLIS_VLLM_METRICS_FUZZ_DIR`) |

The guards: `check_mocks.py` (every write a runner makes into vLLM is listed in `mocks.json`),
`check_goldens.py <base> <head>` (a golden's decisions change only in a commit whose
`golden-reviewed: <trace>: <reason>` trailer names that trace and gives the reason), `compare_goldens.py` / `compare_goldens_t3.py` (each golden is what the pinned release
produces now), `check_manifest.py` (tier 5's manifest against vLLM's live registry) and
`test_checks.py` (each guard fails on what it guards). The mutation sweeps: `mutate.py` (T2),
`mutate_t3.py`, `mutate_t4.py` (with the registry), `mutate_offloading.py` (the OffloadingConnector), `mutate_e24.py` (the engine interface) and
`mutate_metrics.py` (T5).

## Block-size oracle

`oracle/blocksize/oracle.py` answers a different question from the tiers above: what block size
vLLM resolves an unstated `block_size` to. vLLM chooses it from the attention backends it
selects for the model, GPU and KV dtype (`Platform.update_block_size_for_backend`,
`vllm/platforms/interface.py:757-781`; the backend priority list `platforms/cuda.py:83-180`; each
backend's preferred and supported sizes). The oracle runs vLLM's own `CudaPlatform.get_valid_backends`
and `_preferred_block_size_for_backends` on a CPU, with the GPU-only wheels replaced
(`oracle/blocksize/stubs.py`; each replacement is listed in the table, as FID-1 requires).

A CPU cannot discover four facts: whether vLLM's wheel ships FlashAttention 3 and 4, whether
FlashInfer's TRTLLM kernels are usable, and whether the wheel carries FlashMLA. The oracle runs
every combination of model, compute capability and KV dtype under every setting of those facts. A
combination resolves only if the size is the same under all of them; otherwise it is refused,
naming the fact. `oracle/blocksize/TABLE.tsv` is the table to read: each combination, its backend,
its size and the size under each setting of the facts. The committed table is
`core/vllm/blocksize_table.json`, which
`core/vllm/blocksize.go` embeds, and CI fails if it differs from what the oracle prints:

```sh
BLIS_ORACLE_CATALOG=<blis-substrate>/catalog/models \
  python -I oracle/blocksize/oracle.py <vllm checkout> table > oracle/blocksize/TABLE.tsv   # for reading
  python -I oracle/blocksize/oracle.py <vllm checkout> json > core/vllm/blocksize_table.json
```

A derived-only model (its licence does not let the catalog redistribute it, as minimax-m2.7)
has no `config.json` in the catalog. The oracle then fetches that file from the `source` and
`revision` in the model's `model.yaml` and refuses it unless its SHA-256 equals `config.sha256`
there, so the oracle reads the model's real file and no attention fact is copied. This needs
network access to huggingface.co (in CI too); a failed fetch fails the run and never drops a model.
`BLIS_ORACLE_FETCH_CACHE` names the cache directory (default: a temporary directory), keyed by
digest and re-verified on every read; `HF_TOKEN`, if set, is sent to huggingface.co only. The file
is never written into the repository or the catalog.

Each combination is also run at every tensor-parallel width from 1 to 16 (`TP_WIDTHS`), and
resolves only if the size is the same at all of them. The table is valid for a width up to 16, and
`ResolveBlockSize` refuses a wider pool, citing blis-vllm#68.

It needs vLLM's `requirements/common.txt` and `torch`; it does not need a GPU. At a new vLLM
release, re-read each model's attention layer (the `LAYERS` table cites the line that builds it)
and rerun it.

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

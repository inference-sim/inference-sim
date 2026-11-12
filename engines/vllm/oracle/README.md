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
| T3/T4 data parallel with NIXL on every rank | `run_dp.py` | `scenarios-dp-nixl/` | | `./core/engine` |
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

## Indexer cache dtype oracle

`oracle/indexer/oracle.py` answers what `attention_config.indexer_kv_dtype` does on DeepSeek V4 and
V4.1: which dtype the sparse indexer's key cache runs in and how many bytes one key record takes.
It takes `AttentionConfig.resolve_indexer_kv_dtype`, `dsa_indexer_uses_fp4`, V4.1's
`_indexer_k_cache_head_dim` and V4's record-width statement out of their modules with `ast`, as
`oracle/lower/extract.py` does, and runs them unmodified over a stub platform
(`oracle/indexer/replacements.json`; `python3 -I oracle/lower/check_replacements.py
oracle/indexer/oracle.py oracle/indexer/replacements.json` checks the list). Its table, every dtype on
SM9x and SM10x, is `rules/v0_32/testdata/vllm_indexer_kv_dtype.json`; CI fails if it differs from
`python -I oracle/indexer/oracle.py <vllm checkout> check rules/v0_32/testdata/vllm_indexer_kv_dtype.json`
(`print` writes it), and `TestIndexerKVDTypeIsWhatVLLMsOwnFunctionsGive` compares the pack with it.

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
naming the fact.

One fact is assumed, not discovered. `ASSUMED_FACTS` in `oracle/blocksize/oracle.py` states, for
compute capability 9.0, that vLLM's wheel ships FlashAttention 3 (`fa3_available`), with its
evidence: the pinned Dockerfile builds for arch 9.0 with CUDA 13.0.3, `CMakeLists.txt` includes the
`vllm_flash_attn` project on every CUDA build, and one recorded Kimi-K2.5 h200 run on
`vllm/vllm-openai:v0.22.0` agrees. The condition that builds `_vllm_fa3_C` is in the external
flash-attention repository, which the pinned tree does not hold, so this is an assumption about
vLLM's release images. For a Hopper combination the size must then be the same only over the
settings that agree with it. The table still shows all 16 settings, and a combination that resolves
only by the assumption carries it (`assumed_facts` in the JSON, a `Source` clause in
`ResolveBlockSize`). The MLA models with KV `auto`, `bfloat16` or `float16` on SM9x (FlashAttention
MLA, size 16) resolve this way; with an fp8 KV they still depend on `flashmla_built`. An engine run
from a build without FA3 is resolved wrongly. `oracle/blocksize/TABLE.tsv` is the table to read: each combination, its backend,
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

## Quantization oracle

`oracle/quantization/oracle.py` runs vLLM's own `ModelConfig` (`vllm/config/model.py:1370-1412`,
with `DeepseekV4FP8Config.override_quantization_method`, `vllm/models/deepseek_v4/quant_config.py:140-157`)
over the catalog's deepseek-v4-pro `config.json` once for each value a deployment can state for
`quantization`, and over the same file without its `quantization_config` as a control. The
committed answers are `oracle/quantization/resolved.json`, which
`compat/kernelmodel/quantization_v4_test.go` holds the kernel's plan to, and CI fails if the file
differs from what the oracle prints:

```sh
BLIS_ORACLE_CATALOG=<blis-substrate>/catalog/models \
  python -I oracle/quantization/oracle.py <vllm checkout> json > oracle/quantization/resolved.json
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
`config.enable_chunked_prefill`, optional, is `SchedulerConfig.enable_chunked_prefill` (default
true; the tier-2 and tier-3 runners and the connector runners read it, the metrics and DP runners do not). With it false vLLM's `SchedulerConfig` refuses a
`max_num_batched_tokens` below `max_model_len`, so a scenario that turns it off states them equal
or in that order.

## Trace

Per step: `scheduled` (tokens per request), `preempted`, `finished`, `free_blocks`,
`num_running`, `num_waiting`, `kv_usage`, and `blocks` (each running request's blocks as
`[hash or null, ref_cnt]`, never block ids), plus `provenance`.

### Data-parallel connector scenarios

`scenarios-dp-nixl/` runs `run_dp.py` with a NIXL connector on every rank. A rank is one DP engine
core: its own vLLM `Scheduler`, KV cache and `NixlConnector` scheduler side, as vLLM gives each rank
its own `engine_id` (`<uuid>_dp<rank>`, `vllm/v1/engine/core.py:1417-1426@5df63a72`). The prefill rank
that ran a request names itself in the request's `kv_transfer_params`, and the decode side pulls that
rank's blocks for that request, so a P/D pair at DP is per rank and per request; the ranks are coupled
only by the DP wave (lockstep steps, dummy batches, the all-reduce of the unfinished flags). The
scenario is a DP scenario (`data_parallel_size`, `dp_sync_interval`, `prefill_schedule_interval`,
`stats_interval`, `stats_lag`, `async_scheduling`) plus:

```json
"connector": {"name": "NixlConnector", "kv_roles": ["kv_producer", "kv_consumer"],
              "kv_load_failure_policy": "fail", "extra_config": {}}
```

`kv_roles` gives one role per rank (`kv_producer`, `kv_consumer` or `kv_both`; `kv_role` gives one for
all). The scheduler settings are the connector scenarios' (`max_model_len` is required,
`enable_prefix_caching` defaults to true). A request takes the connector scenarios' fields
(`remote_prefill` with `recv_delay` and `recv_failed`, `remote_decode` with `send_delay`,
`invalid_blocks`, `abort_step`) and `rank`, the rank the P/D proxy sends it to
(`data_parallel_rank`, which the load balancer obeys without reading the load counts); without
`rank` the balancer chooses.

The transport is the connector oracle's mock, run on each rank's own scheduler: a pull is reported
`recv_delay` steps after the rank's step that parked the request, a hold is reported `send_delay`
steps after the request finished. A request waiting for remote KV is unfinished, so its rank's wave
stays running; the rank steps, schedules nothing, and runs a dummy batch. A producer rank that only
holds finished requests' blocks has requests to step but no unfinished ones, so it does not keep
the wave running. A rank's record has the DP fields (`dummy`, `throttled`, and with
`async_scheduling` `processed` and `queue_len`) and what the connector oracle records for a
scheduler (`scheduled`, `preempted`, `aborted`, `transport`, `finished`, the KV state and the
connector-visible state). The trace is tier 3; the Go test is `TestDPNixlAgainstVLLM`
(`core/engine/dp_nixl_test.go`).

Not modelled: a rank with no connector (every rank here has one), `engine_id` and the side-channel
port (`VLLM_NIXL_SIDE_CHANNEL_PORT + data_parallel_index`), which name the peer and decide no
scheduling step, and the in-flight accounting of aborts, which the harness applies when it queues the
abort (the client drops a finished request when the engine reports it, `core_client.py:1703-1710`).

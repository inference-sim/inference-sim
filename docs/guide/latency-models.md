# Latency Models

BLIS takes its latency model from one source: **blis-latency-kernel**
(Go module `github.com/inference-sim/blis-latency-kernel`, pinned at `v0.2.0`). The adapter
in `sim/kernelmodel` translates a BLIS batch into the kernel's request shapes and converts
the kernel's answers to BLIS ticks; it computes no cost of its own. There is no other
backend and no `--latency-model` flag: the roofline and trained-physics backends, the
alpha/beta coefficients, `hardware_config.json` calibration, and the
`trained_physics_coefficients` block of `defaults.yaml` were removed.

## Who does what

inference-sim simulates the cluster: arrivals, admission, routing, scheduling, KV block
accounting, placement and metrics. blis-latency-kernel prices: step time, KV and fixed
memory, P/D KV transfer, offload tier transfer and host overheads. It reads model graphs,
chips, fabrics and storage devices from **blis-catalog** and fitted coefficients from
**blis-registry**; every file format involved, the scenario included, is defined by
**blis-schemas**. See [The BLIS repositories](../concepts/architecture.md#the-blis-repositories).

## Running

```bash
git clone --branch 0.2.2 --depth 1 https://github.com/inference-sim/blis-catalog.git
git clone --branch v0.1.2 --depth 1 https://github.com/inference-sim/blis-registry.git
export BLIS_CATALOG=$PWD/blis-catalog
./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios <dir of scenario files> --registry $PWD/blis-registry \
  --rate 10 --num-requests 100
```

The pinned releases are explained in
[Catalog compatibility](../getting-started/installation.md#catalog-compatibility).

## Inputs

Four roots, all required on `blis run` and `blis replay`:

| input | what it names |
|---|---|
| `--scenario` | a scenario **file name** inside `--scenarios` |
| `--scenarios` | a directory of scenario YAML files |
| `--registry` | a `blis-registry` clone root, holding the coefficient sets a scenario names |
| `--catalog` / `BLIS_CATALOG` | a `blis-catalog` clone root |

A scenario directory can be any directory of scenario YAMLs. The kernel module ships the
scenarios it was scored on, at
`$(go env GOMODCACHE)/github.com/inference-sim/blis-latency-kernel@v0.2.0/testdata/aisimulate`
(after `go mod download`). This repo's own fixtures are in `testdata/scenarios/`:
`glm-5-h200-3p1d-ib.yaml` (P/D) and `glm-5-h200-tp8-mtp3.yaml` (MTP speculation).

### Scenario structure

A scenario is a blis-schemas file of two YAML documents, a `Scenario` and a `Deployment`:

```yaml
kind: Scenario
name: llama-3.1-70b-instruct-h200-fp8-vllm-tp4
engine_version: "0.29.0"
model: llama-3.1-70b-instruct
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads,
               cost-model-attention, cost-model-recurrent, cost-model-memory]
cluster:
  hardware: h200
  nodes: 1
  gpus_per_node: 8
  # fabric: ib-400g      # the inter-node fabric, needed when a handoff crosses nodes
---
kind: Deployment
name: llama-3.1-70b-instruct-h200-fp8-vllm-tp4
pools:
  - role: colocated        # colocated | prefill | decode
    nodes: 1
    parallel: {tp: 4, pp: 1, dp: 1, enable_expert_parallel: false}
    engine:
      quantization: fp8
      cache_dtype: fp8
      block_size: 16
      max_num_batched_tokens: 8192
      max_num_seqs: 256
      max_model_len: 32768   # required
      cudagraph_mode: PIECEWISE
      gpu_memory_utilization: 0.9
      # enable_prefix_caching: false
      # speculative: {method: mtp, num_spec_tokens: 3}
# offload: ...               # optional
# pd_transfer: {connector: NixlConnector}   # optional, P/D scenarios
```

### What comes from the scenario, and what stays on the CLI

The scenario and the kernel supply, with no flag to restate them: the model, the hardware,
TP/DP/EP, the KV block budget (from the kernel's memory methods, per data-parallel rank),
block size, `max_num_seqs`, `max_num_batched_tokens`, `max_model_len` (a count is kept; unstated takes the model
graph's context window; `auto` or `-1` takes that window shrunk to what one rank's KV cache
holds, as vLLM v0.31.0 starts; a window one rank cannot hold, or none at all, is refused), prefix caching, the KV cache dtype, and the speculative draft length and
method. The flags that used to state these (`--model`, `--hardware`, `--tp`, `--dp`,
`--total-kv-blocks`, `--max-num-seqs`, `--max-model-len`, `--kv-cache-dtype`,
`--num-speculative-tokens`, ...) no longer exist; to change one, edit the scenario.

Everything a scenario does not state stays on the command line: the workload, rate and
concurrency, routing, admission, scheduling, flow control, saturation detectors, the
instance count and P/D topology, LoRA, KV offload tiers, and seed/horizon/output flags. See
the [Configuration Reference](../reference/configuration.md).

## What the kernel prices

- **Step time.** One forward pass over the scheduled batch, composed from per-operator
  costs (GEMMs, attention, grouped expert GEMMs, recurrent state updates, collectives,
  elementwise and host terms) with coefficients fitted against vendor kernel sweeps.
  BLIS reads the kernel's no-overlap edge of the band it reports. Expert geometry comes
  from the model graph in the catalog, not from an HF config.
- **Memory.** Weights, fixed overheads and the per-block KV cost, from which BLIS takes the
  KV block budget for each data-parallel rank.
- **Host overheads.** Per-request queueing (tokenization and preprocessing), per-output-token
  processing, and the fixed per-request completion overhead.
- **P/D KV transfer** and **offload tier transfer**, below.

A scenario that cannot be resolved -- absent file, unreadable coefficient set, unknown
engine version, a deployment that does not fit -- aborts the run naming what failed. There
is no fallback.

## Prefill/decode disaggregation

A scenario with a `prefill` pool and a `decode` pool, run with `--prefill-instances` and
`--decode-instances`. BLIS builds one kernel per pool, so each pool is priced at its own
parallelism and engine settings. The KV handoff from a prefill instance to a decode
instance is priced by the kernel's `PDTransferTime` over the scenario's fabric, between the
two instances' placements (NVLink when they share a node). Each instance count must not
exceed its pool's rank capacity: pool nodes × `gpus_per_node` / (pp × tp × pcp).

Refused: a P/D topology over a colocated scenario; a disaggregated scenario run without
`--prefill-instances`/`--decode-instances`; pools that differ in block size, dp or draft
configuration; a handoff connector other than NIXL's; shared (`--prefill-decode-instances`) or
encode instances.

The handoff follows vLLM v0.31.0's NIXL protocol, so the scenario's `pd_transfer.connector`
must be `NixlConnector`, `NixlPullConnector`, `NixlPushConnector` or a `MultiConnector` over
one; any other connector is refused. Three further rules follow from vLLM:

- **An offload tier needs a MultiConnector.** With a tier, `pd_transfer.connector` must be
  `MultiConnector`: setting `kv_offloading_size` replaces `kv_connector` with the
  OffloadingConnector (`vllm/config/vllm.py:1124-1158`), so a plain NIXL connector and a
  tier cannot run together.
- **A MultiConnector is assumed to list NixlConnector first.** The first connector in config
  order that reports tokens loads them (`multi_connector.py:134-141, 429-448`). blis-schemas
  states neither the children nor their order, so BLIS assumes llm-d's composition and says
  so on stderr.
- **A model whose NIXL prefill backs off is refused.** That covers Mamba-state models and
  multi-module MTP drafting K > 1: the decode side recomputes the trailing prompt tokens
  (`_prefill_backoff`, `nixl/base_scheduler.py:385-407`), which BLIS does not model. DSpark
  backs off nothing: its prefill lookahead is 1 (`config/vllm.py:657-678`).

- **Decode-side load (INV-PD-8).** The decode instance takes its own prefix-cache hit first
  (whole blocks, never the prompt's last token) and reserves the rest
  (`WAITING_FOR_REMOTE_KVS`). The handoff moves and is priced for only those blocks, so a
  session's later turn on the same decode instance pulls only its new blocks.
- **Prefill KV hold (INV-PD-7).** The prefill instance keeps a finished request's blocks,
  which count against its KV capacity, until the pull completes. They are freed at once if
  the decode side never took the request, and at NIXL's 30 s lease expiry if the decode
  instance went away mid-pull.
- **Contention (INV-P2-2, `--pd-transfer-contention`, default off).** Concurrent handoffs
  share the links they cross. The links come from the kernel's `PDTransferDemand`: the
  sender's NIC egress and the receiver's NIC ingress across nodes, or the NVLink domain on
  one node. Each link is divided max-min fairly and re-divided as handoffs start and finish.
  The floor is never shared. A handoff alone still takes exactly `PDTransferTime`. With
  contention off, every handoff is priced as if alone.
- **Offload tiers (#1940).** The scenario's `offload` block (or `--kv-offload-config`, see
  [KV offload](#kv-offload)) gives every pool its own tier of that size: llm-d's
  MultiConnector of NIXL and the OffloadingConnector. A pool's tier stores
  what its engine computes. The prefill pool's tier serves prefix hits before compute. The
  decode pool's tier serves the requests the decode pool runs itself (those the decider
  does not disaggregate), but never a remotely prefilled prompt, which NIXL, listed first,
  loads. A 0-byte tier is refused, as vLLM refuses it.

The PD section on stdout, and the results file's `pd` block, report the handoffs, their p50
and p90 durations, the queueing delay contention added (duration minus the uncontended
price), each link's peak concurrent handoffs, the prefill KV hold's peak, and each pool's
CPU-tier hit rate.


## Data and expert parallelism (MoE)

With `dp > 1` BLIS runs one engine per data-parallel rank, each sized as one EngineCore
(the kernel's per-rank KV budget and caps). `enable_expert_parallel` is priced by the
kernel. A dense model with `dp > 1` is refused.

**Lockstep (#1909).** The `dp` ranks of one logical instance are one vLLM DP group. vLLM
v0.31.0 runs an MoE model at `dp > 1` as `DPEngineCoreProc` ranks
(`vllm/v1/engine/core.py:1386-1395`), with or without expert parallelism: `use_all2all` is
true at `dp > 1` (`vllm/config/parallel.py:738-743`), so every MoE layer is a collective over
the ranks. BLIS therefore steps the group's ranks together:

- Every rank runs every step. A rank with nothing ready runs vLLM's dummy batch, one uniform
  decode of `1 + num_spec_tokens` tokens (`execute_dummy_batch`, `core.py:2280-2287`;
  `vllm/v1/worker/gpu_worker.py:1376-1382`).
- Each rank is priced by the kernel at its own batch padded to the group's largest token
  count, when that count is within the CUDA-graph capture ceiling the kernel resolved
  (`coordinate_batch_across_dp`, `vllm/v1/worker/dp_utils.py:99-181`: ranks pad only when
  the synced graph mode is not NONE). Past the ceiling, under `FULL_DECODE_ONLY` with a mixed
  batch, or when the layout runs no graphs, each rank costs its own batch. The ceiling and the
  graph modes come from the kernel; a layout that runs graphs with no readable ceiling is
  refused at startup rather than priced unpadded.
- The step lasts the slowest rank's price, plus that rank's own offload reload time, and every
  rank's clock advances by it. Requests that reach any rank at a step's own tick (arrival,
  queueing, adapter load) join that step, whichever rank they are on.
- The wave ends only when the unfinished-request all-reduce finds every rank idle. That
  all-reduce runs on the wave's first step and every 16th (`dp_sync_interval`,
  `vllm/config/parallel.py:150`; `core.py:2333-2350`), so up to 15 dummy steps follow the last
  request. Dummy steps are not work: a rank's clock, and so `SimEndedTime` and the rates
  divided by it, stop at its last real event.

**How requests reach ranks: `parallel.data_parallel_load_balancing` (or `--dp-load-balancing`, which may only restate a mode the scenario states).** vLLM v0.31.0's serve flags choose
(`vllm/engine/arg_utils.py:536-538, 1225-1242`; `EngineCoreClient.make_async_mp_client`,
`vllm/v1/engine/core_client.py:155-166`), and lockstep is the same in every mode:

| mode | vLLM | router sees |
|---|---|---|
| `internal` (default) | no LB flag: one front-end balances all ranks (`DPLBAsyncMPClient`) | one endpoint per replica (below) |
| `hybrid` | `--data-parallel-hybrid-lb`: one front-end per node balances that node's ranks | the same, for a replica on one node; a replica across nodes is refused |
| `external` | `--data-parallel-external-lb` or `--data-parallel-multi-port-external-lb`: every rank serves its own port (`DPAsyncMPClient`) | every rank, with its own signals and its own prefix cache; the router picks the rank |

**One endpoint per replica** (internal, hybrid). The router sees each DP group as one pod, listed under its rank
0 instance ID; a request routed to it is assigned to a rank by vLLM v0.31.0's internal DP load
balancer (`DPLBAsyncMPClient.get_core_engine_for_request`,
`vllm/v1/engine/core_client.py:1589-1640`): the rank with the lowest
`max(client_count x in_flight, waiting + running)`, plus `waiting x 6 x max(0, kv_usage - 0.5)`
when anything waits, over the per-rank counts the DP coordinator publishes every 100 ms
(`vllm/v1/engine/coordinator.py:154-162`). Between publishes the chosen rank's waiting count
is bumped and the scan start rotates, as the front-end does; `in_flight` is the front-end's
exact count of its unfinished requests per rank. The endpoint's signals follow how
llm-d's endpoint picker reads a pod's per-engine series (`engine="<rank>"` labels,
`vllm/v1/metrics/loggers.py:480-486`):

| signal | endpoint value |
|---|---|
| queue depth, running batch, in-flight requests, KV tokens and capacity | sum over ranks |
| KV utilization | max over ranks (llm-d-router#3085; v0.10.0 read rank 0's series alone) |
| prefix cache (precise scorer) | rank 0's cache only |

The prefix-cache view is rank 0's because vLLM publishes each rank's KV events on `base + rank`
(`vllm/distributed/kv_events.py:355-356`) and llm-d subscribes an endpoint at the pool port plus
its rank index (`pkg/kvevents/pool.go:134`), which is 0 for an internal-balancer pod.

Known deviations and limits:

- Pad tokens are the kernel's `Batch.PadTokens` (blis-schemas#74): they run the GEMMs,
  collectives, MoE dispatch and experts, and no attention, logits or sampling; an idle rank's
  dummy batch is all pad. This needs the kernel and schemas from the dsv41 branches, which
  `go.mod` pins by `replace` until they are released.
- vLLM also pads every rank under dual-batch overlap (`should_ubatch`, `dp_utils.py:181`).
  BLIS does not model DBO, which the kernel does not price (blis-latency-kernel#48).
- One front-end, with the whole in-flight view, stands for the `api_server_count` front-ends
  (dp by default), so `client_count` is 1.
- The coordinator's counts are sampled at the first assignment of each 100 ms period, not at
  the publish instant.
- A P/D pool's ranks step in lockstep but stay individually routable.
- The autoscaler is still refused with `dp > 1`, and a rank whose startup placement is
  deferred refuses the run.

## Speculative Decoding / MTP (#1528)

A pool that states `engine.speculative: {method, num_spec_tokens}` drafts `K` tokens per
step. BLIS does not predict acceptance, since it runs no draft model: it resolves it (below).

### Speculative acceptance (#1939)

The acceptance length AL (tokens per verification step, counting the bonus token, in
`[1, K+1]`; rate `α = (AL − 1)/K`) is resolved by blis-schemas' `acceptance.Resolve` from the
highest layer that states it, lowest to highest:

1. **Registry profile** -- `<registry>/acceptance/*.yaml`, keyed by (model, method, K,
   thinking, category). The workload's `category` and `thinking` select it, or
   `--speculative-category` and `--speculative-thinking`, which override the workload's.
2. **Workload** -- the simulated workload's own `acceptance` value (a catalog preset's, or the
   scenario's trace or replay binding's).
3. **Scenario** -- the scenario's `acceptance` pin: an inline value, or `profiles: <set>` to
   use that registry set in place of the default.
4. **CLI** -- `--speculative-acceptance-length AL` or `--speculative-acceptance-rate α`
   (mutually exclusive).

Changing the deployment's K selects that K's profile, so a K sweep with no override takes a
different AL at each K. Nothing falls back silently: an unmatched category, a missing thinking
mode, a value stated for another K or an AL outside `[1, K+1]` is refused, unless a
fallback category names one to use, which is then recorded: the scenario pin's
`fallback_category`, or `--speculative-fallback-category` when the pin states none (blis-schemas'
`acceptance.Resolve` lets the scenario's fallback win; the CLI outranks it only for the
acceptance value itself). The `--metrics-path` file records
`acceptance: {length, rate, source_layer, entry}`, the length and rate being those the run
advanced requests by and `entry` the profile set, key, provenance, any fallback, the hardware a
profile was measured on when it is not the run's, and the per-position rates (vLLM's
minimum-variance schedule when the source states none). A run that does not speculate records
none.

- **Method: the engine release's list.** `method` is checked against the rules pack of the
  scenario's `engine_version` (blis-schemas `rules.Pack.SpeculativeMethods`), not a list in
  BLIS, so vLLM 0.31.0 accepts `dspark` (DeepSeek-V4.1's DSpark), `dflash`, `eagle3`,
  `suffix` and the rest of that release's methods. A method the model graph's speculator
  does not describe, or one vLLM refuses for the model (MTP on DeepSeek-V4.1), is refused
  by the kernel. `testdata/scenarios/deepseek-v4.1-flash-h100-tp8-dspark5.yaml` is a DSpark
  example (#1932).
- **Cost: verify width `w = K+1`.** Every decoding request's forward pass is priced by the
  kernel at the full verify width, whatever is later accepted. The kernel's step time
  includes the drafter's passes (autoregressive `K x 1`, or DSpark's parallel block `1 x K`
  with its context stack and head), and the drafter's weights and caches are in its fixed
  memory, so BLIS adds no draft cost of its own (INV-K1). The decode threshold the kernel
  classifies by is widened to `1 + K`, as vLLM widens `reorder_batch_threshold`. Step time
  is independent of `α` and non-decreasing in `K`.
- **Progress: `g = 1 + α·K` tokens per step** (mean), applied deterministically through a
  per-request fractional carry (no RNG), so runs stay byte-identical for a seed (INV-6).
  Rejected drafts do not advance the request: the next step's context is the previous one
  plus the accepted tokens, never plus the verify width (INV-K2).

Progress stops at the completion boundary: the final step is granted only the tokens the
request still needs, as vLLM trims the accepted tail once `check_stop` fires. A request's
output-token count, and a closed-loop `accumulate` session's context growth, are therefore
identical to a `K=0` run. Under speculation the `itl_*` and `step_interval_*` percentiles are
per verification step; the `--metrics-path` file's `request_itl_*` (aiperf's per-token ITL) and
`interactivity_*` are per token, about the step interval over `1 + α·K` (see
[Metrics & Results](results.md#measured-window-aggregates-metrics-path-file)).

**Occupancy: the speculative footprint (#1627).** As vLLM v0.31.0 does, a speculating decode
schedules its whole verify width, `1 + K` tokens, and debits all of it from
`max_num_batched_tokens` whatever is later accepted, and every allocation holds KV slots for the
request's computed tokens, everything scheduled this step, and the drafter's lookahead past them
(`num_lookahead_tokens`: `K` for DSpark, EAGLE, MTP and draft models, `K + 1` for DFlash, 0 for
n-gram, suffix and Medusa). Progress still advances by the accepted count. Below saturation
nothing binds and results match the accepted-count accounting; under KV pressure fewer requests
fit and preemption starts earlier, as in the engine. vLLM's second budget is modelled too:
`input_budget` (`max_num_batched_tokens`) is debited every scheduled request's tokens plus
`max_num_new_slots_for_drafting` -- `K - 1` for DSpark, `K` for DFlash, 1 for a serial draft
model, 0 otherwise -- so a DSpark K=5 decode costs 10 of it, and a prefill chunk scheduled
beside decodes gets `min(token_budget, input_budget - draft_slots)`.

## Per-sequence bounded state (#1941)

The KV pool is sized from the kernel's per-token bytes; each resident sequence's bounded state
-- window rings, compressor and indexer state, bounded recurrent state -- is charged on top,
in pool blocks, the way vLLM v0.31.0 holds it:

- **On demand.** Every allocation resizes the sequence's state to the kernel's
  `SequenceBoundedBytesAt(computed, in flight)`: a window ring holds pages for the tokens
  scheduled this step plus the window behind them, and blocks the window has passed are freed
  first (vLLM's `allocate_slots` after `remove_skipped_blocks`). The hold grows while a prefill
  chunk is in flight and settles at the steady `SequenceFixedBytes` once decoding.
- **Gated at admission.** A new or resumed request is admitted only if its whole prompt and its
  state at the admission cap -- each window group at `min(cdiv(prompt), cap)`, the cap being
  `cdiv(window - 1 + max_in_flight_tokens, block) + 1` pages, which grows with
  `max_num_batched_tokens` -- are free (`full_sequence_must_fit`). The cap is never held.

For DeepSeek-V4.1-Flash with DSpark on h100 tp8 the steady hold is 49 blocks (about 5.7 MB) a
sequence per rank; the admission cap at 32,768 batched tokens is 16,425 blocks (about 1.9 GB).
vLLM's startup "GPU KV cache size" sizes a request at that cap plus its tokens, and BLIS
reproduces that figure (pinned against the engine's log on H200), but it is a report, not what
a running sequence holds. The startup log reports the steady hold, the cap and the steady
hold's share of the pool at `max_num_seqs`. A model with no such caches is charged nothing
and gated as before (BLIS does not apply vLLM's full-prompt gate to it).

## KV offload

The scenario's `offload` block is the run's KV offload configuration (#1910). It maps onto
vLLM's OffloadingConnector:

- **Tiers.** A `cpu_dram` tier of `bytes` is the CPU tier (`cpu_bytes_to_use`). vLLM's
  defaults apply to everything the schema does not state: block size, `offload_prompt_only`,
  `tokens_per_hash`.
- **Accepted settings.** `connector` may be `OffloadingConnector` or left out; `spec` may be
  `CPUOffloadingSpec`, `TieringOffloadingSpec`, or left out; `eviction_policy` may be `lru`.
- **Refused by name:**
  - another connector or spec;
  - `prefetch_depth`;
  - `arc`;
  - a secondary tier (blis-schemas does not state the fs tier's `root_dir`, `direct_io` and
    thread counts; describe those with `--kv-offload-config` instead);
  - an offload block with no `cpu_dram` tier.
- **0 bytes.** blis-schemas refuses a 0-byte tier, as vLLM refuses `cpu_bytes_to_use: 0`.
- **Per engine.** The block is deployment-level and the schema has no per-pool form, while
  `cpu_bytes_to_use` belongs to one engine's connector. So every engine gets its own tier of
  the stated bytes: the colocated pool, and each prefill and each decode engine.
- **The flag only restates.** `--kv-offload-config` may restate the scenario's block; it must
  then resolve identically, or the run is refused naming the fields that differ. With no
  scenario block, the flag alone configures the offload.

`--kv-offload-config` tiers must each name a catalog `device_class`; the kernel prices every
transfer with `TierTime` for that device at the tier's in-service queue depth. Explicit
`read_bandwidth`, `write_bandwidth` and `base_latency` in the tier config are refused.
The legacy `--kv-cpu-blocks` tier is priced by the kernel from the catalog `cpu_dram`
device. See [KV Offload](kv-offload-calibration.md).

## LoRA

The per-step adapter compute overhead is applied to the kernel's step time, and the static
adapter HBM reservation is set aside from the kernel's KV budget before blocks are counted.
The LoRA cost coefficients come from `--defaults-filepath` (the `lora:` block of
`defaults.yaml`) or `--lora-config`.

## What it does not yet do

- Predict speculative acceptance (it is an input).
- Price shared prefill-decode or encode instances.
- Model vLLM's async scheduling: an engine runs one forward pass at a time and schedules the
  next only when the current one ends (INV-8), with no overlap of scheduling and execution.
- Run a model or chip absent from the catalog, or a coefficient set absent from the
  registry. Adding either is a change to blis-catalog or blis-registry, not to this repo.

## The published accuracy figures

`cmd/metricscore` is the scorer behind the evaluation tables. It simulates nothing itself. For
every measured point it writes two files and runs `blis replay` on them, exactly as a user replays
a trace:

- a scenario: the sweep's own, with every run setting the measured engine passed;
- a trace of the InferenceX benchmark client's main phase: the `10*CONC` target lengths numpy drew
  at the client's seed 0, in request order, which are also the lengths the client sent (verified
  per point; see below).

The trace runs in pool mode at concurrency `CONC`. On a point whose engine caches prefixes, the
main phase runs after copies of request 0 have completed. That leaves the one state the
benchmark's drained warm-up leaves which the simulator reads: request 0's prompt cached, on an
idle engine.

TTFT and TPOT are the client's own `mean_ttft` and `mean_tpot` over the main-phase requests. Its
figures are therefore the CLI's. What is and is not reproduced, with the tests that check each
claim, is in [What the accuracy comparison reproduces](../kernel-exclusive/APPLES-TO-APPLES.md).

At the tagged releases above, with the vendored catalog and registry and the measurement corpora in
`BLIS_MEASUREMENTS` (see [Reproducing](../kernel-exclusive/REPRODUCE.md#the-measurement-data)):

```bash
go build -o blis main.go
go run ./cmd/metricscore -blis ./blis -framework vllm
```

Its flags are `-blis`, `-gaps`, `-corpus`, `-scenarios`, `-catalog`, `-registry`,
`-engine-settings`, `-absolutes`, `-framework`, `-config-tier`, `-simulated-only` and `-seed`.
`-seed` seeds only the synthetic prompts' token IDs, which move no latency.

`-gaps <file>` (default: stderr) lists every point that cannot be scored as the measured run was,
with the repository whose change would close the gap. Possible gaps:

- a model, chip or fabric missing from blis-catalog;
- coefficients missing from blis-registry;
- a run setting blis-schemas cannot express;
- a setting the measured run did not record;
- sent prompt lengths not verified equal to the targets;
- an engine BLIS does not model (sglang, TensorRT-LLM).

The report ends with a count of points per cause. The scorer reads those gaps to put every sweep
into one of three tiers, and prints every arm on each:

- **STRICT**: every point has an engine log, and every setting it passed is carried. The admission
  settings and `max_model_len` were measured. The sent prompt lengths are verified. No other gap.
- **MEASURED-OR-RESOLVED**: as STRICT, but an unpassed admission setting may take the value vLLM
  resolves for it.
- **ALL (approximate)**: every scoreable sweep. These are not the measured runs as they ran.

Mean absolute error, BLIS / AISimulate / AIC, on the same points:

| tier | sweeps | n (shape / mape) | TPOT shape | TPOT mape | TTFT shape | TTFT mape |
|---|---|---|---|---|---|---|
| STRICT | 0 | 0 | -- | -- | -- | -- |
| MEASURED-OR-RESOLVED | 10 | 52 / 62 | 11.38 / 8.73 / 9.15 | 16.79 / 19.85 / 19.08 | 13.96 / 23.22 / 22.34 | 23.47 / 42.96 / 23.77 |
| ALL (approximate) | 113 | 460 / 573 | 9.64 / 10.43 / 10.56 | 15.04 / 15.39 / 15.71 | 17.85 / 23.05 / 30.12 | 25.30 / 37.86 / 36.66 |

No sweep is STRICT. Every logged vLLM sweep either passed `max_cudagraph_capture_size`, which
blis-schemas cannot express, or left an admission setting unpassed.

On the MEASURED-OR-RESOLVED tier BLIS is **behind on TPOT shape** (11.38% against 8.73% and 9.15%) and
ahead on the other three metrics. The TPOT-shape deficit comes from two sweeps of one deployment,
minimax-m2.5-h200-fp8-vllm-tp4:

- **8k1k:** BLIS over-predicts the rise of TPOT with concurrency by +14.7% to +22.6% from
  concurrency 8 to 256, scoring 16.42% against AISimulate's 5.10%.
- **1k8k:** BLIS over-predicts it by +12.8% to +38.4% from concurrency 8 to 64, scoring 27.83%
  against AISimulate's 15.86%.

Without those two sweeps the two models are level, 8.73% against 8.75%. On the same tier BLIS's
TTFT is biased low: its signed mean TTFT mape is **-21.53%**.

A figure quoted without its tier and `-framework` cannot be checked. `-config-tier` (sweeps with
or without an engine log) changes the ALL row as well.

## The interface

The simulator consumes latency through the `sim.LatencyModel` interface
(`sim/latency_model.go`): `StepTime(batch)`, `QueueingTime(req)`,
`OutputTokenProcessingTime()` and `PostDecodeFixedOverhead()`, all in microseconds (ticks).
`sim/kernelmodel` is its only production implementation. Improving or extending latency
pricing is done in blis-latency-kernel (the cost model) and blis-registry (its
coefficients), not here; see [Extension Recipes](../contributing/extension-recipes.md).

## Further Reading

- [Configuration Reference](../reference/configuration.md) -- the scenario flags and what the scenario supplies
- [Supported Models](../reference/models.md) -- which models and chips a run can use
- [Core Engine](../concepts/core-engine.md) -- where step time enters the event loop

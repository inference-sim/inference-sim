# Trace Replay

`blis replay` runs a TraceV2 trace through the discrete-event simulator. A trace is a header
YAML file and a data CSV file. Two commands write one: `blis run --trace-output` (header
`mode: generated`) and `blis convert weka|otel` (header `mode: converted`). `blis replay
--trace-output` writes `mode: replayed`. Replay runs on blis-latency-kernel through the same
deployment path as `blis run`.

```bash
./blis replay --trace-header trace.yaml --trace-data trace.csv \
  --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios <dir of scenario files> --registry $PWD/blis-registry
```

!!! note "What a header mode means"
    A `generated` or `replayed` trace records the deployment BLIS ran. Replay keeps its header
    authoritative (INV-13): a scenario offload block or `--kv-offload-config` may only restate
    the recorded `kv_offload`, never add one. A `converted` trace records external traffic and
    no deployment, so replay takes the offload tiers from the scenario's offload block or
    `--kv-offload-config` ([#2067](https://github.com/inference-sim/inference-sim/issues/2067)).
    The header modes are `generated`, `replayed` and `converted`; any other value is refused.
    `mode: real` was written by `blis observe`, which v0.16.0 removed
    ([#1901](https://github.com/inference-sim/inference-sim/issues/1901)), and is refused.

## `blis replay`

Replays a TraceV2 file through the BLIS discrete-event simulator. Instead of generating synthetic requests, replay loads request timing and token counts from the trace. It takes the same deployment inputs as `blis run` — `--scenario`, `--scenarios`, `--registry` and the catalog (`--catalog` or `BLIS_CATALOG`) — and builds the deployment and its kernels through the same code path.

### Replay-Specific Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--trace-header` | `string` | `""` | Path to TraceV2 header YAML (required) |
| `--trace-data` | `string` | `""` | Path to TraceV2 data CSV (required) |
| `--results-path` | `string` | `""` | Deprecated (use `--output`): file to write a per-request JSON array (request_id, ttft_us, e2e_us, input_tokens, output_tokens, slo_class, model, itl_mean_us) |
| `--scenario` | `string` | `""` | Scenario file name within `--scenarios` (required) |
| `--scenarios` | `string` | `""` | Directory of scenario YAMLs (required) |
| `--registry` | `string` | `""` | blis-registry clone root (required) |
| `--trace-output` | `string` | `""` | Export replay results as TraceV2 files (`<prefix>.yaml` + `<prefix>.csv`); header `mode: "replayed"` |
| `--timeout` | `int` | `300` | Per-request deadline in seconds, overriding every round's trace deadline ([#2064](https://github.com/inference-sim/inference-sim/issues/2064)). Negative disables the deadline; 0 is rejected. **Applied only when explicitly set** — unset, each round keeps its own recorded `deadline_us` (round 0 and every follow-up alike), or no deadline when `deadline_us` is 0. Symmetric with `blis run --timeout`. |

!!! note "Follow-up rounds honor the trace deadline (#2064)"
    In a closed-loop replay, each session follow-up round now takes its deadline from the
    trace — the recorded `deadline_us` relative to that round's arrival, or no deadline when
    `deadline_us` is 0 — the same rule round 0 already followed. Before #2064 follow-ups
    silently received BLIS's 300 s default, so heavy agentic sessions (a long reasoning step
    or a large prefill under load) timed out at 300 s, cancelling the session and dropping
    every later round with nothing in the log. A timeout is now always warned about (with the
    timed-out-request count and a cancelled-session count), and `--timeout` sets or disables
    one deadline for all rounds.

Replay also accepts the shared simulation flags `blis run` keeps — the llm-d router's environment flags (`--gateway-latency`, `--epp-scrape-jitter`, `--kv-event-delay`; its deployment inputs are the scenario's `router:` block and pool labels), scheduling, saturation detection, `--num-instances`, P/D topology, LoRA, the speculative acceptance flags (`--speculative-acceptance-rate`/`-length` and the selectors; a run's `--workload` preset terms are not in the trace, so a replay passes them as selectors), KV offload, seed and horizon. Engine settings (block size, batch limits, max model length, prefix caching, cache dtype, parallelism, speculative draft length) come from the scenario. See [Configuration](../reference/configuration.md) for the full list.

A trace exported by `blis run` records the router inputs in its header (with any
configuration inlined). Replay rebuilds that router. A replay scenario that states a router
(`router:` block or pool labels) must state the recorded one, and router environment flags must
match the recorded ones; a difference is refused. A scenario that states no router replays the
recorded one. A trace without a recorded router (one from `blis convert`)
routes with the router the scenario states, or the router's default configuration.

!!! note "Replay with the same scenario and flags as the run"
    The deployment is not round-tripped through the trace header: replay rebuilds it from
    `--scenario` (and the kernel's KV budget) exactly as `blis run` does. A trace exported by
    `blis run` and replayed with the **same scenario and identical flags, including
    `--horizon`**, produces equal `metrics`, `requests` and `breakdowns` (INV-13). Pass `--horizon` on both legs:
    `blis run` defaults it to unlimited while `blis replay` auto-computes 2x the max arrival
    time (see the table below), so a default-horizon replay may truncate a run that was still
    draining. An MoE scenario with `dp > 1` replays the same way — one replica per
    data-parallel rank on both legs. The autoscaler and node pools are rejected by
    `blis replay`.

    ```bash
    # 9223372036854775807 is 2^63-1 — `blis run`'s own --horizon default, i.e. unlimited.
    ./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
      --scenarios <dir of scenario files> --registry $PWD/blis-registry \
      --rate 10 --num-requests 40 --seed 42 \
      --horizon 9223372036854775807 --trace-output traces/run1

    ./blis replay --trace-header traces/run1.yaml --trace-data traces/run1.csv \
      --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
      --scenarios <dir of scenario files> --registry $PWD/blis-registry \
      --seed 42 --horizon 9223372036854775807
    ```

### How Replay Differs from `blis run`

| Aspect | `blis run` | `blis replay` |
|--------|-----------|---------------|
| **Request source** | Generated from workload spec or CLI distributions | Loaded from TraceV2 CSV |
| **Arrival times** | Synthesized by arrival process (Poisson, etc.) | Exact timestamps from trace |
| **Token counts** | Sampled from distributions | Recorded values |
| **Horizon** | From `--horizon` flag or spec | Auto-computed as 2x max arrival time (override with `--horizon`) |
| **Output format** | One `Result` document on stdout ([Metrics & Results](results.md)) | The same `Result` document; `--results-path` (deprecated) additionally writes a per-request JSON array (request_id, ttft_us, e2e_us, input_tokens, output_tokens) |
| **Session support** | Session manager creates follow-ups | Session structure encoded in trace (no manager needed) |
| **Trace export** | `--trace-output` (header `mode: "generated"`) | `--trace-output` (header `mode: "replayed"`); see restrictions below |

!!! note "`--trace-output` faithfully re-exports closed-loop follow-up rounds (#1630)"
    A closed-loop replay's `--trace-output` captures **every** round — the round-0
    requests **and** the follow-up rounds the session manager / pool driver generate
    during the run (previously only the round-0 wave was exported, #1621):

    - **Accumulate corpus** (`session_context_growth: accumulate` — e.g. a
      `blis convert otel` / `blis convert weka` corpus): the re-export re-derives per-round
      `input_tokens` **deltas** and `input_tokens_reset` compaction markers from the
      replayed absolute inputs, sets `session_context_growth: accumulate` on the header,
      and records the per-round think time. Re-replaying it (closed-loop) reproduces the
      original run's per-request per-round input and metrics (INV-13 round-trip).
    - **Non-accumulate multi-round corpus**: the re-export carries absolute per-round
      `input_tokens` (follow-ups inherit their session's `prefix_group`/`prefix_length`,
      so the prefix is not double-counted) plus the captured `think_time_us`.
    - **Pool mode** (`--concurrent-sessions N`): the re-export is a **complete** session
      corpus — all sessions (originals + clones) with every round, as a closed-loop
      corpus. Replay it with `--session-mode closed-loop` (it is not re-cloned). Aggregate
      **conservation** metrics (completed/injected requests, total input/output tokens)
      reproduce; per-request and cache/latency aggregates are not guaranteed — pool
      admission timing is data-dependent, and a non-accumulate clone's cache-busting
      divergence is not preserved when it shares a `prefix_group`.

    Fixed-mode (`--session-mode fixed`) re-export is unchanged. Known boundary: a
    length-capped round (output truncated by the scenario's `max_model_len`) reproduces its
    input length but may diverge in prefix-cache token *content* across the cap boundary —
    replay huge-ISL agentic corpora with a scenario whose `max_model_len` is large enough to
    avoid capping.

!!! warning "The scenario must match the server"
    Replay prices every step with blis-latency-kernel for the deployment the scenario states. To compare against a real server, the scenario's model, hardware, parallelism and engine settings must match that server's. See [Latency Model](latency-models.md).

---

## Replaying OTel agentic traces

BLIS can replay captured OpenTelemetry agent traces (e.g. the Exgentic
`agent-llm-traces` dataset) as a fixed pool of N concurrent closed-loop sessions.
Each session is a linear chain of LLM calls: BLIS waits for call N to complete,
applies the recorded think/tool gap, then sends call N+1 — whose prompt reuses
call N's entire prompt plus its output plus new tokens (a strictly-growing shared
prefix for realistic KV-cache reuse).

The dataset ships as Parquet, so first explode it to per-session OTel JSON:

```python
import pyarrow.parquet as pq, json, os
os.makedirs("otel_json", exist_ok=True)
t = pq.read_table("data/train-00001-of-00039.parquet").to_pylist()
for i, row in enumerate(t):
    json.dump({"spans": row["spans"]}, open(f"otel_json/trace_{i:04d}.json", "w"))
```

Then convert and replay:

```bash
# Convert the trace corpus to a TraceV2 pair (accumulate = growing shared prefix).
blis convert otel --input otel_json --trace-output corpus \
  --context-growth accumulate --max-think-time 15s

# Replay a fixed pool of 8 concurrent sessions, 200 total (corpus duplicated to fill).
blis replay --trace-header corpus.yaml --trace-data corpus.csv \
  --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios <dir of scenario files> --registry $PWD/blis-registry \
  --concurrent-sessions 8 --total-sessions 200
```

The recorded source model names are pure provenance and are dropped during
conversion — all calls are simulated under the scenario's model. (They are deliberately not
written into the trace: `TraceRecord.Model` becomes the request's model at replay, and
a name that differs from the simulated model would report every request under a model
the run never simulated.)
`--concurrent-sessions` implies closed-loop session semantics; without it,
replay behaves as a standard fixed/closed-loop replay.

By default the pool is **self-draining**: it runs until all `--total-sessions`
sessions complete, regardless of how many waves that takes. You may pass `--horizon`
to impose a hard cap on the simulated run — sessions still queued when the cap is
reached are reported as un-admitted (a warning is logged) rather than silently
dropped.

!!! tip "Subsetting a large corpus: add `--shuffle-corpus`"
    `--total-sessions N` on its own is **deterministic, ordered selection**, not a
    random sample: with `N` below the corpus size it replays the **first `N`
    sessions in file order** and never touches the tail — a biased sample for a
    time-sorted or harness-grouped corpus. To draw a *representative* subset, add
    `--shuffle-corpus`, which applies a seeded Fisher-Yates permutation (reproducible
    from `--seed`, on a stream independent of token generation) before selection:
    `--total-sessions N --shuffle-corpus` yields a seeded-random `N`-of-corpus
    subset, while `--total-sessions ≥ corpus` simply randomizes the admission order
    with every session still running.

!!! warning "Size the context window to the trace"
    Real agentic traces (e.g. Exgentic `agent-llm-traces`) often carry very large
    prompts — tens of thousands to well over 100K input tokens per call, growing
    further each round as the shared prefix accumulates — which can exceed a
    scenario's `max_model_len`. If the run completes cleanly but reports
    `completed_requests: 0` with a matching `dropped_unservable` count, inspect the
    trace for empty prompts as well as requests that exceed the context window or KV
    budget. For oversized prompts, use a scenario whose `max_model_len` covers the
    largest round and whose computed KV budget can hold the growing sessions.

### Weka CC traces (`blis convert weka`)

The [SemiAnalysis WekaTrace](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-with-subagents-051926)
datasets are a second agentic-trace family. Unlike the OTel/Exgentic corpus they ship
as **JSONL** — one proxy session per line — so no Parquet explosion step is needed:

```bash
# Convert (one session per JSONL line) to the same TraceV2 corpus format.
# --drop-subagents: this dataset's sessions carry sub-agent groups (see below).
blis convert weka --input traces.jsonl --trace-output corpus \
  --context-growth accumulate --max-think-time 0 --drop-subagents

# Replay closed-loop (or as a concurrent pool, exactly as for OTel above).
blis replay --trace-header corpus.yaml --trace-data corpus.csv \
  --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios <dir of scenario files> --registry $PWD/blis-registry \
  --session-mode closed-loop
```

Weka sessions grow to very long contexts; use a scenario whose pools' `max_model_len` covers
them (a round longer than it is length-capped).

The reader converts each session's **linear main-agent stream**. Sub-agent fan-out —
`type:"subagent"` groups, whose nested requests run concurrently with the main agent — is
**not modelled yet** ([#1477](https://github.com/inference-sim/inference-sim/issues/1477)), so a
session containing one is **refused** by default, naming the session and the work it carries.
`--drop-subagents` converts such sessions anyway: the groups are dropped and their wall-clock is
absorbed into the following main turn's think gap, which keeps the main agent's pacing but
offers the engine none of the children's load, overlap or KV state — the replay
**under-loads** the deployment, and the converter warns with the number of dropped groups and
requests. Per-round **pure client think** is
recomputed as `max(0, t_i − t_{i-1} − api_time_{i-1})` between consecutive main turns
(carried in the `think_time_us` column, `--max-think-time` default `0` = uncapped,
since Weka gaps are genuine away-from-keyboard times). The column is **non-lossy**
(#1608): a genuinely-zero recomputed think (an overlapping turn) is recorded as `&0`,
distinct from a not-recorded (empty) cell, so an all-overlap session replays with the
recorded zeros rather than degrading to arrival-gap think. The recorded `claude-*`
model names are dropped during conversion (same reason as OTel: no phantom per-model metrics).

!!! note "Context compaction is represented (#1609)"
    Weka input token counts are very large (p50 ≈ 110K, p90 ≈ 395K), so the
    `max_model_len` / KV-budget sizing warning above applies with extra
    force. Real Claude Code traffic **compacts/trims context constantly** — ~30% of
    rounds on the full `051926` dataset have `in_N < in_{N-1}+out_{N-1}` (the model
    summarized or trimmed). The shared agentic-trace encoder emits a per-round
    `input_tokens_reset` marker (the recorded absolute) on exactly those non-monotone
    rounds, and accumulate closed-loop replay **re-seeds its growing buffer to that
    absolute at the compaction boundary**. The reconstructed cumulative input therefore
    tracks the recorded total — previously the buffer could only grow, clamped the delta
    to 0, and **over-counted by ≈3–4×** (+312% on that dataset). The re-seed intentionally
    breaks strict prefix identity across the boundary (a summary is not a literal prefix of
    the pre-compaction context), which also corrects the prefix-cache hit-rate over-estimate.
    The marker is a trailing conditional CSV column: absent for monotone sessions and for
    `blis run`, so a trace with no compaction round replays byte-identically to before
    (INV-6). Traces converted by an **older build** (no marker column) still over-count —
    re-run `convert` to get compaction-aware output. (The separate think-time lossy-0
    sentinel was resolved in #1608 — see the non-lossy note above.)

### Weka subagent replay as aiperf runs it: `--agentic-scenario inferencex-agentx-mvp` (#1937)

`blis convert weka` keeps only the main agent. To replay a Weka v7 corpus **with** its
subagents, exactly as aiperf 0.12.0's `aiperf profile --scenario inferencex-agentx-mvp`
does, give replay the raw corpus instead of a TraceV2 file:

```bash
blis replay --agentic-scenario inferencex-agentx-mvp \
  --corpus traces.jsonl --corpus-sha256 <hex> \
  --concurrency 64 --seed 42 --benchmark-duration 900s --grace-period 30s \
  --max-context-length max_model_len \
  --scenario <scenario.yaml> --scenarios <dir> --registry <blis-registry> \
  --metrics-path out.json
```

`examples/agentx-dsv41-flash-h100-2x-tp8-dspark5.yaml` is a two-replica DeepSeek-V4.1-Flash
H100 deployment to run it on. A sweep over concurrencies can add `--corpus-cache <dir>`:
the first run writes a binary snapshot of the parsed corpus (keyed by the corpus digest,
the max-context filter, chain splitting and the build), later runs load it in well under a
second instead of re-parsing the JSON, and a missing, damaged or mismatched snapshot is
ignored, logged and rewritten -- output is byte-identical either way.

Not supported: the accelerated cache-pressure warm-up (`warmup: cache_pressure`, refused),
`--synthesis-max-osl` (no such cap is applied; the descriptor has no field for it),
`--trace-output` (refused), a trace-idle or per-turn delay cap (forbidden by the scenario),
and model mapping (every request goes to the scenario's model). A warm-up request that ends
in an error, which makes aiperf abort, is counted and warned about.

The workload can also be given as `--workload-spec <file>` with a blis-schemas replay descriptor (`replay:`), which the
flags override; the descriptor is validated as aiperf's scenario validator validates the
scenario (a window under 900 s, another cache bust or idle cap, a missing corpus digest are
refused). The corpus is streamed once and refused on a digest mismatch.

What is replayed, each step a port of aiperf's code and checked send for send against
aiperf's own runtime (`TestAgentXRuntimeMatchesAiperf`):

- **Session trees.** Each trace is rebuilt as aiperf rebuilds it: flattened-agent chain
  detection on the top-level requests and inside every subagent, SPAWN branches and
  SPAWN_JOIN gates, end-to-start delays `max(0, t_k - t_{k-1} - api_{k-1})`, and the
  recorded cross-stream replay barriers. Traces whose peak `in + max(out, 1)` (parent and
  subagents) exceeds `--max-context-length` are dropped; `max_model_len` is the
  deployment's resolved value.
- **Lanes.** `--concurrency` counts session trees, not requests. Lane *i* replays root
  `pool[i mod N]` from a snapshot instant t* drawn as aiperf draws it (numpy
  `default_rng(sha256("{seed}:{trace}:{lane}")[:8]).uniform` over the trace's span). A
  drained tree recycles its lane into the next root at turn 0.
- **Warm-up, window, grace.** Every stream live at t* first sends its last pre-t* request
  with max_tokens 1, aligned so every lane reaches t* together; profiling starts when the
  warm-up drains and sends for `--benchmark-duration`; requests in flight then have
  `--grace-period` to finish, after which they are cancelled and leave no aiperf record.
  A 10 s global idle cap pulls in pending timers whenever nothing is in flight.
- **Prompts.** A turn's prompt is its full message list (aiperf re-sends the whole history
  every turn), split into user and assistant messages as aiperf's reconstructor splits it,
  rendered through DeepSeek-V3-family chat-template tokens (BOS, a header per message, EOS
  after assistant messages, the generation prompt) with the tree's cache-bust marker (10
  tokens) at the head of the first user message. Message content is exactly the recorded
  `in` tokens: hash blocks (equal across the tree's conversations) and per-turn synthetic
  tails. So prefixes are shared within a tree up to the first message boundary two
  conversations place differently, never across trees. The template adds 0.09% to the
  content (median 27 messages per prompt). Outputs are `max(out, 1)` tokens (ignore_eos);
  generated tokens are never fed back.
- **Outcomes.** A cluster refusal (admission, routing, flow-control rejection or shed, queue
  TTL, eviction, P/D decode-KV drop) or a client timeout is an error and the stream goes on;
  a prompt past `max_model_len` is a context overflow and ends the stream; a request still in
  flight at the grace end is cancelled. P/D, flow control and REDIRECT drains are supported.
- **`--horizon`.** The replay sets the run's end itself (the grace end, or the profiling
  phase's completion when that comes first); a smaller `--horizon` cuts the run there, and
  the measured window then ends at it.

Every request carries `Phase` (`warmup` / `profile`, `grace` once it completes after the
window) and an `agentic` tag (lane, tree, trace, stream, kind, turn, outcome:
`completed`, `error`, `context_overflow`, `cancelled`), echoed per request in the results
file, which also gets the measured window (the profiling phase's start and completion) and an
`agentic` summary block. The laws are INV-AX1 … INV-AX4 in the invariant registry.

### Faithful high-concurrency replay: `--session-mode fixed-accumulate` (#1692)

An accumulate corpus can be replayed three ways. Only the third is faithful at
concurrency > 1:

| Mode | Arrivals | Accumulate inputs | Faithful high-conc? |
|------|----------|-------------------|---------------------|
| `--session-mode closed-loop` | regenerated (`completion + think`) | ✅ reconstructed | ❌ self-throttles — arrivals chain to sim completion, so the queue drains instead of piling up and TTFT collapses to compute-only (30–100× low at conc ≥ 8) |
| `--session-mode fixed` | recorded | ❌ **hard-rejected** on an accumulate corpus (reads deltas as absolutes) | ❌ not usable |
| `--session-mode fixed-accumulate` | **recorded** | ✅ reconstructed | ✅ recorded arrivals carry the real cross-session overlap, so N large prefills pile into the scheduler at the real clock and produce genuine queue wait |

```bash
# Convert once, then replay with recorded arrivals AND reconstructed growing context.
# The motivating shape is a large MLA MoE on H200; any scenario works, as long as its
# max_model_len covers the corpus's longest context.
blis convert weka --input traces.jsonl --trace-output corpus --context-growth accumulate --drop-subagents
blis replay --trace-header corpus.yaml --trace-data corpus.csv \
  --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios <dir of scenario files> --registry $PWD/blis-registry \
  --session-mode fixed-accumulate
```

The arrival timestamps are already in the corpus (`ArrivalTimeUs`, written per round by
the converter) — this mode consumes data that exists today; no trace-format change. It
requires an accumulate corpus, and is mutually exclusive with `--concurrent-sessions`
(open-loop, so there is no session pool to maintain) and `--think-time-*` (arrivals are
recorded, not regenerated). INV-10 (session causality) is **scoped to closed-loop** and
does not apply — chaining arrivals to sim completion is exactly the feedback loop this
mode breaks.

!!! warning "Intra-session overlap: recorded arrivals ignore sim completion"
    Each round is injected at its recorded arrival regardless of when the previous round
    of the **same** session finishes in the sim. A real agentic client is serial (round N
    waits for round N−1's response), but the recorded inter-round gap includes the *real*
    server time; the moment sim service time exceeds that real time — exactly the
    queueing regime this mode creates — round N is injected while round N−1 is still
    decoding. Two consequences to keep in mind when reading results:

    1. **Measured concurrency is inflated** above what a real serial client produces,
       because same-session rounds can be in flight at once.
    2. **Prefix-cache hit rate is partly fictional**: round N's prompt contains round
       N−1's output tokens, which (under overlap) round N−1 has not finished emitting —
       so part of the modeled prefix hit corresponds to tokens that did not yet exist.

    This is a distinct effect from the (faithful) *cross*-session overlap the mode
    reproduces, and it is not covered by the INV-10 exemption (which is about the arrival
    *source*). It is inherent to open-loop replay of a serial workload; closed-loop avoids
    it but at the cost of the self-throttling feedback loop this mode exists to break.

!!! note "`--trace-output` produces an absolute (non-accumulate) corpus"
    fixed-accumulate reconstructs each round's growing context in memory, so
    `--trace-output` re-exports the **already-reconstructed absolute** per-round inputs —
    the exported header carries **no** `session_context_growth`, and the CSV records
    absolute `input_tokens` (not deltas). This export is a faithful absolute-mode corpus:
    re-replay it with `--session-mode fixed` (the default). It is **not** an accumulate
    corpus, so `--session-mode fixed-accumulate` will reject it (the accumulate-corpus
    guard) — re-run the original `convert` step if you need to re-export deltas. This is
    lossless: the absolute inputs are exactly what fixed-accumulate reconstructed.

---

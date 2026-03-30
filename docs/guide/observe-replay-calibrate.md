# Observe / Replay / Calibrate

This guide covers the end-to-end pipeline for validating BLIS simulator accuracy against real inference servers: observe real latencies, replay the captured trace through the DES, and calibrate by comparing results.

```bash
# Quick example: observe a real server, replay through the simulator, compare
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --workload-spec workload.yaml --trace-header trace.yaml --trace-data trace.csv
./blis replay --trace-header trace.yaml --trace-data trace.csv \
  --model qwen/qwen3-14b --results-path results.json
./blis calibrate --trace-header trace.yaml --trace-data trace.csv \
  --sim-results results.json --report calibration.json
```

## Pipeline Overview

The observe/replay/calibrate pipeline has three stages:

| Stage | Command | Input | Output |
|-------|---------|-------|--------|
| **Observe** | `blis observe` | Workload spec or distribution params + real server | TraceV2 (header YAML + data CSV) |
| **Replay** | `blis replay` | TraceV2 files + simulator config | SimResult JSON |
| **Calibrate** | `blis calibrate` | TraceV2 + SimResult JSON | Calibration report JSON |

**Data flow:**

```
WorkloadSpec YAML ──► blis observe ──► TraceV2 (header.yaml + data.csv)
                        │                       │
                        ▼                       ▼
                   Real Server            blis replay ──► results.json
                                                │
                                                ▼
                              TraceV2 + results.json
                                                │
                                                ▼
                                        blis calibrate ──► calibration.json
```

**Why three separate commands?** Each stage is independently useful. You can observe without replaying (to collect latency baselines), replay without calibrating (to test simulator behavior on real traces), or re-calibrate with different simulator configs without re-observing.

---

## `blis observe`

Dispatches requests to a real inference server, records per-request timing (TTFT, E2E latency, token counts), and exports the results as a TraceV2 file pair.

### Required Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--server-url` | `string` | `""` | Inference server URL |
| `--model` | `string` | `""` | Model name for API requests |
| `--trace-header` | `string` | `""` | Output path for TraceV2 header YAML |
| `--trace-data` | `string` | `""` | Output path for TraceV2 data CSV |

### Workload Input (one required)

Four input modes are available. At least one must be provided per invocation:

| Mode | Flags | Description |
|------|-------|-------------|
| **Named preset** | `--workload <name> --rate <N>` | Standard workload from `defaults.yaml`; identical token distributions to `blis run --workload <name>` |
| **Workload spec** | `--workload-spec <file>` | Multi-client workload from a YAML file |
| **Distribution synthesis** | `--rate <N>` | Single-client workload with custom token distributions (see Distribution Synthesis Flags) |
| **Closed-loop** | `--concurrency <N>` | Fixed pool of virtual users; arrival is response-driven (token distributions from Distribution Synthesis Flags) |

**Flag reference:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--workload` | `string` | `""` | Preset name (chatbot, summarization, contentgen, multidoc); requires `--rate` |
| `--workload-spec` | `string` | `""` | Path to WorkloadSpec YAML (alternative to `--workload` or `--rate`) |
| `--rate` | `float64` | `0` | Requests per second; required for `--workload` preset mode and distribution synthesis |
| `--concurrency` | `int` | `0` | Number of closed-loop virtual users; mutually exclusive with `--rate` |

**Combinations that produce an error:**

| Combination | Error |
|-------------|-------|
| `--workload` without `--rate` | preset requires a rate |
| `--workload` + `--workload-spec` | mutually exclusive |
| `--workload` + `--concurrency` | mutually exclusive |
| `--rate` + `--concurrency` | mutually exclusive |
| `--workload-spec` + `--concurrency` | use `clients[].concurrency` in the spec file instead |

!!! note
    `--workload-spec` takes priority over `--rate` if both are provided — the spec is used and `--rate` is ignored. All other distribution synthesis flags (`--prompt-tokens`, etc.) are similarly ignored when `--workload-spec` is active.

### Optional Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--api-key` | `string` | `""` | Bearer token for server authentication |
| `--server-type` | `string` | `"vllm"` | Server type (vllm, tgi, etc.) |
| `--max-concurrency` | `int` | `256` | Maximum simultaneous in-flight requests |
| `--warmup-requests` | `int` | `0` | Number of initial requests to exclude from trace |
| `--no-streaming` | `bool` | `false` | Disable streaming (use non-streaming HTTP) |
| `--seed` | `int64` | `42` | RNG seed for workload generation |
| `--horizon` | `int64` | `0` | Observation horizon in microseconds (0 = from spec or unlimited) |
| `--num-requests` | `int` | `0` | Maximum requests to generate (0 = from spec or unlimited) |
| `--think-time-ms` | `int` | `0` | Think time in ms between response and next request (concurrency mode only) |
| `--api-format` | `string` | `"completions"` | API format: `completions` or `chat` |
| `--unconstrained-output` | `bool` | `false` | Do not set `max_tokens` (let server decide output length) |
| `--rtt-ms` | `float64` | `0` | Measured network round-trip time in milliseconds |
| `--defaults-filepath` | `string` | `"defaults.yaml"` | Path to `defaults.yaml` containing preset definitions (preset mode only) |

### Distribution Synthesis Flags

Used when `--rate` or `--concurrency` mode is active (ignored when `--workload-spec` or `--workload <preset>` is provided):

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prompt-tokens` | `int` | `512` | Average prompt token count |
| `--prompt-tokens-stdev` | `int` | `256` | Prompt token standard deviation |
| `--prompt-tokens-min` | `int` | `2` | Minimum prompt tokens |
| `--prompt-tokens-max` | `int` | `7000` | Maximum prompt tokens |
| `--output-tokens` | `int` | `512` | Average output token count |
| `--output-tokens-stdev` | `int` | `256` | Output token standard deviation |
| `--output-tokens-min` | `int` | `2` | Minimum output tokens |
| `--output-tokens-max` | `int` | `7000` | Maximum output tokens |
| `--prefix-tokens` | `int` | `0` | Shared prefix token count |

### Examples

**Named preset mode** — drive the server with a standard workload (same shape as `blis run --workload chatbot`):

```bash
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --workload chatbot --rate 10 --num-requests 100 \
  --trace-header trace.yaml --trace-data trace.csv
```

**Workload-spec mode** — multi-client workload from a YAML spec:

```bash
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --workload-spec workload.yaml \
  --trace-header trace.yaml --trace-data trace.csv
```

**Rate mode** — quick experiment with distribution synthesis:

```bash
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --rate 10 --num-requests 100 \
  --prompt-tokens 256 --output-tokens 128 \
  --trace-header trace.yaml --trace-data trace.csv
```

**Chat completions API** — use `/v1/chat/completions` instead of `/v1/completions`:

```bash
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --api-format chat --workload-spec workload.yaml \
  --trace-header trace.yaml --trace-data trace.csv
```

**Non-streaming with network RTT** — disable SSE streaming and record network latency:

```bash
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --no-streaming --rtt-ms 2.5 --workload-spec workload.yaml \
  --trace-header trace.yaml --trace-data trace.csv
```

!!! info "Streaming and token counts"
    By default, observe uses streaming (SSE) and sends `stream_options: {include_usage: true}` to capture accurate token counts from the final SSE chunk. Non-streaming mode (`--no-streaming`) parses the full response body instead. Both modes extract `finish_reason` from server responses.

!!! info "Prefix sharing"
    When the workload spec defines prefix groups, observe builds deterministic prefix strings from a fixed vocabulary, seeded by the RNG seed and group name. This activates the server's prefix cache for realistic KV cache hit rates.

    Before dispatching requests, observe sends a single calibration request to measure the server's tokens-per-word ratio (typically 1.5–1.7 for BPE tokenizers). Prefix word counts are then scaled so the server tokenizes them to approximately the target `prefix_length` in the spec — matching what `blis run` simulates. The calibration result is logged at startup:

    ```
    INFO Prefix token calibration: 100 words → 167 server tokens (1.670 tokens/word)
    ```

    If calibration fails (server unreachable, timeout, or abnormal ratio), observe falls back to 1:1 word-to-token mapping with a warning.

!!! info "Session support"
    If the workload spec contains session clients, observe runs in closed-loop mode: each completed request may trigger follow-up requests from the session manager, interleaved with pre-generated arrivals by arrival time.

---

## `blis replay`

Replays a captured TraceV2 file through the BLIS discrete-event simulator. Instead of generating synthetic requests, replay loads real request timing and token counts from the trace.

### Replay-Specific Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--trace-header` | `string` | `""` | Path to TraceV2 header YAML (required) |
| `--trace-data` | `string` | `""` | Path to TraceV2 data CSV (required) |
| `--results-path` | `string` | `""` | File to write SimResult JSON for `blis calibrate` consumption |
| `--model` | `string` | `""` | LLM name (required) |

Replay also accepts all shared simulation config flags (`--latency-model`, `--total-kv-blocks`, `--max-num-running-reqs`, etc.) — the same flags available in `blis run`. See [Configuration](../reference/configuration.md) for the full list.

### How Replay Differs from `blis run`

| Aspect | `blis run` | `blis replay` |
|--------|-----------|---------------|
| **Request source** | Generated from workload spec or CLI distributions | Loaded from TraceV2 CSV |
| **Arrival times** | Synthesized by arrival process (Poisson, etc.) | Exact timestamps from trace |
| **Token counts** | Sampled from distributions | Actual observed values |
| **Horizon** | From `--horizon` flag or spec | Auto-computed as 2x max arrival time (override with `--horizon`) |
| **Output format** | Full `MetricsOutput` JSON | `SimResult` JSON array (request_id, ttft_us, e2e_us, input_tokens, output_tokens) |
| **Session support** | Session manager creates follow-ups | Session structure encoded in trace (no manager needed) |

!!! warning "Latency model matters"
    The replay command simulates token generation using the configured latency model. For accurate calibration, choose the latency model that best matches the server's behavior. See [Latency Models](latency-models.md) for guidance on selecting between roofline, blackbox, cross-model, and trained-roofline modes.

---

## `blis calibrate`

Compares real observed latencies (from `blis observe`) against simulator predictions (from `blis replay`) and produces a calibration report.

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--trace-header` | `string` | `""` | Path to TraceV2 header YAML (required) |
| `--trace-data` | `string` | `""` | Path to TraceV2 data CSV (required) |
| `--sim-results` | `string` | `""` | Path to SimResult JSON from `blis replay` (required) |
| `--report` | `string` | `""` | Path to write calibration report JSON (required) |
| `--warmup-requests` | `int` | `-1` | Requests to exclude from comparison (-1 = use trace header value, 0 = include all) |
| `--network-rtt-us` | `int64` | `-1` | Network RTT in microseconds added to sim-side latencies (-1 = use trace header value) |
| `--network-bandwidth-mbps` | `float64` | `0` | Network bandwidth in Mbps for upload/download delay (0 = no delay) |

!!! info "Sentinel defaults"
    The `--warmup-requests` and `--network-rtt-us` flags use `-1` as a sentinel meaning "read the value from the trace header." This allows the calibration to automatically use the warmup count and RTT recorded during observation. Pass `0` explicitly to override (include all requests or apply no RTT correction).

### Interpreting the Calibration Report

The calibration report JSON contains four sections:

**`trace_info`** — Summary of the input data:

```json
{
  "num_requests": 100,
  "warm_up_excluded": 5,
  "matched_pairs": 95,
  "token_mismatches": 2,
  "duration": "2m30s"
}
```

- `matched_pairs`: Requests matched by ID between trace and sim results
- `token_mismatches`: Pairs where observed and simulated token counts differ (indicates potential data quality issues)

**`metrics`** — Per-metric comparison (TTFT and E2E latency):

```json
{
  "ttft": {
    "RealP50": 1234.5,
    "SimP50": 1200.0,
    "RealP99": 4567.8,
    "SimP99": 4500.0,
    "MAPE": 0.05,
    "PearsonR": 0.95,
    "BiasDirection": "over-predict",
    "Quality": "good",
    "Count": 95
  },
  "e2e": { ... }
}
```

| Field | Meaning |
|-------|---------|
| `RealP50/P90/P95/P99` | Real (observed) latency percentiles in microseconds |
| `SimP50/P90/P95/P99` | Simulated latency percentiles in microseconds |
| `MAPE` | Mean Absolute Percentage Error (lower is better) |
| `PearsonR` | Pearson correlation coefficient (closer to 1.0 is better) |
| `BiasDirection` | `over-predict`, `under-predict`, or `neutral` |
| `Quality` | Rating: `excellent`, `good`, `fair`, or `poor` |

**`config_match`** — Tracks which simulator config parameters matched the observed server config (currently reports `matched` and `defaulted` arrays).

**`known_limitations`** — Documents known sources of sim/real divergence (batch step granularity, synthetic prefix tokens, speculative decoding).

---

## Worked Example

This walkthrough demonstrates the full pipeline: define a workload, observe a real vLLM server, replay through the simulator, and interpret the calibration report.

### Step 1: Define a workload

Create a workload spec (`workload.yaml`):

```yaml
rate: 5.0
num_requests: 50
clients:
  - id: "chat-user"
    rate_fraction: 1.0
    slo_class: "standard"
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 64
        min: 32
        max: 1024
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 32
        min: 16
        max: 512
```

### Step 2: Observe the real server

```bash
./blis observe \
  --server-url http://localhost:8000 \
  --model qwen/qwen3-14b \
  --workload-spec workload.yaml \
  --warmup-requests 5 \
  --trace-header trace.yaml \
  --trace-data trace.csv
```

This sends 50 requests to the server at ~5 req/s, excludes the first 5 from the trace (warmup), and writes the TraceV2 files.

### Step 3: Replay through the simulator

```bash
./blis replay \
  --trace-header trace.yaml \
  --trace-data trace.csv \
  --model qwen/qwen3-14b \
  --latency-model roofline \
  --results-path results.json
```

The simulator replays the same requests (arrival times, token counts) through the DES using the roofline latency model and writes per-request results.

### Step 4: Calibrate

```bash
./blis calibrate \
  --trace-header trace.yaml \
  --trace-data trace.csv \
  --sim-results results.json \
  --report calibration.json
```

The calibration command matches requests by ID, applies warmup exclusion and RTT normalization from the trace header, and produces the report.

### Step 5: Interpret results

```bash
cat calibration.json | python3 -m json.tool
```

Look for:

- **MAPE < 0.10** and **Quality = "good" or "excellent"** → simulator is well-calibrated for this workload
- **BiasDirection = "over-predict"** → simulator latencies are higher than reality (conservative)
- **BiasDirection = "under-predict"** → simulator latencies are lower than reality (optimistic — may need latency model tuning)
- **High token_mismatches** → data quality issue; check if the server truncated outputs

If calibration quality is poor, try:

1. **Different latency model:** Switch from `roofline` to `blackbox` or `crossmodel` (see [Latency Models](latency-models.md))
2. **Adjust server config flags:** Match `--max-num-running-reqs` and `--max-num-scheduled-tokens` to the real server's settings
3. **Increase sample size:** Use more requests (`--num-requests`) for statistical stability

---

## Tips

- **Warmup requests:** Always use `--warmup-requests` during observation to exclude cold-start latencies (JIT compilation, KV cache initialization) from the trace.
- **Network RTT:** If observing a remote server, measure RTT with `ping` and pass `--rtt-ms`. The calibrate command uses this to normalize sim-side latencies.
- **Reproducibility:** The `--seed` flag controls workload generation RNG. Same seed + same spec = same request sequence.
- **Graceful shutdown:** Press Ctrl+C during observation to stop gracefully — in-flight requests complete and all recorded data is written to the trace files.
- **Large workloads:** Use `--max-concurrency` to limit in-flight requests and avoid overwhelming the server.

# ServeGen-Informed Workload Generator: Design Document

**Date:** 2026-02-16
**Status:** Draft
**Author:** Sri (with Claude Code)
**Scope:** PR10 (single PR, multiple commits)
**References:**
- ServeGen paper: arxiv:2505.09999 (Alibaba, June 2025)
- ServeGen implementation: `results-old/ServeGen/`
- inference-perf: `results-old/inference-perf/` (kubernetes-sigs)
- BLIS macro plan: `docs/plans/2026-02-11-macro-implementation-plan-v2.md`

---

## 1. Executive Summary

This design extends BLIS from a simulation-only tool to a **full observe-predict-calibrate loop** for LLM inference performance:

1. **Specify** realistic workloads using ServeGen's client decomposition model (heterogeneous clients with skewed rates, bursty arrivals, empirical distributions)
2. **Observe** real inference server behavior by sending actual HTTP requests to vLLM/SGLang and recording per-request traces
3. **Predict** latency for the same workload using BLIS's discrete-event simulator with alpha/beta or roofline estimation
4. **Calibrate** by comparing real vs predicted KPIs with statistical rigor

The implementation is a single PR with multiple commits, each covering a logical layer of the system. The design follows BLIS's BDD/TDD conventions.

### Why This Matters

Simulators are only useful if their predictions match reality. Today, BLIS users must trust that alpha/beta coefficients are accurate without a way to verify. This design closes that gap: run the same workload against a real server and through the simulator, compare the results, and quantify prediction accuracy.

### Design Principles

1. **Tiered complexity**: Minimal config (alpha/beta + trace) gives useful results. Optional server config improves accuracy. No configuration is mandatory beyond what BLIS already requires.
2. **Separation of concerns**: Observe, predict, and compare are independent layers with independent correctness criteria.
3. **Backward compatibility**: Existing `--workload distribution` and `--workload traces` modes are untouched. New functionality is additive.
4. **Determinism**: All workload generation is reproducible given the same seed.
5. **Standard protocols**: Real mode uses OpenAI-compatible API (supported by vLLM, SGLang, TGI, and others).

---

## 2. Architecture Overview

```
                     WorkloadSpec (YAML)
                           |
                    WorkloadGenerator
                    /              \
                   v                v
           Real Mode               Sim Mode
          (observe)               (predict)
              |                       |
              v                       v
     vLLM/SGLang Server        BLIS Discrete-Event
     (OpenAI API)              Simulator
              |                       |
              v                       v
        Trace v2 File           Sim Metrics
        (per-request            (per-request
         real latencies)         predicted latencies)
              \                      /
               v                    v
            Calibration Report (compare)
            - Per-metric error analysis
            - Statistical significance
            - Known limitation annotations
```

### Package Layout

```
sim/
  request.go            # Extended: TenantID, SLOClass, modality fields
  ...                   # Existing files untouched

sim/workload/           # NEW — pure request generation (no HTTP, no metrics)
  spec.go               # WorkloadSpec, ClientSpec, DistSpec types + YAML loading
  client.go             # ClientPool generation (Zipf rates, heterogeneous distributions)
  arrival.go            # ArrivalSampler interface + Poisson, Gamma, Weibull
  distribution.go       # LengthSampler interface + ParetoLogNormal, Exponential, Gaussian, EmpiricalPDF
  generator.go          # WorkloadGenerator: clients -> []*sim.Request
  network.go            # Per-client network delay computation (RTT + bandwidth)
  scenarios.go          # Built-in scenario presets
  multimodal.go         # Multimodal distribution sampling, modality token generation
  reasoning.go          # Reasoning model multi-turn, reason_ratio, conversation chains

sim/cluster/
  ...                   # Existing files
  tracev2.go            # NEW: Trace v2 format — loading, parsing, header, export
  calibrate.go          # NEW: Real-vs-sim comparison, statistical metrics, report

cmd/
  ...                   # Existing files
  observe.go            # NEW: Real mode HTTP client (OpenAI-compatible API),
                        #   streaming/non-streaming, response recording
```

**Import boundary rule:** `sim/` and `sim/cluster/` MUST NOT import `sim/workload/`. The flow is one-directional: `sim/workload/` imports `sim/` to construct `*sim.Request` objects. `sim/cluster/` imports `sim/` for cluster orchestration. `cmd/` imports all three and orchestrates the full pipeline.

```
cmd/ ──────> sim/workload/  ──> sim/
  |                              ^
  └────────> sim/cluster/ ───────┘
  |               ^
  └───────────────┘
```

---

## 3. Core Types

### 3.1 WorkloadSpec (YAML Configuration)

```go
// WorkloadSpec is the top-level workload configuration.
// Loaded from YAML via LoadWorkloadSpec(path).
type WorkloadSpec struct {
    // Version of the spec format. Currently "1".
    Version string `yaml:"version"`

    // Seed for deterministic workload generation.
    // Two runs with the same seed and spec produce identical request sequences.
    Seed int64 `yaml:"seed"`

    // Category of LLM workload.
    // "language" (default), "multimodal", or "reasoning".
    Category string `yaml:"category"`

    // Clients defines heterogeneous client behaviors.
    // If empty, a default single-client spec is generated from top-level params.
    Clients []ClientSpec `yaml:"clients"`

    // AggregateRate is the target total request rate (requests/second).
    // Individual client rates are derived from their RateFraction relative weights.
    AggregateRate float64 `yaml:"aggregate_rate"`

    // Horizon overrides the simulation/recording duration (microseconds).
    // If zero, uses the CLI --horizon value.
    Horizon int64 `yaml:"horizon,omitempty"`
}
```

### 3.2 ClientSpec

```go
// ClientSpec defines a single client's workload behavior.
// Inspired by ServeGen's per-client decomposition (Finding 5):
// heterogeneous clients with skewed rates explain aggregate dynamics.
type ClientSpec struct {
    // ID uniquely identifies this client.
    ID string `yaml:"id"`

    // TenantID groups clients into tenants for fairness metrics.
    TenantID string `yaml:"tenant_id"`

    // SLOClass categorizes latency requirements: "realtime", "interactive", "batch".
    // Used for per-SLO-class metric aggregation and SLOAttainment computation.
    SLOClass string `yaml:"slo_class"`

    // RateFraction is this client's relative share of AggregateRate.
    // All clients' RateFractions are normalized to sum to 1.0.
    RateFraction float64 `yaml:"rate_fraction"`

    // Arrival process configuration.
    Arrival ArrivalSpec `yaml:"arrival"`

    // Input token length distribution.
    InputDist DistSpec `yaml:"input_distribution"`

    // Output token length distribution.
    OutputDist DistSpec `yaml:"output_distribution"`

    // PrefixGroup identifies clients sharing a common input prefix.
    // Clients with the same PrefixGroup get identical prefix token sequences.
    // Empty means no prefix sharing.
    PrefixGroup string `yaml:"prefix_group,omitempty"`

    // Streaming indicates whether this client uses streaming responses.
    // Affects which latency metrics are collected (TTFT/ITL only for streaming).
    Streaming bool `yaml:"streaming"`

    // Network defines client-side network characteristics.
    Network *NetworkSpec `yaml:"network,omitempty"`

    // Lifecycle defines when this client is active/inactive.
    // If nil, client is active for the entire duration.
    Lifecycle *LifecycleSpec `yaml:"lifecycle,omitempty"`

    // Multimodal configuration (only for category: multimodal).
    Multimodal *MultimodalSpec `yaml:"multimodal,omitempty"`

    // Reasoning configuration (only for category: reasoning).
    Reasoning *ReasoningSpec `yaml:"reasoning,omitempty"`
}
```

### 3.3 Supporting Spec Types

```go
// ArrivalSpec configures the inter-arrival time process.
type ArrivalSpec struct {
    // Process type: "poisson", "gamma", "weibull".
    // - poisson: Exponential inter-arrivals (CV = 1.0)
    // - gamma: Gamma-distributed inter-arrivals (CV > 1 = bursty)
    // - weibull: Weibull-distributed inter-arrivals
    // See ServeGen Finding 1: Gamma fits large models, Weibull fits mid-size.
    Process string `yaml:"process"`

    // CV is the coefficient of variation (std_dev / mean of inter-arrival times).
    // CV > 1.0 indicates bursty arrivals. Only used for gamma and weibull.
    // For poisson, CV is always 1.0 (ignored if set).
    CV *float64 `yaml:"cv,omitempty"`
}

// DistSpec parameterizes a token length distribution.
type DistSpec struct {
    // Type of distribution:
    // - "pareto_lognormal": Pareto + LogNormal mixture (ServeGen input model)
    // - "exponential": Exponential distribution (ServeGen output model)
    // - "gaussian": Clamped Gaussian (backward-compatible with existing BLIS)
    // - "empirical": Empirical PDF from data file (ServeGen-faithful)
    Type string `yaml:"type"`

    // Params holds distribution-specific parameters.
    // pareto_lognormal: {alpha, xm, mu, sigma, mix_weight}
    //   mix_weight = probability of drawing from Pareto (vs LogNormal)
    // exponential: {mean}
    // gaussian: {mean, std_dev, min, max}
    // empirical: {file} — path to JSON file with PDF array
    Params map[string]float64 `yaml:"params,omitempty"`

    // File path for empirical distribution (alternative to params["file"]).
    File string `yaml:"file,omitempty"`
}

// NetworkSpec defines client-side network characteristics.
type NetworkSpec struct {
    // RTT is the round-trip time in milliseconds.
    // Added to TTFT and E2E when comparing with real measurements.
    RTTMs float64 `yaml:"rtt_ms"`

    // BandwidthMbps is the client's network bandwidth in megabits per second.
    // Used to compute upload delay for multimodal inputs and download delay for responses.
    BandwidthMbps float64 `yaml:"bandwidth_mbps,omitempty"`
}

// LifecycleSpec defines client activity windows.
// Models ServeGen Finding 5: aggregate dynamics emerge from client churn.
type LifecycleSpec struct {
    // Windows defines active periods for this client.
    // Outside these windows, the client generates no requests.
    // Timestamps are in microseconds from simulation start.
    Windows []ActiveWindow `yaml:"windows"`
}

// ActiveWindow represents a period when a client is active.
type ActiveWindow struct {
    StartUs int64 `yaml:"start_us"`
    EndUs   int64 `yaml:"end_us"`
}

// MultimodalSpec configures multimodal request generation.
type MultimodalSpec struct {
    // TextDist distribution for text token count per request.
    TextDist DistSpec `yaml:"text_distribution"`

    // ImageDist distribution for per-image token count.
    ImageDist DistSpec `yaml:"image_distribution"`

    // ImageCountDist distribution for number of images per request.
    ImageCountDist DistSpec `yaml:"image_count_distribution"`

    // AudioDist distribution for per-audio-segment token count.
    AudioDist DistSpec `yaml:"audio_distribution"`

    // AudioCountDist distribution for number of audio segments per request.
    AudioCountDist DistSpec `yaml:"audio_count_distribution"`

    // VideoDist distribution for per-video-segment token count.
    VideoDist DistSpec `yaml:"video_distribution"`

    // VideoCountDist distribution for number of video segments per request.
    VideoCountDist DistSpec `yaml:"video_count_distribution"`
}

// ReasoningSpec configures reasoning model workload generation.
type ReasoningSpec struct {
    // ReasonRatioDist distribution for reason_tokens / total_output_tokens.
    // Bimodal in practice (ServeGen Finding 9).
    ReasonRatioDist DistSpec `yaml:"reason_ratio_distribution"`

    // MultiTurn enables multi-turn conversation simulation.
    MultiTurn *MultiTurnSpec `yaml:"multi_turn,omitempty"`
}

// MultiTurnSpec configures multi-turn conversation behavior.
type MultiTurnSpec struct {
    // MaxRounds is the maximum number of rounds per conversation.
    MaxRounds int `yaml:"max_rounds"`

    // ThinkTimeUs is the client-side delay between receiving a response
    // and sending the next round (microseconds). Models user "thinking time".
    ThinkTimeUs int64 `yaml:"think_time_us"`

    // ContextGrowth defines how input tokens grow per round.
    // "accumulate": input = system_prompt + all prior (prompt+response) + new question
    //   (matches inference-perf's LocalUserSession behavior)
    // "fixed": each round has independent input length (no context accumulation)
    ContextGrowth string `yaml:"context_growth"`
}
```

### 3.4 Request Struct Extensions

All new fields are added in PR10. They are additive — existing code is unaffected:

```go
// In sim/request.go — additions to existing Request struct:
type Request struct {
    // ... all existing fields preserved ...

    // TenantID identifies the client/tenant that generated this request.
    // Empty for legacy workloads (--workload distribution/traces).
    TenantID string

    // SLOClass categorizes latency requirements: "realtime", "interactive", "batch".
    // Empty for legacy workloads.
    SLOClass string

    // Streaming indicates whether this request uses streaming response mode.
    // When true, TTFT and ITL metrics are meaningful.
    // When false, only E2E latency is meaningful.
    Streaming bool

    // SessionID links multi-turn conversation rounds.
    // Empty for single-turn requests.
    SessionID string

    // RoundIndex is the 0-based round number within a multi-turn session.
    RoundIndex int

    // Modality breakdown (all zero for text-only requests):
    TextTokenCount  int // Number of text input tokens
    ImageTokenCount int // Total image tokens across all images
    AudioTokenCount int // Total audio tokens across all segments
    VideoTokenCount int // Total video tokens across all segments

    // ReasonRatio is the fraction of output tokens that are reasoning tokens.
    // Only meaningful for reasoning model workloads. Zero for language/multimodal.
    ReasonRatio float64
}
```

These fields are additive — all existing code that reads Request continues to work. `InputTokens` and `OutputTokens` (the `[]int` slices) remain the authoritative token sequences. The new count fields are metadata for metrics aggregation.

**YAML loading pattern:** `LoadWorkloadSpec(path)` follows the same pattern established by `LoadPolicyBundle` in `sim/bundle.go` (PR8): strict YAML parsing with `KnownFields(true)`, a `Validate()` method, and `*float64` pointer types for optional fields to distinguish zero from unset.

---

## 4. Arrival Samplers

### Interface

```go
// ArrivalSampler generates inter-arrival times for a client.
type ArrivalSampler interface {
    // SampleIAT returns the next inter-arrival time in microseconds.
    SampleIAT(rng *rand.Rand) int64
}
```

### Implementations

**PoissonSampler**: `IAT ~ Exponential(1/rate)`. CV is always 1.0.
```go
type PoissonSampler struct {
    rateMicros float64 // requests per microsecond
}
```

**GammaSampler**: `IAT ~ Gamma(shape=1/CV^2, scale=CV^2/rate)`.
For CV > 1.0, produces bursty arrivals. ServeGen Finding 1: best fit for M-large models.
```go
type GammaSampler struct {
    shape float64 // 1/CV^2
    scale float64 // CV^2 / rate (in microseconds)
}
```
Go implementation note: `math/rand` doesn't have a Gamma sampler. We implement using Marsaglia and Tsang's method (standard, well-tested algorithm for Gamma random variates).

**WeibullSampler**: `IAT ~ Weibull(shape, scale)` derived from rate and CV.
ServeGen Finding 1: best fit for M-mid models.
```go
type WeibullSampler struct {
    shape float64 // Weibull shape parameter
    scale float64 // Weibull scale parameter (in microseconds)
}
```
Go implementation: inverse CDF method (`scale * (-ln(1-U))^(1/shape)`).

### Factory

```go
func NewArrivalSampler(spec ArrivalSpec, ratePerMicrosecond float64) ArrivalSampler
```

---

## 5. Length Distribution Samplers

### Interface

```go
// LengthSampler generates token count samples.
type LengthSampler interface {
    // Sample returns a non-negative token count.
    Sample(rng *rand.Rand) int
}
```

### Implementations

**Important design note:** ServeGen's actual implementation uses **empirical PDFs** (not parametric distributions) for per-client generation. The paper's parametric findings (Pareto+LogNormal for input, Exponential for output) describe the *aggregate* distribution shape across all clients — they are analytical observations, not the generation mechanism. Our implementation supports both approaches:

- **Empirical PDF** (ServeGen-faithful): Primary mode for realistic workloads. Samples via inverse CDF from raw probability arrays. Can load ServeGen data files directly.
- **Parametric** (convenience): For quick experimentation where users want to specify distributions with a few parameters rather than providing data files.

**EmpiricalPDFSampler** (PRIMARY — ServeGen-faithful):
Direct PDF sampling via inverse CDF. Stores a cumulative distribution array and samples via binary search (`sort.SearchFloat64s`).
```go
type EmpiricalPDFSampler struct {
    cdf []float64 // Cumulative distribution function. Index i = token count i.
    // cdf[i] = P(X <= i). Binary search on uniform random gives sample.
}
```

**ParetoLogNormalSampler** (parametric convenience):
Mixture model for input lengths (matches ServeGen Finding 3 aggregate shape).
With probability `mix_weight`, draw from Pareto(alpha, xm); otherwise LogNormal(mu, sigma).
```go
type ParetoLogNormalSampler struct {
    alpha     float64 // Pareto shape
    xm        float64 // Pareto scale (minimum value)
    mu        float64 // LogNormal mean of ln(X)
    sigma     float64 // LogNormal std dev of ln(X)
    mixWeight float64 // Probability of Pareto draw
}
```

**ExponentialSampler** (parametric convenience):
For output lengths (matches ServeGen Finding 3 aggregate shape).
```go
type ExponentialSampler struct {
    mean float64 // 1/lambda
}
```

**GaussianSampler** (backward-compatible):
Clamped Gaussian matching existing BLIS behavior.
```go
type GaussianSampler struct {
    mean, stdDev float64
    min, max     int
}
```

**EmpiricalPDFSampler**: Direct PDF sampling via inverse CDF (ServeGen-faithful).
Stores a cumulative distribution array and samples via binary search.
```go
type EmpiricalPDFSampler struct {
    cdf []float64 // Cumulative distribution function
    // Index i represents token count i. cdf[i] = P(X <= i).
}
```

### Factory

```go
func NewLengthSampler(spec DistSpec) (LengthSampler, error)
```

---

## 6. Workload Generator

### Core Generation Pipeline

```go
// GenerateRequests creates a request sequence from a WorkloadSpec.
// Deterministic given the same spec and seed.
func GenerateRequests(spec *WorkloadSpec, horizon int64) ([]*sim.Request, error)
```

**Algorithm:**

1. Parse and validate WorkloadSpec.
2. Create per-client ArrivalSampler and LengthSampler instances.
3. Normalize RateFractions so they sum to 1.0.
4. For each client:
   a. Compute client rate = `AggregateRate * client.RateFraction`.
   b. If client has Lifecycle, only generate during active windows.
   c. Sample inter-arrival times via ArrivalSampler.
   d. For each arrival: sample input/output lengths via LengthSampler.
   e. Generate synthetic token IDs (`GenerateRandomTokenIDs`) of sampled lengths.
   f. For clients in the same PrefixGroup, prepend shared prefix tokens.
   g. Set TenantID, SLOClass, Streaming, and modality metadata on each Request.
5. Merge all clients' requests, sort by ArrivalTime.
6. Assign sequential IDs.

### Multi-Turn Generation

For reasoning clients with `MultiTurn` enabled:

1. Generate round 0 as a normal request with a new SessionID.
2. Round N+1's arrival time = round N's estimated completion time + ThinkTimeUs.
   - In sim mode: completion is estimated via alpha/beta (fast, deterministic).
   - In real mode: round N+1 is only generated after round N's response arrives.
3. If `ContextGrowth == "accumulate"`:
   - Round N+1's input tokens = system_prefix + sum(all prior input + output tokens) + new question tokens.
   - This is computed at generation time for sim mode (using pre-specified output lengths).
   - In real mode, actual output token counts from the server drive this.
4. Each round gets the same SessionID with incrementing RoundIndex.

---

## 7. Trace v2 Format

### Header File (YAML)

The trace header captures metadata for reproducibility and calibration:

```yaml
trace_version: 2
time_unit: microseconds
created_at: "2026-02-16T14:30:00Z"
mode: real                          # "real" or "generated"
warm_up_requests: 10                # Requests to exclude from calibration
workload_spec: "workload.yaml"     # Original spec file (if generated)

# Optional — improves calibration accuracy when provided
server:
  type: vllm                        # "vllm", "sglang", "tgi"
  model: meta-llama/Llama-3.1-8B-Instruct
  tensor_parallel: 1
  max_num_seqs: 256
  block_size: 16
  gpu_memory_utilization: 0.9
  max_model_len: 8192
  quantization: null
  enforce_eager: false

hardware:
  gpu: H100
  gpu_count: 1

network:
  measured_rtt_ms: 2.3
```

### Request Data (CSV)

```csv
request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks
0,client_42,tenant_A,realtime,sess_7,0,group_X,true,512,128,512,0,0,0,0.0,0,1708100000000,1708100000045200,1708100001590300,47
1,client_42,tenant_A,realtime,sess_7,1,group_X,true,698,64,698,0,0,0,0.0,1590500,,,,,
```

**Column definitions:**

| Column | Type | Description |
|--------|------|-------------|
| `request_id` | int | Sequential request ID |
| `client_id` | string | Client that generated this request |
| `tenant_id` | string | Tenant for fairness grouping |
| `slo_class` | string | "realtime", "interactive", "batch" |
| `session_id` | string | Multi-turn session (empty for single-turn) |
| `round_index` | int | Round within session (0-based) |
| `prefix_group` | string | Shared prefix group ID |
| `streaming` | bool | Whether streaming was used |
| `input_tokens` | int | Input token count (from server usage) |
| `output_tokens` | int | Output token count (from server usage) |
| `text_tokens` | int | Text portion of input (multimodal only) |
| `image_tokens` | int | Total image tokens (multimodal only) |
| `audio_tokens` | int | Total audio tokens (multimodal only) |
| `video_tokens` | int | Total video tokens (multimodal only) |
| `reason_ratio` | float | Reasoning token fraction (reasoning only) |
| `arrival_time_us` | int64 | Intended arrival time (from workload spec) |
| `send_time_us` | int64 | Actual send time (real mode, wall-clock us) |
| `first_chunk_time_us` | int64 | First response chunk received (real mode) |
| `last_chunk_time_us` | int64 | Last response chunk received (real mode) |
| `num_chunks` | int | Number of SSE chunks received (streaming) |

**Real mode** populates all columns. **Sim mode** (when loading traces) uses `arrival_time_us`, `input_tokens`, `output_tokens`, and metadata columns. The `send_time_us`, `first_chunk_time_us`, `last_chunk_time_us` columns carry the real measurements for calibration comparison.

---

### Error and Failure Records

The trace CSV includes additional columns for recording failures:

| Column | Type | Description |
|--------|------|-------------|
| `status` | string | "ok", "error", "timeout" |
| `error_message` | string | Server error message (empty for "ok") |

Failed requests are recorded in the trace but **excluded from calibration statistics** by default. The calibration report includes a section on error rates:

```json
{
  "request_summary": {
    "total": 1000,
    "ok": 985,
    "error": 10,
    "timeout": 5,
    "excluded_warm_up": 10,
    "calibrated": 975
  }
}
```

---

## 8. Real Mode

### Concurrent Load Generation

Real mode must sustain production-scale request rates. The architecture uses Go's goroutine concurrency model:

```go
// LoadGenerator dispatches requests at the target rate using concurrent workers.
type LoadGenerator struct {
    numWorkers      int            // Concurrent sender goroutines (default: GOMAXPROCS)
    maxConcurrency  int            // Max in-flight requests per worker (default: 100)
    client          *RealClient    // Shared HTTP client (connection pooling)
    requestCh       chan *PendingRequest  // Buffered channel for request dispatch
    resultCh        chan *RequestRecord   // Results channel for recorder
}
```

**Dispatch flow:**
1. `WorkloadGenerator` produces requests with target send times (from arrival samplers).
2. `LoadGenerator` places requests on `requestCh` at their scheduled times.
3. `numWorkers` goroutines pull from `requestCh` and send HTTP requests concurrently.
4. Each worker records `send_time`, `first_chunk_time`, `last_chunk_time` per request.
5. Results flow to `Recorder` via `resultCh`.

**Schedule delay tracking** (per inference-perf methodology): For each request, the `schedule_delay = actual_send_time - intended_send_time`. If median schedule delay exceeds 10ms, the load generator is saturating and the trace header records a warning:

```yaml
load_generator:
  achieved_rate: 98.5
  requested_rate: 100.0
  schedule_delay_p50_ms: 2.3
  schedule_delay_p99_ms: 15.1
  saturated: false         # true if p50 > 10ms
```

**Max throughput:** `numWorkers * maxConcurrency` concurrent in-flight requests. With Go's efficient goroutine scheduling and HTTP/2 connection multiplexing, this sustains 10K+ QPS on modern hardware (comparable to inference-perf's multiprocessing architecture).

### HTTP Client

```go
// RealClient sends requests to an OpenAI-compatible inference server.
// Uses http.Client with connection pooling for concurrent access.
type RealClient struct {
    baseURL    string
    apiKey     string          // optional
    httpClient *http.Client    // shared, goroutine-safe with connection pool
    modelName  string
    serverType string          // "vllm", "sglang", "tgi"
}
```

Supports:
- **Completion API** (`/v1/completions`): Prompt-based, simpler
- **Chat API** (`/v1/chat/completions`): Message-based, required for multi-turn

For streaming:
- Parses SSE chunks (`data: {...}\n\n` lines)
- Records timestamp of each chunk for ITL measurement
- Extracts `usage` from the **final chunk** only (per vLLM behavior — intermediate chunks have unreliable incremental counts)

For non-streaming:
- Single response with `usage` field containing exact token counts

### Server-Side Metrics Scraping (Optional)

When `--prometheus-url` is provided, real mode scrapes server-side Prometheus metrics during the run:

```yaml
# Captured in trace header
server_metrics:
  scrape_interval_ms: 1000
  captured:
    - vllm:num_requests_running
    - vllm:num_requests_waiting
    - vllm:gpu_cache_usage_perc
    - vllm:avg_prompt_throughput_toks_per_s
    - vllm:avg_generation_throughput_toks_per_s
```

These time-series are stored alongside the trace and provide server-side ground truth for calibration. This enables comparing:
- Real queue depth vs sim queue depth over time
- Real batch size vs sim batch size over time
- Real KV cache utilization vs sim KV cache utilization over time

Server metrics scraping is **optional** — calibration works without it (using client-side measurements only), but with it the calibration report can pinpoint exactly where sim diverges from reality.

### Recording Pipeline

```go
// Recorder captures per-request timing and metrics (goroutine-safe).
type Recorder struct {
    mu      sync.Mutex
    records []TraceRecord
    header  TraceHeader
}

// RecordRequest captures one request-response cycle.
func (r *Recorder) RecordRequest(req RequestRecord) { ... }

// Export writes trace v2 files (header YAML + data CSV).
func (r *Recorder) Export(headerPath, dataPath string) error { ... }
```

### Multi-Turn in Real Mode

1. Generate round 0 from WorkloadSpec.
2. Send round 0 to server, wait for response.
3. Extract `completion_tokens` from response `usage`.
4. Compute round 1's input: original input count + completion_tokens + new question tokens.
5. Generate synthetic input tokens of the computed length (for sim replay).
6. Repeat until MaxRounds or session ends.

The key insight: real mode uses **server-reported token counts** for context growth, not client-side tokenization. This ensures the trace captures exactly what the server processed.

### Server Config Discovery

When `--server-type vllm` is provided, real mode attempts:
1. `GET /v1/models` to discover model name
2. Parse `--server-config server.yaml` if provided (recommended)
3. Fall back to sensible defaults if neither is available

---

## 9. Sim Mode: Trace v2 Replay

### Loading Traces

```go
// LoadTraceV2 reads a trace v2 header + CSV and returns requests + metadata.
func LoadTraceV2(headerPath, dataPath string) (*TraceV2, error)

type TraceV2 struct {
    Header   TraceHeader
    Requests []*sim.Request
    RealMetrics []RealRequestMetric // per-request real latencies (for calibration)
}
```

### Synthetic Token ID Generation

When loading trace v2 into sim mode:

1. For each request, generate random token IDs of length `input_tokens` and `output_tokens`.
2. For requests in the same `prefix_group`:
   - Generate a shared prefix token sequence once per group (deterministic from seed + group name).
   - Prepend the shared prefix to each request's input tokens.
3. For multi-turn sessions:
   - Round 0 gets fresh token IDs.
   - Round N > 0 gets: shared prefix + accumulated context tokens + new question tokens.
   - Accumulated context length = sum of prior rounds' (input + output) token counts from the trace.

This ensures BLIS's KV cache prefix matching works correctly without needing actual token IDs from the real server.

### Server Config Matching

If the trace header contains `server` configuration:
- `max_num_seqs` maps to BLIS's `MaxBatchSize`
- `block_size` maps to BLIS's KV cache block size
- `max_model_len` maps to BLIS's maximum context length

If not provided, BLIS uses its existing defaults. The calibration report notes which parameters were matched vs defaulted.

---

## 10. Network Latency Model

### Model

For each request with a NetworkSpec:

```
upload_delay_us = (input_bytes / bandwidth_bytes_per_us)
download_delay_us = (output_bytes / bandwidth_bytes_per_us)

# Approximate bytes: 4 bytes per token (token IDs are int32)
input_bytes = input_tokens * 4
output_bytes = output_tokens * 4

client_ttft_us = server_ttft_us + rtt_us + upload_delay_us
client_e2e_us = server_e2e_us + rtt_us + upload_delay_us + download_delay_us
```

For multimodal requests, upload delay includes image/audio/video token bytes.

### Where It Applies

- **Sim mode**: Network delay is added to sim-predicted latencies when producing client-perspective metrics.
- **Calibration**: When comparing real vs sim, the sim's client-perspective metrics (with network) are compared against real measurements (which inherently include network).
- **Without network config**: RTT defaults to 0, bandwidth defaults to infinity. Sim metrics represent pure server-side latency.

---

## 11. Calibration Framework

### Methodology

The calibration compares **per-request** real measurements against sim predictions for the same request sequence.

**Compared metrics (when available):**

| Metric | Real Source | Sim Source | Requires |
|--------|-----------|-----------|----------|
| TTFT | `first_chunk_time - send_time` | `FirstTokenTime + network_rtt` | Streaming |
| E2E | `last_chunk_time - send_time` | `CompletionTime - ArrivalTime + network_rtt` | Any |
| TPOT | `(E2E - TTFT) / (output_tokens - 1)` | Same formula on sim times | Streaming, output > 1 |
| Throughput | `total_output_tokens / wall_clock_duration` | `total_output_tokens / sim_duration` | Any |

**Statistical measures per metric:**

Basic (always computed):
- **MAPE** (Mean Absolute Percentage Error): Average of |real - sim| / real across all requests.
- **Pearson r**: Correlation between real and sim values. High r means the sim captures relative ordering even if absolute values are off.
- **Per-percentile error**: Compare P50, P90, P95, P99 of real vs sim distributions.
- **Bias direction**: Is sim systematically over-predicting or under-predicting?

Distributional (for rigorous comparison):
- **Kolmogorov-Smirnov test**: Two-sample KS test between real and sim latency distributions. Reports D-statistic and p-value. Rejects null hypothesis (distributions are the same) when p < 0.05.
- **QQ data**: Quantile-quantile pairs (real quantiles vs sim quantiles at 100 evenly-spaced percentiles). Enables QQ plot visualization. Perfect calibration produces points on the y=x line.
- **Bootstrap confidence intervals**: 95% CI for MAPE and Pearson r via 1000 bootstrap resamples. Reports whether the CI includes target thresholds.

When server-side Prometheus metrics are available (optional):
- **Time-series comparison**: Per-metric MAE between real and sim time-series (queue depth, batch size, KV utilization) at matching time points.

### Warm-Up Handling

Requests with `request_id < warm_up_requests` (from trace header) are excluded from calibration statistics but included in the trace for completeness.

### Calibration Report

```go
type CalibrationReport struct {
    TraceInfo struct {
        NumRequests      int
        WarmUpExcluded   int
        Duration         string
        ServerConfigProvided bool
    }
    Metrics map[string]MetricComparison // "ttft", "e2e", "tpot", "throughput"
    ConfigMatch struct {
        Matched  []string // e.g., ["max_num_seqs=256", "block_size=16"]
        Defaulted []string // e.g., ["gpu_memory_utilization (no server config)"]
    }
    KnownLimitations []string
}

type MetricComparison struct {
    RealP50, SimP50     float64
    RealP90, SimP90     float64
    RealP95, SimP95     float64
    RealP99, SimP99     float64
    MAPE                float64
    PearsonR            float64
    BiasDirection       string // "over-predict", "under-predict", "neutral"
    ErrorP50Pct         float64
    ErrorP99Pct         float64
}
```

### Calibration Quality Guidance

The calibration report includes a quality rating based on observed metrics:

| Rating | MAPE | Pearson r | Interpretation |
|--------|------|-----------|---------------|
| **Excellent** | < 10% | > 0.95 | Sim predictions are highly accurate. Alpha/beta coefficients well-tuned for this workload. |
| **Good** | 10-20% | 0.85-0.95 | Sim captures trends correctly. Useful for capacity planning and relative comparisons. |
| **Fair** | 20-35% | 0.70-0.85 | Sim gives directional guidance only. Consider providing server config or re-tuning coefficients. |
| **Poor** | > 35% | < 0.70 | Sim predictions unreliable for this workload. Likely cause: mismatched server config, incorrect alpha/beta, or workload outside trained regime. |

These thresholds are guidelines, not pass/fail gates. The report always includes raw numbers for users to make their own judgments.

### Edge Case Handling

| Edge Case | Behavior |
|-----------|----------|
| All clients same prefix group | Valid. Prefix cache will be heavily utilized. May cause unrealistic cache hit rates if real server has different eviction timing. Noted in report. |
| Client rate fraction = 0 | Client is excluded from generation. Warning logged. |
| Very short horizon (< 1 second) | Valid but calibration statistics may be unreliable. Report warns: "Insufficient requests for statistical significance (N < 30)." |
| Very high aggregate rate | Generator produces requests as fast as the rate demands. In real mode, schedule delay is recorded. If delay > 10% of inter-arrival time, report warns about load generator saturation. |
| Server returns no usage field | Error recorded. Request excluded from calibration. Warning: "Server did not return usage metrics. Token counts estimated from response length (less accurate)." |
| Server timeout | Recorded as status="timeout". Excluded from calibration. Error rate reported. |
| NaN/Inf in distribution params | Rejected during WorkloadSpec validation with descriptive error message. |

### Known Limitations (Always Included)

The calibration report always includes these annotations:

1. **Prefix cache divergence**: "Sim constructs synthetic prefix token IDs. Prefix cache hit rates may differ from real server, especially after evictions."
2. **Speculative decoding not modeled**: "If the real server uses speculative decoding, actual token generation patterns differ from sim's sequential model."

---

## 12. CLI Interface

### New Flags

```
# Generate workload from spec and run sim
--workload-spec workload.yaml

# Real mode: send to server and record trace
--real-mode
--server-url http://localhost:8000
--server-type vllm                   # "vllm", "sglang", "tgi"
--server-config server.yaml          # optional, recommended for calibration accuracy
--api-type chat                      # "chat" or "completion"
--api-key ""                         # optional, for authenticated endpoints
--trace-output traces/              # directory for trace v2 files

# Calibrate: compare real trace against sim
--calibrate traces/trace-header.yaml
--calibration-output calibration.json
```

### Example Workflows

**Workflow 1: Pure simulation (existing BLIS, unchanged)**
```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --workload distribution --rate 10
```

**Workflow 2: ServeGen-style simulation**
```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --workload-spec workload.yaml \
  --num-instances 4 --routing-policy weighted
```

**Workflow 3: Real mode (observe)**
```bash
./simulation_worker run \
  --workload-spec workload.yaml \
  --real-mode \
  --server-url http://localhost:8000 \
  --server-type vllm \
  --api-type chat \
  --trace-output traces/
```

**Workflow 4: Sim replay of real trace (predict)**
```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --workload traces \
  --workload-traces-filepath traces/trace-data.csv \
  --trace-header traces/trace-header.yaml
```

**Workflow 5: Calibrate (compare)**
```bash
./simulation_worker calibrate \
  --model meta-llama/llama-3.1-8b-instruct \
  --calibrate traces/trace-header.yaml \
  --calibration-output calibration.json
```

---

## 13. Commit Grouping

PR10 is a single PR with multiple commits. The former PR10a-g sub-PRs map to logical commit groups:

| Commit Group | Focus | ~LOC | Key Files |
|-------------|-------|------|-----------|
| **1. Spec + Samplers** | `WorkloadSpec`, `ClientSpec`, `ArrivalSampler` (Poisson/Gamma/Weibull), `LengthSampler` (ParetoLogNormal/Exponential/Gaussian/EmpiricalPDF), YAML loading, validation | ~800 | `sim/workload/spec.go`, `arrival.go`, `distribution.go`, `client.go`, `generator.go`, `sim/request.go`, `cmd/root.go` |
| **2. Real Mode** | HTTP client (OpenAI-compatible, streaming + non-streaming), per-request timing, trace v2 export (header YAML + data CSV), error recording | ~700 | `cmd/observe.go`, `sim/workload/tracev2.go`, `sim/workload/recorder.go` |
| **3. Trace Replay** | Trace v2 loading, synthetic token ID generation from counts + prefix groups, server config matching | ~500 | `sim/workload/tracev2.go`, `sim/workload/generator.go`, `sim/cluster/workload.go` |
| **4. Calibration** | Per-request real-vs-sim comparison, MAPE/Pearson r/per-percentile error, warm-up exclusion, `calibrate` subcommand | ~500 | `sim/workload/calibrate.go`, `cmd/root.go` |
| **5. Multimodal + Reasoning** | Per-modality tokens, ReasonRatio, multi-turn conversations, context accumulation, lifecycle windows | ~600 | `sim/workload/multimodal.go`, `sim/workload/reasoning.go`, `sim/request.go` |
| **6. Network Model** | RTT + bandwidth per client, client-perspective TTFT/E2E adjustment | ~400 | `sim/workload/network.go` |
| **7. Metrics Extensions** | Per-SLO-class distributions, SLOAttainment, JainFairnessIndex, scenario presets | ~400 | `sim/cluster/metrics.go`, `sim/workload/scenarios.go` |

**Total:** ~3,900 LOC across ~12 new files + 4 modified files.

---

## 14. Example Workload Spec

```yaml
version: "1"
seed: 42
category: language
aggregate_rate: 100.0  # 100 requests/second total

clients:
  - id: "high-volume-batch"
    tenant_id: "tenant-A"
    slo_class: "batch"
    rate_fraction: 0.7          # 70% of traffic
    streaming: false
    arrival:
      process: gamma
      cv: 3.5                   # Bursty
    input_distribution:
      type: pareto_lognormal
      params:
        alpha: 1.5
        xm: 50
        mu: 5.5
        sigma: 1.2
        mix_weight: 0.3
    output_distribution:
      type: exponential
      params:
        mean: 256
    prefix_group: "shared-system-prompt"
    network:
      rtt_ms: 2.0
      bandwidth_mbps: 1000

  - id: "realtime-chat"
    tenant_id: "tenant-B"
    slo_class: "realtime"
    rate_fraction: 0.3          # 30% of traffic
    streaming: true
    arrival:
      process: poisson          # Less bursty (interactive users)
    input_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 50
        min: 10
        max: 2048
    output_distribution:
      type: exponential
      params:
        mean: 64
    network:
      rtt_ms: 50.0              # Remote users
      bandwidth_mbps: 100
```

---

## 15. Known Limitations and Future Work

### Modeling Limitations

1. **Continuous batching**: BLIS models discrete batch steps. Real servers use iteration-level scheduling. This is a fundamental approximation that affects TTFT prediction under high load.

2. **Prefix cache fidelity**: Sim constructs synthetic token IDs. Cache hit patterns may differ from real servers, especially under eviction pressure.

3. **Speculative decoding**: Not modeled. If the real server uses speculative decoding, token generation timing differs from sim's sequential model.

4. **Quantization effects**: BLIS's alpha/beta coefficients may not capture quantization-specific latency characteristics. Roofline mode with hardware config is more accurate for quantized models.

### Native ServeGen Data File Loading (In Scope)

ServeGen ships production-derived per-client data files. BLIS supports loading these directly:

```yaml
# workload-spec using ServeGen data
version: "1"
seed: 42
category: language
aggregate_rate: 100.0
servegen_data:
  path: "data/language/m-large"   # Directory containing chunk-*-trace.csv and chunk-*-dataset.json
  span_start: 0                    # Start time in seconds (optional, default 0)
  span_end: 3600                   # End time in seconds (optional, default: full duration)
```

When `servegen_data` is specified:
1. Load all `chunk-{id}-trace.csv` files → per-client arrival patterns (rate, CV, Gamma/Weibull params per 600s window)
2. Load all `chunk-{id}-dataset.json` files → per-client empirical PDFs (input_tokens, output_tokens per ~6h window)
3. Each chunk becomes a `ClientSpec` with:
   - `ArrivalSpec` derived from the trace CSV (pattern type + fitted params)
   - `InputDist` and `OutputDist` as empirical PDFs from the dataset JSON
   - `RateFraction` from the client's rate relative to total
4. Generate requests using the standard pipeline

This is first-class support, not a conversion tool. Users can point BLIS at ServeGen's `data/` directory and get production-realistic workloads immediately.

**Multimodal and reasoning data files** are also supported:
- `data/multimodal/mm-image/` → clients with text_tokens, image_tokens, audio_tokens, video_tokens, output_tokens
- `data/reason/deepseek-r1/` → clients with input_tokens, output_tokens, reason_ratio

### Future Extensions

1. **Automated alpha/beta fitting**: Use calibration data to automatically tune alpha/beta coefficients (gradient-free optimization to minimize MAPE).

2. **Continuous batching model**: Replace discrete steps with iteration-level scheduling for higher fidelity at the cost of simulation speed.

3. **Integration with inference-perf**: Share workload spec format so inference-perf can generate ServeGen-style workloads using the same YAML.

### Deferred to Micro-Plan (Address During Implementation)

These items were identified during design review but deferred to implementation-level planning:

1. **Backpressure / schedule-delay tracking**: Real mode should report when the load generator cannot maintain target rate (per inference-perf methodology).
2. **Multi-turn chat API message construction**: Specify exact chat messages array format for multi-turn (system prompt, role alternation, context window overflow handling).
3. **CI strategy for real mode**: Define mock server approach for testing HTTP client in CI without a real inference server.
4. **Server error recovery**: Define retry/backoff strategy for 429 (rate limit) and 503 (overloaded) responses.

---

## 16. References

1. **ServeGen**: Xiang et al., "ServeGen: Workload Characterization and Generation of Large Language Model Serving in Production," arXiv:2505.09999, 2025.
2. **inference-perf**: kubernetes-sigs/inference-perf, GenAI inference performance benchmarking tool.
3. **vLLM**: vllm-project/vllm, high-throughput LLM serving engine.
4. **SGLang**: sgl-project/sglang, fast serving framework for LLMs.
5. **BLIS macro plan**: `docs/plans/2026-02-11-macro-implementation-plan-v2.md`

# Training & Validation Data - Experiment Collection

## Hardware Platform

All experiments were collected on **NVIDIA H100 80GB HBM3 (SXM)** GPUs in the OpenShift cluster.

- **GPU Model**: NVIDIA-H100-80GB-HBM3
- **GPU Family**: Hopper (compute capability 9.0)
- **CUDA Runtime**: 12.8
- **Driver Version**: 570.211.01

## Dataset Overview

This directory contains **15 experiments** from vLLM inference runs:

- **12 experiments with full tracing**: Journey traces (`traces.json`), KV cache events (`kv_events.jsonl`), and performance metrics → **Used for reconstruction and coefficient fitting**
- **3 experiments with metrics only**: Performance metrics without journey traces → **Used for evaluation only**

| Data Type | With full tracing (12) | Metrics only (3) |
|-----------|:----------------------:|:----------------:|
| Journey traces (`traces.json`) | ✓ | ✗ |
| KV events (`kv_events.jsonl`) | ✓ | ✗ |
| Performance metrics | ✓ | ✓ |
| vLLM logs | ✓ | ✓ |

**🔄 Dataset Update (2026-03-30)**: Original reasoning experiments (3) were collected from an overloaded server (85% failure rate, 4-minute timeout latencies). They have been **replaced with reasoning-lite experiments** (3) collected under normal operating conditions. Reasoning-lite uses reduced workload intensity (1 req/s vs original higher rates), resulting in 100% success rate and normal latency distributions.

### Model Coverage

**Dense Models**:
- **Llama-2-7B-HF**: TP=1 → 1 GPU, 4 experiments with full tracing (codegen, roleplay, general, reasoning-lite)
- **Llama-3.1-70B-Instruct**: TP=4 → 4 GPUs, 2 experiments with full tracing (general-lite, codegen)
- **Mistral-Nemo-Instruct-2407 (12B)**: TP=1-2 → 1-2 GPUs, 2 experiments with full tracing (codegen, general-lite)
- **Qwen2.5-7B-Instruct**: TP=1 → 1 GPU, 2 experiments with full tracing (roleplay, reasoning-lite)
- **Yi-34B**: TP=2 → 2 GPUs, 1 experiment with full tracing (general-lite)

**MoE Models**:
- **Llama-4-Scout-17B-16E** (17B parameters, 16 experts): TP=2 → 2 GPUs, 4 experiments
  - 3 with metrics only (general, codegen, roleplay)
  - 1 with full tracing (reasoning-lite)
  - Expert Parallelism: EP=TP*DLP (auto-enabled for MoE)

### Naming Conventions

Experiments follow two naming patterns:

**Timestamp-based** (4 experiments):
```
YYYYMMDD-HHMMSS-{model}-tp{N}-{workload}
```
Example: `20260217-155451-llama-2-7b-tp1-codegen`

**ID-based** (11 experiments):
```
{id}-{model}-tp{N}-{workload}-{variant}
```
Examples: `63-mistral-nemo-12b-tp1-codegen-1-1`, `17-llama-4-scout-17b-16e-tp2-general-2`, `66-qwen2-5-7b-instruct-tp1-reasoning-lite-1-1`

## Workload Types

- **codegen**: Code generation tasks (single-turn completions)
- **roleplay**: Conversational role-play scenarios (multi-turn)
- **general**: General-purpose completions (mixed use cases)
- **general-lite**: Reduced-intensity general workload (lower token counts or rates)
- **reasoning-lite**: Multi-turn chat with shared prefix caching (reduced intensity, 1 req/s)
  - Uses `shared_prefix` data type with multi-turn chat enabled
  - System prompt (100 tokens) + question (934 tokens) + output (1448 tokens) ≈ 2482 tokens/request
  - 23 unique system prompts, 1 user per prompt
  - Designed for normal server operation (vs original reasoning workload which caused overload)

## Directory Structure

All experiments share this structure, with tracing files (`traces.json`, `kv_events.jsonl`) present only in experiments with full tracing (12 of 15):

```
<experiment>/
├── exp-config.yaml              # vLLM server configuration
├── profile.yaml                 # Workload profile (single-line JSON)
├── traces.json                  # OTEL journey traces (only in 11 experiments with full tracing)
├── kv_events.jsonl              # KV cache events (only in 11 experiments with full tracing)
├── vllm.log                     # vLLM server logs
├── vllm_logging.json            # Structured vLLM logs
└── results/
    ├── config.yaml
    ├── per_request_lifecycle_metrics.json
    ├── stage_0_lifecycle_metrics.json
    ├── stage_1_lifecycle_metrics.json
    ├── summary_lifecycle_metrics.json
    ├── stdout.log
    └── stderr.log
```

## File Schemas

### 1. `exp-config.yaml`

vLLM server configuration.

**Key fields**: `model`, `tensor_parallelism`, `data_parallelism`, `max_num_batched_tokens`, `max_model_len`, `max_num_seqs`

### 2. `profile.yaml`

Workload profile specification (single-line JSON).

**Key sections**: `api` (type, streaming), `data` (token distributions), `load` (multi-stage rates), `server` (vLLM config)

### 3. `traces.json`

**Availability**: 11 experiments with full tracing

OpenTelemetry (OTEL) journey traces in OTLP JSON format. Per-step batch composition and timing data.

**Key Fields**: `step.id`, `step.ts_start_ns`, `step.ts_end_ns`, `batch.num_prefill_reqs`, `batch.num_decode_reqs`, `batch.scheduled_tokens`, `kv.usage_gpu_ratio`

**Usage**: Required for `reconstruct_steps.py` and coefficient fitting.

### 4. `kv_events.jsonl`

**Availability**: 11 experiments with full tracing

KV cache block operations (line-delimited JSON). Event types: `BlockStored`, `CacheStoreCommitted`, `TransferInitiated`, `TransferCompleted`

**Usage**: Available for KV cache analysis (not currently used in reconstruction pipeline).

### 5. `per_request_lifecycle_metrics.json`

Per-request latency breakdown. Large file (hundreds of MB).

**Key fields**: `request_id`, `ttft`, `e2e_latency`, `inter_token_latency_mean`, `num_input_tokens`, `num_output_tokens`

**Usage**: Ground truth for `evaluate.py`.

### 6. `summary_lifecycle_metrics.json`

Experiment-level aggregated metrics (P50/P95/P99 latencies, throughput, error counts).

### 7. `stage_N_lifecycle_metrics.json`

Per-stage aggregated metrics (same schema as summary, scoped to load stage).

## Experiment List

| Experiment | Model | TP | DLP | GPUs | Workload | Traces? |
|------------|-------|----|-----|------|----------|:-------:|
| **With Full Tracing (12)** | | | | | | |
| 20260217-155451-llama-2-7b-tp1-codegen | Llama-2-7B-HF | 1 | 1 | 1 | codegen | ✓ |
| 20260217-162547-llama-2-7b-tp1-roleplay | Llama-2-7B-HF | 1 | 1 | 1 | roleplay | ✓ |
| 20260217-231439-llama-2-7b-tp1-general | Llama-2-7B-HF | 1 | 1 | 1 | general | ✓ |
| 48-llama-4-scout-17b-16e-tp2-reasoning-lite-2-1 | Scout-17B-16E | 2 | 1 | 2 | reasoning-lite | ✓ |
| 60-llama-3-1-70b-tp4-general-lite-4-1 | Llama-3.1-70B | 4 | 1 | 4 | general-lite | ✓ |
| 61-llama-3-1-70b-tp4-codegen-4-1 | Llama-3.1-70B | 4 | 1 | 4 | codegen | ✓ |
| 62-mistral-nemo-12b-tp2-general-lite-2-1 | Mistral-Nemo-12B | 2 | 1 | 2 | general-lite | ✓ |
| 63-mistral-nemo-12b-tp1-codegen-1-1 | Mistral-Nemo-12B | 1 | 1 | 1 | codegen | ✓ |
| 64-qwen2-5-7b-instruct-tp1-roleplay-1-1 | Qwen2.5-7B | 1 | 1 | 1 | roleplay | ✓ |
| 65-01-ai-yi-34b-tp2-general-lite-2-1 | Yi-34B | 2 | 1 | 2 | general-lite | ✓ |
| 66-qwen2-5-7b-instruct-tp1-reasoning-lite-1-1 | Qwen2.5-7B | 1 | 1 | 1 | reasoning-lite | ✓ |
| 67-llama-2-7b-hf-tp1-reasoning-lite-1-1 | Llama-2-7B-HF | 1 | 1 | 1 | reasoning-lite | ✓ |
| **Metrics Only (3)** | | | | | | |
| 17-llama-4-scout-17b-16e-tp2-general-2 | Scout-17B-16E | 2 | 1 | 2 | general | ✗ |
| 20-llama-4-scout-17b-16e-tp2-codegen-2 | Scout-17B-16E | 2 | 1 | 2 | codegen | ✗ |
| 21-llama-4-scout-17b-16e-tp2-roleplay-2 | Scout-17B-16E | 2 | 1 | 2 | roleplay | ✗ |

**Key**:
- **TP**: Tensor Parallelism (model sharding across GPUs)
- **DLP**: Data-Local Parallelism (multiple batches per pod, within TP×DLP GPUs)
- **GPUs**: Total GPUs = TP × DLP

## Data Provenance

- **Collection Framework**: inference-sim data collection pipeline
- **vLLM Version**: inference-sim/vllm fork (with journey and step tracing)
- **Cluster**: OpenShift production cluster (pokprod)
- **GPU Nodes**: SYS-821GE-TNHR with 8× H100-80GB-HBM3 per node

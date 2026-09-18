# Quickstart: LoRA Control-Plane Subsystem

**Feature**: `007-lora-control-plane`

This shows the intended end-state usage once all 7 PRs land. Earlier PRs deliver
subsets (e.g., PR1–2 give adapter metrics + capacity with no cost effect).

## 1. Verify the no-op default (INV-6 / SC-001)

With no LoRA config, output is unchanged:

```bash
go build -o blis main.go
./blis run --model qwen/qwen3-14b > baseline.json      # pre- and post-feature identical
```

## 2. Declare an adapter registry + capacity

`lora-workload.yaml` (config file):

```yaml
lora:
  adapter_capacity: 4
  adapters:
    - {id: adapter_hot,  rank: 8}
    - {id: adapter_warm, rank: 16}
    - {id: adapter_cold, rank: 32}

clients:
  - {id: c_hot,  model: qwen/qwen3-14b, adapter: adapter_hot,  rate_fraction: 0.7, arrival: {process: poisson}, input_distribution: {...}, output_distribution: {...}}
  - {id: c_cold, model: qwen/qwen3-14b, adapter: adapter_cold, rate_fraction: 0.3, arrival: {process: poisson}, input_distribution: {...}, output_distribution: {...}}
```

```bash
./blis run --model qwen/qwen3-14b --config lora-workload.yaml | jq .adapters
# => per-adapter load_count / ttft / throughput
```

## 3. Turn on cost physics + LoRA-aware routing

```bash
./blis run --model qwen/qwen3-14b --config lora-workload.yaml \
  --lora-scorer-weight 2 \
  --scorers "lora-affinity:2,queue-depth:1,kv-utilization:1"
```

- Cold `adapter_cold` requests show a higher TTFT tail (load latency) than warm ones (SC-004).
- Batches mixing more distinct adapters take longer per step (US3-2).
- With `lora-affinity` weighted in, requests route to instances already holding their adapter → fewer loads/evictions vs an adapter-blind router (SC-005).

## 4. Determinism & run/replay parity

```bash
./blis run --model qwen/qwen3-14b --config lora-workload.yaml --seed 42 > a.json
./blis run --model qwen/qwen3-14b --config lora-workload.yaml --seed 42 > b.json
diff a.json b.json        # empty (INV-6)

./blis run --model qwen/qwen3-14b --config lora-workload.yaml --trace-output t
./blis replay --trace-header t.yaml --trace-data t.csv --model qwen/qwen3-14b --config lora-workload.yaml
# per-request metrics identical (INV-13)
```

## 5. (Optional) Fidelity vs the Digital Twin (PR7)

```bash
./blis calibrate --trace-header t.yaml --trace-data t.csv --sim-results results.json \
  --adapter-reference dt_llama31_8b.yaml --report calibration.json
# per-adapter TTFT and throughput MAPE vs DT reference (SC-007 target ≤ 20% on both)
```

## Acceptance smoke checklist

- [ ] No-op run byte-identical to baseline (SC-001, INV-6).
- [ ] `adapters` block present with per-adapter TTFT/throughput when configured (US1).
- [ ] Resident set never exceeds capacity (SC-003; invariant test).
- [ ] Cold TTFT > warm TTFT for same input (SC-004).
- [ ] `lora-affinity` reduces loads under skewed popularity (SC-005).
- [ ] `go build ./... && go test ./... -count=1 && golangci-lint run ./...` all clean.

#!/bin/bash
# H-Liveness: Scheduler Liveness Under Admissible Load
#
# Hypothesis: For ALL scheduler configurations (FCFS, SJF, priority-FCFS) at
# arrival rates below saturation (rho < 0.9), every admitted request should
# eventually complete (zero still_queued, zero still_running at simulation end),
# and the queue length trace should be bounded (no monotonic growth).
#
# Classification: Deterministic (liveness is exact pass/fail)
# Family: Scheduler invariants (safety/liveness)
# VV&UQ: Verification
#
# Experiment design:
#   - Schedulers: fcfs, sjf, priority-fcfs
#   - Workloads: uniform (input=128, output=128) and mixed (50% short + 50% long)
#   - Instances: 4, routing: least-loaded
#   - Seeds: 42, 123, 456
#   - Round 1: Rate=100 req/s (rho ~0.3), 500 requests, uniform + mixed workloads
#   - Round 2: Rate=230 req/s (rho ~0.7) and Rate=280 req/s (rho ~0.85), 2000 requests, mixed workload only
#   - Saturation derivation: ~328 req/s for 4 instances (see FINDINGS.md)
#
# Design notes:
#   ED-1: Controlled comparison — only scheduler varies between configs
#   ED-2: Rate well below saturation ensures no capacity-related drops
#   ED-3: Precondition — rho << 1, all requests must complete
#   ED-5: Reproducible — deterministic seeds, builds binary
#   ED-6: Reference: hypotheses/h12-conservation/run.sh (conservation experiment)
#         Config diff: this experiment varies scheduler (h12 varied routing+admission).
#         Same model, same beta coefficients, similar request counts.
#
# Reference: https://github.com/inference-sim/inference-sim/issues/313
#
# Usage: ./run.sh [--rebuild]
#
# Requires: Go 1.24+, Python 3 (standard library only)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$REPO_ROOT/blis"

# Build if needed
if [[ "${1:-}" == "--rebuild" ]] || [[ ! -x "$BINARY" ]]; then
    echo "Building blis..."
    (cd "$REPO_ROOT" && go build -o blis main.go)
fi

MODEL="meta-llama/llama-3.1-8b-instruct"
SEEDS=(42 123 456)
NUM_INSTANCES=4
SCHEDULERS=(fcfs sjf priority-fcfs)

# Rate configurations:
#   Round 1: rho~0.3  → rate=100 req/s, 500 requests, uniform + mixed workloads
#   Round 2: rho~0.7  → rate=230 req/s, 2000 requests, mixed workload only
#            rho~0.85 → rate=280 req/s, 2000 requests, mixed workload only
#
# Saturation derivation (approximate):
#   Beta coefficients: [6910.42, 17.67, 2.84]
#   stepTime = 6910.42 + 17.67*cacheMissTokens + 2.84*decodeTokens (microseconds)
#   Mixed workload average: 50% short (32in/64out) + 50% long (512in/256out)
#   Average output tokens = (64+256)/2 = 160
#   With 4 instances and batching, empirical saturation ~328 req/s for uniform.
#   Mixed workload has higher average service time; effective saturation may be lower.
#   Rates chosen: 230 (~0.7x), 280 (~0.85x) of ~328 req/s.

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# Generate uniform workload YAML (all requests same size)
# Args: seed, outfile, rate, num_requests
make_uniform_workload() {
    local seed=$1
    local outfile=$2
    local rate=$3
    local num_requests=$4

    cat > "$outfile" << YAMLEOF
version: "1"
seed: $seed
category: language
aggregate_rate: $rate
num_requests: $num_requests
clients:
  - id: "uniform"
    tenant_id: "default"
    slo_class: "interactive"
    rate_fraction: 1.0
    streaming: false
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 128
    output_distribution:
      type: constant
      params:
        value: 128
YAMLEOF
}

# Generate mixed workload YAML (50% short + 50% long via two clients)
# Args: seed, outfile, rate, num_requests
make_mixed_workload() {
    local seed=$1
    local outfile=$2
    local rate=$3
    local num_requests=$4

    cat > "$outfile" << YAMLEOF
version: "1"
seed: $seed
category: language
aggregate_rate: $rate
num_requests: $num_requests
clients:
  - id: "short"
    tenant_id: "default"
    slo_class: "realtime"
    rate_fraction: 0.5
    streaming: false
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 32
    output_distribution:
      type: constant
      params:
        value: 64
  - id: "long"
    tenant_id: "default"
    slo_class: "batch"
    rate_fraction: 0.5
    streaming: false
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 512
    output_distribution:
      type: constant
      params:
        value: 256
YAMLEOF
}

run_sim() {
    local stdout_file=$1
    local workload_yaml=$2
    local seed=$3
    local sched=$4
    local priority_policy=$5

    timeout 600 "$BINARY" run \
        --model "$MODEL" \
        --num-instances "$NUM_INSTANCES" \
        --workload-spec "$workload_yaml" \
        --seed "$seed" \
        --scheduler "$sched" \
        --priority-policy "$priority_policy" \
        --routing-policy least-loaded \
        --admission-policy always-admit \
        --total-kv-blocks 1000000 \
        --log error \
        2>/dev/null \
        > "$stdout_file" \
        || echo "    WARNING: timeout or error for $sched/$seed"
}

echo "============================================================================"
echo "  H-Liveness: Scheduler Liveness Under Admissible Load"
echo "  Reference: issue #313, docs/standards/experiments.md (scheduler invariants)"
echo "  Type: Deterministic (liveness = pass/fail)"
echo "  Family: Scheduler invariants | VV&UQ: Verification"
echo "============================================================================"
echo ""
echo "  Config: ${NUM_INSTANCES} instances, least-loaded routing, always-admit"
echo "  Schedulers: ${SCHEDULERS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo ""

# ── Round 1: rho~0.3 — All scheduler x workload x seed combinations ──────

ROUND1_RATE=100
ROUND1_REQUESTS=500

echo "  === Round 1: rho~0.3 (rate=${ROUND1_RATE} req/s, ${ROUND1_REQUESTS} requests) ==="
echo "  Workloads: uniform + mixed"
echo ""

for sched in "${SCHEDULERS[@]}"; do
    priority_policy="constant"
    if [[ "$sched" == "priority-fcfs" ]]; then
        priority_policy="slo-based"
    fi

    for workload_type in uniform mixed; do
        for seed in "${SEEDS[@]}"; do
            echo "  Running: scheduler=${sched} workload=${workload_type} seed=${seed} rate=${ROUND1_RATE} ..."

            wl="$RESULTS_DIR/${sched}_${workload_type}_r${ROUND1_RATE}_s${seed}_wl.yaml"
            stdout="$RESULTS_DIR/${sched}_${workload_type}_r${ROUND1_RATE}_s${seed}_stdout.txt"

            if [[ "$workload_type" == "uniform" ]]; then
                make_uniform_workload "$seed" "$wl" "$ROUND1_RATE" "$ROUND1_REQUESTS"
            else
                make_mixed_workload "$seed" "$wl" "$ROUND1_RATE" "$ROUND1_REQUESTS"
            fi

            run_sim "$stdout" "$wl" "$seed" "$sched" "$priority_policy"
        done
    done
done

# ── Round 2: High-load experiments — mixed workload only ──────────────────

ROUND2_RATES=(230 280)  # rho~0.7, rho~0.85
ROUND2_REQUESTS=2000

echo ""
echo "  === Round 2: High-load (rates=${ROUND2_RATES[*]} req/s, ${ROUND2_REQUESTS} requests, mixed only) ==="
echo ""

for rate in "${ROUND2_RATES[@]}"; do
    for sched in "${SCHEDULERS[@]}"; do
        priority_policy="constant"
        if [[ "$sched" == "priority-fcfs" ]]; then
            priority_policy="slo-based"
        fi

        for seed in "${SEEDS[@]}"; do
            echo "  Running: scheduler=${sched} workload=mixed seed=${seed} rate=${rate} ..."

            wl="$RESULTS_DIR/${sched}_mixed_r${rate}_s${seed}_wl.yaml"
            stdout="$RESULTS_DIR/${sched}_mixed_r${rate}_s${seed}_stdout.txt"

            make_mixed_workload "$seed" "$wl" "$rate" "$ROUND2_REQUESTS"
            run_sim "$stdout" "$wl" "$seed" "$sched" "$priority_policy"
        done
    done
done

# ── Round 2b: Constrained-batch experiments — force queueing ──────────────
# With max-num-running-reqs=256 (default), batching is so efficient that the
# queue is effectively always empty even at rate=280. To exercise the scheduler's
# ordering logic, we constrain the batch to 8 requests per instance, which
# forces queueing at high rates and makes scheduler order observable.

ROUND2B_RATE=280
ROUND2B_REQUESTS=2000
ROUND2B_MAX_RUNNING=8

echo ""
echo "  === Round 2b: Constrained-batch (rate=${ROUND2B_RATE}, max-running=${ROUND2B_MAX_RUNNING}, ${ROUND2B_REQUESTS} requests, mixed only) ==="
echo ""

run_sim_constrained() {
    local stdout_file=$1
    local workload_yaml=$2
    local seed=$3
    local sched=$4
    local priority_policy=$5
    local max_running=$6

    timeout 600 "$BINARY" run \
        --model "$MODEL" \
        --num-instances "$NUM_INSTANCES" \
        --workload-spec "$workload_yaml" \
        --seed "$seed" \
        --scheduler "$sched" \
        --priority-policy "$priority_policy" \
        --routing-policy least-loaded \
        --admission-policy always-admit \
        --total-kv-blocks 1000000 \
        --max-num-running-reqs "$max_running" \
        --log error \
        2>/dev/null \
        > "$stdout_file" \
        || echo "    WARNING: timeout or error for $sched/$seed"
}

for sched in "${SCHEDULERS[@]}"; do
    priority_policy="constant"
    if [[ "$sched" == "priority-fcfs" ]]; then
        priority_policy="slo-based"
    fi

    for seed in "${SEEDS[@]}"; do
        echo "  Running: scheduler=${sched} workload=mixed seed=${seed} rate=${ROUND2B_RATE} max-running=${ROUND2B_MAX_RUNNING} ..."

        wl="$RESULTS_DIR/${sched}_mixed_r${ROUND2B_RATE}_b${ROUND2B_MAX_RUNNING}_s${seed}_wl.yaml"
        stdout="$RESULTS_DIR/${sched}_mixed_r${ROUND2B_RATE}_b${ROUND2B_MAX_RUNNING}_s${seed}_stdout.txt"

        make_mixed_workload "$seed" "$wl" "$ROUND2B_RATE" "$ROUND2B_REQUESTS"
        run_sim_constrained "$stdout" "$wl" "$seed" "$sched" "$priority_policy" "$ROUND2B_MAX_RUNNING"
    done
done

# ── Round 2c: Token budget isolation — verify binding constraint ─────────
# Round 2b showed SJF 31% faster than FCFS with max-running=8. But the default
# --max-num-scheduled-tokens=2048 may be the actual binding constraint (not the
# request count limit). This round raises the token budget to 65536 to isolate.

ROUND2C_RATE=280
ROUND2C_REQUESTS=2000
ROUND2C_MAX_RUNNING=8
ROUND2C_TOKEN_BUDGET=65536

echo ""
echo "  === Round 2c: Token budget isolation (rate=${ROUND2C_RATE}, max-running=${ROUND2C_MAX_RUNNING}, tokens=${ROUND2C_TOKEN_BUDGET}, mixed only) ==="
echo ""

run_sim_token_isolation() {
    local stdout_file=$1
    local workload_yaml=$2
    local seed=$3
    local sched=$4
    local priority_policy=$5
    local max_running=$6
    local token_budget=$7

    timeout 600 "$BINARY" run \
        --model "$MODEL" \
        --num-instances "$NUM_INSTANCES" \
        --workload-spec "$workload_yaml" \
        --seed "$seed" \
        --scheduler "$sched" \
        --priority-policy "$priority_policy" \
        --routing-policy least-loaded \
        --admission-policy always-admit \
        --total-kv-blocks 1000000 \
        --max-num-running-reqs "$max_running" \
        --max-num-scheduled-tokens "$token_budget" \
        --log error \
        2>/dev/null \
        > "$stdout_file" \
        || echo "    WARNING: timeout or error for $sched/$seed"
}

for sched in "${SCHEDULERS[@]}"; do
    priority_policy="constant"
    if [[ "$sched" == "priority-fcfs" ]]; then
        priority_policy="slo-based"
    fi

    for seed in "${SEEDS[@]}"; do
        echo "  Running: scheduler=${sched} workload=mixed seed=${seed} rate=${ROUND2C_RATE} max-running=${ROUND2C_MAX_RUNNING} tokens=${ROUND2C_TOKEN_BUDGET} ..."

        wl="$RESULTS_DIR/${sched}_mixed_r${ROUND2C_RATE}_t${ROUND2C_TOKEN_BUDGET}_s${seed}_wl.yaml"
        stdout="$RESULTS_DIR/${sched}_mixed_r${ROUND2C_RATE}_t${ROUND2C_TOKEN_BUDGET}_s${seed}_stdout.txt"

        make_mixed_workload "$seed" "$wl" "$ROUND2C_RATE" "$ROUND2C_REQUESTS"
        run_sim_token_isolation "$stdout" "$wl" "$seed" "$sched" "$priority_policy" "$ROUND2C_MAX_RUNNING" "$ROUND2C_TOKEN_BUDGET"
    done
done

echo ""
echo "============================================================================"
echo "  Analysis"
echo "============================================================================"
echo ""

python3 "$SCRIPT_DIR/analyze.py" \
    --results-dir "$RESULTS_DIR" \
    --schedulers "${SCHEDULERS[*]}" \
    --seeds "${SEEDS[*]}" \
    --round1-rate "$ROUND1_RATE" \
    --round1-requests "$ROUND1_REQUESTS" \
    --round2-rates "${ROUND2_RATES[*]}" \
    --round2-requests "$ROUND2_REQUESTS" \
    --round2b-rate "$ROUND2B_RATE" \
    --round2b-requests "$ROUND2B_REQUESTS" \
    --round2b-max-running "$ROUND2B_MAX_RUNNING" \
    --round2c-rate "$ROUND2C_RATE" \
    --round2c-requests "$ROUND2C_REQUESTS" \
    --round2c-max-running "$ROUND2C_MAX_RUNNING" \
    --round2c-token-budget "$ROUND2C_TOKEN_BUDGET"

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis"
echo "============================================================================"

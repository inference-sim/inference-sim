#!/bin/bash
# H-Cross-Model — Cross-model generalization validation
# Validates 15 confirmed behavioral findings with Qwen/Qwen2.5-7B-Instruct (H100, TP=1)
# Reference: Issue #396
# Config diff (ED-6): vs llama-3.1-8b (H100/TP=2):
#   model: Qwen/Qwen2.5-7B-Instruct (vs meta-llama/llama-3.1-8b-instruct)
#   alpha: [4680.3, 0.0, 0.0] (vs [1601.35, 3.51, 1805.54]) — alpha1,alpha2=0
#   beta: [7051.8, 19.5, 25.4] (vs [6910.42, 17.67, 2.84]) — beta2 is 9x higher
#   kv_blocks: 65833 (vs 132139)
#   tp: 1 (vs 2), max_scheduled_tokens: 4096 (vs 2048)
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"

setup_experiment "${1:-}"

# =============================================================================
# Qwen model configuration (from issue #396)
# =============================================================================
QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"
QWEN_ALPHA="4680.303204056608,0.0,0.0"
QWEN_BETA="7051.796874715078,19.538416565504026,25.431830886933543"
QWEN_KV_BLOCKS=65833
QWEN_MAX_RUNNING=256
QWEN_MAX_SCHEDULED=4096
QWEN_TP=1

# Common Qwen flags for every blis_run call
QWEN_COMMON="--model $QWEN_MODEL --hardware H100 --tp $QWEN_TP \
  --alpha-coeffs $QWEN_ALPHA --beta-coeffs $QWEN_BETA \
  --total-kv-blocks $QWEN_KV_BLOCKS \
  --max-num-running-reqs $QWEN_MAX_RUNNING \
  --max-num-scheduled-tokens $QWEN_MAX_SCHEDULED"

INSTANCES=4

# Helper: run with Qwen params and 4 instances
qwen_run() {
    local timeout=$1
    local output=$2
    shift 2
    blis_run "$timeout" "$output" $QWEN_COMMON --num-instances $INSTANCES "$@"
}

# =============================================================================
# EXPERIMENT 1: H12 — Conservation (INV-1)
# Claim: injected == completed + still_queued + still_running + dropped_unservable
# Type: Deterministic (single seed sufficient)
# Uses --rate CLI mode (no workload YAML needed)
# =============================================================================
echo "=== EXPERIMENT: H12 — Conservation ===" >&2

POLICIES=(
    "round-robin fcfs always-admit"
    "least-loaded fcfs always-admit"
    "round-robin sjf always-admit"
    "round-robin priority-fcfs always-admit"
    "least-loaded sjf always-admit"
    "weighted fcfs always-admit"
    "round-robin fcfs token-bucket"
    "least-loaded fcfs token-bucket"
    "weighted priority-fcfs always-admit"
    "weighted sjf always-admit"
)

for i in "${!POLICIES[@]}"; do
    read -r routing scheduler admission <<< "${POLICIES[$i]}"
    out="$RESULTS_DIR/h12_${routing}_${scheduler}_${admission}.txt"
    qwen_run $TIMEOUT_STANDARD "$out" \
        --rate 500 --num-requests 200 --seed 42 \
        --routing-policy "$routing" --scheduler "$scheduler" \
        --admission-policy "$admission" --log error || true
done

# =============================================================================
# EXPERIMENT 2: H13 — Determinism (INV-6)
# Claim: Same seed → byte-identical output
# Type: Deterministic (uses --rate CLI mode)
# =============================================================================
echo "=== EXPERIMENT: H13 — Determinism ===" >&2

H13_POLICIES=(
    "round-robin fcfs always-admit"
    "least-loaded sjf always-admit"
    "weighted priority-fcfs token-bucket"
)

for i in "${!H13_POLICIES[@]}"; do
    read -r routing scheduler admission <<< "${H13_POLICIES[$i]}"
    tag="h13_${routing}_${scheduler}_${admission}"
    qwen_run $TIMEOUT_STANDARD "$RESULTS_DIR/${tag}_run1.txt" \
        --rate 1000 --num-requests 200 --seed 42 \
        --routing-policy "$routing" --scheduler "$scheduler" \
        --admission-policy "$admission" --log error
    qwen_run $TIMEOUT_STANDARD "$RESULTS_DIR/${tag}_run2.txt" \
        --rate 1000 --num-requests 200 --seed 42 \
        --routing-policy "$routing" --scheduler "$scheduler" \
        --admission-policy "$admission" --log error
done

# =============================================================================
# EXPERIMENT 3: H-Liveness — All schedulers satisfy liveness
# Claim: No starvation under admissible load for all schedulers
# Type: Deterministic (still_queued == 0 and completed == injected)
# =============================================================================
echo "=== EXPERIMENT: H-Liveness ===" >&2

LIVENESS_YAML="$RESULTS_DIR/liveness_workload.yaml"
cat > "$LIVENESS_YAML" <<'YAML'
version: "1"
seed: 42
category: language
aggregate_rate: 50.0
num_requests: 200
clients:
  - id: "uniform"
    rate_fraction: 1.0
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
YAML

for sched in fcfs sjf priority-fcfs; do
    qwen_run $TIMEOUT_STANDARD "$RESULTS_DIR/liveness_${sched}.txt" \
        --workload-spec "$LIVENESS_YAML" --seed 42 \
        --routing-policy least-loaded --scheduler "$sched" \
        --admission-policy always-admit --log error
done

# Constrained batch (--max-num-running-reqs 4) to make scheduler matter
LIVENESS_LOW_YAML="$RESULTS_DIR/liveness_low_workload.yaml"
cat > "$LIVENESS_LOW_YAML" <<'YAML'
version: "1"
seed: 42
category: language
aggregate_rate: 5.0
num_requests: 100
clients:
  - id: "uniform"
    rate_fraction: 1.0
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
YAML

for sched in fcfs sjf priority-fcfs; do
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/liveness_constrained_${sched}.txt" \
        $QWEN_COMMON --num-instances $INSTANCES \
        --max-num-running-reqs 4 \
        --workload-spec "$LIVENESS_LOW_YAML" --seed 42 \
        --routing-policy least-loaded --scheduler "$sched" \
        --admission-policy always-admit --log error
done

# =============================================================================
# EXPERIMENT 4: H-Overload — Conservation under 10x overload
# Claim: INV-1 holds, no panics
# Type: Deterministic (uses --rate CLI mode)
# =============================================================================
echo "=== EXPERIMENT: H-Overload ===" >&2

for rate in 300 750 1500; do
    for policy in "round-robin fcfs always-admit" "least-loaded fcfs always-admit" "round-robin fcfs token-bucket"; do
        read -r routing scheduler admission <<< "$policy"
        tag="overload_${rate}_${routing}_${admission}"
        # Must pass --stderr BEFORE $QWEN_COMMON so blis_run catches it
        blis_run $TIMEOUT_EXTENDED "$RESULTS_DIR/${tag}.txt" \
            --stderr "$RESULTS_DIR/${tag}_stderr.txt" \
            $QWEN_COMMON --num-instances $INSTANCES \
            --rate "$rate" --num-requests 2000 --seed 42 --horizon 5000000 \
            --routing-policy "$routing" --scheduler "$scheduler" \
            --admission-policy "$admission" --log error || true
    done
done

# =============================================================================
# EXPERIMENT 5: H-Phase — TTFT linear in input, decode linear in output
# Claim: R² > 0.95 for linear fits (deterministic with batch=1)
# Note: With alpha1=0 for Qwen, TTFT linearity in input comes from beta1
#       in StepTime, not from alpha-based QueueingTime.
# =============================================================================
echo "=== EXPERIMENT: H-Phase ===" >&2

for input_tokens in 64 128 256 512 1024; do
    PHASE_YAML="$RESULTS_DIR/phase_input_${input_tokens}.yaml"
    cat > "$PHASE_YAML" <<YAML
version: "1"
seed: 42
category: language
aggregate_rate: 0.001
num_requests: 10
clients:
  - id: "sweep"
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: $input_tokens
    output_distribution:
      type: constant
      params:
        value: 256
YAML
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/phase_input_${input_tokens}.txt" \
        $QWEN_COMMON --num-instances 1 \
        --max-num-running-reqs 1 --total-kv-blocks 1000000 \
        --workload-spec "$PHASE_YAML" --seed 42 \
        --results-path "$RESULTS_DIR/phase_input_${input_tokens}_results.json" \
        --log error
done

for output_tokens in 64 128 256 512 1024; do
    PHASE_YAML="$RESULTS_DIR/phase_output_${output_tokens}.yaml"
    cat > "$PHASE_YAML" <<YAML
version: "1"
seed: 42
category: language
aggregate_rate: 0.001
num_requests: 10
clients:
  - id: "sweep"
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 256
    output_distribution:
      type: constant
      params:
        value: $output_tokens
YAML
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/phase_output_${output_tokens}.txt" \
        $QWEN_COMMON --num-instances 1 \
        --max-num-running-reqs 1 --total-kv-blocks 1000000 \
        --workload-spec "$PHASE_YAML" --seed 42 \
        --results-path "$RESULTS_DIR/phase_output_${output_tokens}_results.json" \
        --log error
done

# =============================================================================
# EXPERIMENT 6: H-MMK — DES matches M/M/k at low utilization
# Claim: E2E monotonically increases with rho; within reasonable bounds at low rho
# Type: Statistical/Validation
# =============================================================================
echo "=== EXPERIMENT: H-MMK ===" >&2

# Calibration: batch=1, 1 instance, ultra-low rate
MMK_CAL_YAML="$RESULTS_DIR/mmk_calibrate.yaml"
cat > "$MMK_CAL_YAML" <<'YAML'
version: "1"
seed: 42
category: language
aggregate_rate: 0.01
num_requests: 10
clients:
  - id: "calibrate"
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 1
    output_distribution:
      type: exponential
      params:
        mean: 128
YAML

blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/mmk_calibrate.txt" \
    $QWEN_COMMON --num-instances 1 \
    --max-num-running-reqs 1 --total-kv-blocks 1000000 \
    --workload-spec "$MMK_CAL_YAML" --seed 42 \
    --results-path "$RESULTS_DIR/mmk_calibrate_results.json" \
    --log error

# Run at ρ = 0.1, 0.2, 0.3, 0.5 (4 instances, batch=1)
# Pre-computed mu estimate: service_time ≈ 7071us (prefill) + 128*7077us (decode) ≈ 912ms
# mu ≈ 1.095 req/s per server; rate = rho * k * mu = rho * 4 * 1.095
for rho in 0.1 0.2 0.3 0.5; do
    local_rate=$(python3 -c "print(round($rho * 4 * 1.095, 3))")
    for seed in 42 123 456; do
        MMK_YAML="$RESULTS_DIR/mmk_rho${rho}_seed${seed}.yaml"
        cat > "$MMK_YAML" <<YAML
version: "1"
seed: $seed
category: language
aggregate_rate: $local_rate
num_requests: 2000
clients:
  - id: "mmk"
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 1
    output_distribution:
      type: exponential
      params:
        mean: 128
YAML
        blis_run $TIMEOUT_EXTENDED "$RESULTS_DIR/mmk_rho${rho}_seed${seed}.txt" \
            $QWEN_COMMON --num-instances 4 \
            --max-num-running-reqs 1 --total-kv-blocks 1000000 \
            --workload-spec "$MMK_YAML" --seed "$seed" \
            --routing-policy least-loaded \
            --results-path "$RESULTS_DIR/mmk_rho${rho}_seed${seed}_results.json" \
            --log error || true
    done
done

# =============================================================================
# EXPERIMENT 7: Prefix-Affinity — Prefix-aware routing outperforms load-only
# Claim: Prefix-affinity TTFT < load-only TTFT for prefix-heavy workloads
# Type: Statistical/Dominance (3 seeds)
# =============================================================================
echo "=== EXPERIMENT: Prefix-Affinity ===" >&2

PREFIX_YAML="$RESULTS_DIR/prefix_workload.yaml"
cat > "$PREFIX_YAML" <<'YAML'
version: "1"
seed: 42
category: language
aggregate_rate: 200.0
num_requests: 500
clients:
  - id: "prefix-heavy"
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 50
        min: 64
        max: 512
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 30
        min: 16
        max: 256
    prefix_group: "shared-context"
    prefix_length: 512
YAML

for seed in 42 123 456; do
    qwen_run $TIMEOUT_STANDARD "$RESULTS_DIR/prefix_cache_seed${seed}.txt" \
        --workload-spec "$PREFIX_YAML" --seed "$seed" \
        --routing-policy weighted --routing-scorers "prefix-affinity:3,queue-depth:2" \
        --scheduler fcfs --admission-policy always-admit --log error

    qwen_run $TIMEOUT_STANDARD "$RESULTS_DIR/prefix_load_seed${seed}.txt" \
        --workload-spec "$PREFIX_YAML" --seed "$seed" \
        --routing-policy weighted --routing-scorers "queue-depth:1" \
        --scheduler fcfs --admission-policy always-admit --log error
done

# Round 2 control: high rate (2000) to confirm rate-dependent mechanism
PREFIX_HIGH_YAML="$RESULTS_DIR/prefix_high_workload.yaml"
cat > "$PREFIX_HIGH_YAML" <<'YAML'
version: "1"
seed: 42
category: language
aggregate_rate: 2000.0
num_requests: 500
clients:
  - id: "prefix-heavy"
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 50
        min: 64
        max: 512
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 30
        min: 16
        max: 256
    prefix_group: "shared-context"
    prefix_length: 512
YAML

for seed in 42 123 456; do
    qwen_run $TIMEOUT_STANDARD "$RESULTS_DIR/prefix_high_cache_seed${seed}.txt" \
        --workload-spec "$PREFIX_HIGH_YAML" --seed "$seed" \
        --routing-policy weighted --routing-scorers "prefix-affinity:3,queue-depth:2" \
        --scheduler fcfs --admission-policy always-admit --log error

    qwen_run $TIMEOUT_STANDARD "$RESULTS_DIR/prefix_high_load_seed${seed}.txt" \
        --workload-spec "$PREFIX_HIGH_YAML" --seed "$seed" \
        --routing-policy weighted --routing-scorers "queue-depth:1" \
        --scheduler fcfs --admission-policy always-admit --log error
done

# =============================================================================
# EXPERIMENT 8: H1-SJF — SJF reduces TTFT for short requests
# Claim: SJF TTFT < FCFS TTFT for short requests in bimodal workload
# Type: Statistical/Dominance (3 seeds)
# =============================================================================
echo "=== EXPERIMENT: H1-SJF ===" >&2

SJF_YAML="$RESULTS_DIR/sjf_workload.yaml"
cat > "$SJF_YAML" <<'YAML'
version: "1"
seed: 42
category: language
aggregate_rate: 3000.0
num_requests: 1000
clients:
  - id: "short"
    rate_fraction: 0.5
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
    rate_fraction: 0.5
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 1024
    output_distribution:
      type: constant
      params:
        value: 128
YAML

for seed in 42 123 456; do
    qwen_run $TIMEOUT_STANDARD "$RESULTS_DIR/sjf_fcfs_seed${seed}.txt" \
        --workload-spec "$SJF_YAML" --seed "$seed" \
        --routing-policy least-loaded --scheduler fcfs \
        --admission-policy always-admit \
        --results-path "$RESULTS_DIR/sjf_fcfs_seed${seed}_results.json" \
        --log error

    qwen_run $TIMEOUT_STANDARD "$RESULTS_DIR/sjf_sjf_seed${seed}.txt" \
        --workload-spec "$SJF_YAML" --seed "$seed" \
        --routing-policy least-loaded --scheduler sjf \
        --admission-policy always-admit \
        --results-path "$RESULTS_DIR/sjf_sjf_seed${seed}_results.json" \
        --log error
done

# =============================================================================
# EXPERIMENT 9: H3 — queue-depth distributes more evenly than kv-utilization
# Claim: queue-depth routing produces more uniform distribution at high rates
# Type: Statistical/Dominance (3 seeds, uses --rate CLI mode)
# =============================================================================
echo "=== EXPERIMENT: H3 ===" >&2

for seed in 42 123 456; do
    qwen_run $TIMEOUT_STANDARD "$RESULTS_DIR/h3_qd_seed${seed}.txt" \
        --rate 1000 --num-requests 500 --seed "$seed" \
        --routing-policy weighted --routing-scorers "queue-depth:1" \
        --scheduler fcfs --admission-policy always-admit --log error

    qwen_run $TIMEOUT_STANDARD "$RESULTS_DIR/h3_kv_seed${seed}.txt" \
        --rate 1000 --num-requests 500 --seed "$seed" \
        --routing-policy weighted --routing-scorers "kv-utilization:1" \
        --scheduler fcfs --admission-policy always-admit --log error
done

# =============================================================================
# EXPERIMENT 10: H8 — KV pressure monotonically increases preemptions
# Claim: Reducing KV blocks increases preemption frequency
# Type: Statistical/Monotonicity (3 seeds)
# =============================================================================
echo "=== EXPERIMENT: H8 ===" >&2

H8_YAML="$RESULTS_DIR/h8_workload.yaml"
cat > "$H8_YAML" <<'YAML'
version: "1"
seed: 42
category: language
aggregate_rate: 2000.0
num_requests: 200
clients:
  - id: "pressure"
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 512
        std_dev: 50
        min: 64
        max: 1024
    output_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 50
        min: 32
        max: 512
YAML

for blocks in 5000 3000 2200 2100 2000; do
    preflight_kv_check "$blocks" 16 1024
    for seed in 42 123 456; do
        blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/h8_blocks${blocks}_seed${seed}.txt" \
            $QWEN_COMMON --num-instances $INSTANCES \
            --total-kv-blocks "$blocks" \
            --workload-spec "$H8_YAML" --seed "$seed" \
            --routing-policy least-loaded --scheduler fcfs \
            --admission-policy always-admit --log error || true
    done
done

# =============================================================================
# EXPERIMENT 11: H9 — TTFT decreases monotonically with prefix length
# Claim: More prefix caching → lower TTFT
# Type: Statistical/Monotonicity (3 seeds)
# =============================================================================
echo "=== EXPERIMENT: H9 ===" >&2

for prefix in 0 64 128 256 512; do
    H9_YAML="$RESULTS_DIR/h9_prefix${prefix}.yaml"
    if [ "$prefix" -eq 0 ]; then
        # No prefix group for prefix=0
        cat > "$H9_YAML" <<'YAML'
version: "1"
seed: 42
category: language
aggregate_rate: 100.0
num_requests: 200
clients:
  - id: "prefix-sweep"
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 512
    output_distribution:
      type: constant
      params:
        value: 128
YAML
    else
        cat > "$H9_YAML" <<YAML
version: "1"
seed: 42
category: language
aggregate_rate: 100.0
num_requests: 200
clients:
  - id: "prefix-sweep"
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 512
    output_distribution:
      type: constant
      params:
        value: 128
    prefix_group: "shared"
    prefix_length: $prefix
YAML
    fi
    for seed in 42 123 456; do
        qwen_run $TIMEOUT_STANDARD "$RESULTS_DIR/h9_prefix${prefix}_seed${seed}.txt" \
            --workload-spec "$H9_YAML" --seed "$seed" \
            --routing-policy weighted --routing-scorers "prefix-affinity:3,queue-depth:2" \
            --scheduler fcfs --admission-policy always-admit --log error
    done
done

# Round 2 control: H9 isolation (single instance, batch=1, ultra-low rate)
# This eliminates batching and queueing effects to isolate prefix caching
for prefix in 0 256 512; do
    H9_CTRL_YAML="$RESULTS_DIR/h9_ctrl_prefix${prefix}.yaml"
    if [ "$prefix" -eq 0 ]; then
        cat > "$H9_CTRL_YAML" <<'YAML'
version: "1"
seed: 42
category: language
aggregate_rate: 0.001
num_requests: 10
clients:
  - id: "prefix-isolated"
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 512
    output_distribution:
      type: constant
      params:
        value: 128
YAML
    else
        cat > "$H9_CTRL_YAML" <<YAML
version: "1"
seed: 42
category: language
aggregate_rate: 0.001
num_requests: 10
clients:
  - id: "prefix-isolated"
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 512
    output_distribution:
      type: constant
      params:
        value: 128
    prefix_group: "shared"
    prefix_length: $prefix
YAML
    fi
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/h9_ctrl_prefix${prefix}.txt" \
        $QWEN_COMMON --num-instances 1 \
        --max-num-running-reqs 1 --total-kv-blocks 1000000 \
        --workload-spec "$H9_CTRL_YAML" --seed 42 \
        --routing-policy weighted --routing-scorers "prefix-affinity:3,queue-depth:2" \
        --log error
done

# =============================================================================
# EXPERIMENT 12: H10 — Tiered KV reduces preemptions
# Claim: CPU tier offloading halves preemption rate
# Type: Statistical/Dominance (3 seeds)
# =============================================================================
echo "=== EXPERIMENT: H10 ===" >&2

H10_YAML="$RESULTS_DIR/h10_workload.yaml"
cat > "$H10_YAML" <<'YAML'
version: "1"
seed: 42
category: language
aggregate_rate: 2000.0
num_requests: 200
clients:
  - id: "kv-pressure"
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 512
        std_dev: 50
        min: 64
        max: 1024
    output_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 50
        min: 32
        max: 512
YAML

for seed in 42 123 456; do
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/h10_single_seed${seed}.txt" \
        $QWEN_COMMON --num-instances $INSTANCES \
        --total-kv-blocks 2100 \
        --workload-spec "$H10_YAML" --seed "$seed" \
        --routing-policy least-loaded --scheduler fcfs \
        --admission-policy always-admit --log error || true

    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/h10_tiered_seed${seed}.txt" \
        $QWEN_COMMON --num-instances $INSTANCES \
        --total-kv-blocks 2100 --kv-cpu-blocks 700 \
        --kv-offload-threshold 0.9 \
        --workload-spec "$H10_YAML" --seed "$seed" \
        --routing-policy least-loaded --scheduler fcfs \
        --admission-policy always-admit --log error || true
done

# =============================================================================
# EXPERIMENT 13: H5 — Token-bucket admission reduces TTFT under bursts
# Claim: Token-bucket TTFT << no-admission TTFT under bursty arrivals
# Type: Statistical/Dominance (3 seeds)
# =============================================================================
echo "=== EXPERIMENT: H5 ===" >&2

H5_YAML="$RESULTS_DIR/h5_workload.yaml"
cat > "$H5_YAML" <<'YAML'
version: "1"
seed: 42
category: language
aggregate_rate: 2000.0
num_requests: 500
clients:
  - id: "bursty"
    rate_fraction: 1.0
    arrival:
      process: gamma
      cv: 3.5
    input_distribution:
      type: exponential
      params:
        mean: 512
    output_distribution:
      type: exponential
      params:
        mean: 256
YAML

for seed in 42 123 456; do
    qwen_run $TIMEOUT_STANDARD "$RESULTS_DIR/h5_noadmit_seed${seed}.txt" \
        --workload-spec "$H5_YAML" --seed "$seed" \
        --routing-policy least-loaded --scheduler fcfs \
        --admission-policy always-admit --log error

    qwen_run $TIMEOUT_STANDARD "$RESULTS_DIR/h5_bucket_seed${seed}.txt" \
        --workload-spec "$H5_YAML" --seed "$seed" \
        --routing-policy least-loaded --scheduler fcfs \
        --admission-policy token-bucket \
        --token-bucket-capacity 80000 --token-bucket-refill-rate 160000 \
        --log error
done

# =============================================================================
# EXPERIMENT 14: H14 — Pathological configs produce worse behavior
# Claim: always-busiest routing produces >2x worse TTFT
# Type: Statistical/Dominance (3 seeds)
# =============================================================================
echo "=== EXPERIMENT: H14 ===" >&2

H14_YAML="$RESULTS_DIR/h14_workload.yaml"
cat > "$H14_YAML" <<'YAML'
version: "1"
seed: 42
category: language
aggregate_rate: 2000.0
num_requests: 500
clients:
  - id: "realtime"
    rate_fraction: 0.333
    slo_class: "realtime"
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 30
        min: 16
        max: 256
    output_distribution:
      type: gaussian
      params:
        mean: 64
        std_dev: 20
        min: 8
        max: 128
  - id: "interactive"
    rate_fraction: 0.334
    slo_class: "interactive"
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 50
        min: 32
        max: 512
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 30
        min: 16
        max: 256
  - id: "batch"
    rate_fraction: 0.333
    slo_class: "batch"
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 512
        std_dev: 100
        min: 64
        max: 1024
    output_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 50
        min: 32
        max: 512
YAML

for seed in 42 123 456; do
    qwen_run $TIMEOUT_STANDARD "$RESULTS_DIR/h14_baseline_seed${seed}.txt" \
        --workload-spec "$H14_YAML" --seed "$seed" \
        --routing-policy least-loaded --scheduler fcfs \
        --admission-policy always-admit --log error

    qwen_run $TIMEOUT_STANDARD "$RESULTS_DIR/h14_patho_seed${seed}.txt" \
        --workload-spec "$H14_YAML" --seed "$seed" \
        --routing-policy always-busiest --scheduler fcfs \
        --admission-policy always-admit --log error
done

# =============================================================================
# EXPERIMENT 15: H-Arrival — Arrival generators (model-agnostic)
# Uses --rate CLI mode
# =============================================================================
echo "=== EXPERIMENT: H-Arrival ===" >&2

qwen_run $TIMEOUT_STANDARD "$RESULTS_DIR/arrival_poisson.txt" \
    --rate 100 --num-requests 1000 --seed 42 \
    --routing-policy round-robin --scheduler fcfs \
    --admission-policy always-admit \
    --results-path "$RESULTS_DIR/arrival_poisson_results.json" \
    --log error

# =============================================================================
# ANALYSIS
# =============================================================================
echo "=== Running analysis ===" >&2
python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR"

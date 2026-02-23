#!/bin/bash
# H19: Roofline vs Blackbox Mode — Policy Ranking Equivalence
#
# Tests whether roofline and blackbox latency modes produce the same
# relative TTFT ranking across routing policies, even though absolute
# latencies differ.
#
# Experiment 1 (Round 1): 3 routing policies x 2 latency modes x 3 seeds = 18 runs
#   Policies: round-robin, least-loaded, weighted
#   Modes: blackbox (alpha/beta regression), roofline (FLOPs/bandwidth)
#   Seeds: 42, 123, 456
#
# Experiment 2 (Round 2 — RCV-4 control): 3 policies x 3 seeds = 9 runs
#   Blackbox with alpha=0 and real beta. Isolates the effect of alpha overhead
#   on P99 ranking divergence. If alpha=0 blackbox P99 rankings match roofline,
#   then alpha overhead is confirmed as the mechanism.
#
# Note on alpha coefficients:
# The CLI only activates roofline mode when alpha AND beta are all zeros
# (it's a fallback path). So roofline mode inherently has alpha=[0,0,0],
# meaning QueueingTime=0 and OutputTokenProcessingTime=0. This changes the
# scheduling timeline (QueuedEvent delays, TTFT computation), which can
# perturb P99 tail rankings between policies.
#
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"
setup_experiment "${1:-}"

# Experiment parameters
INSTANCES=4
NUM_REQUESTS=200
RATE=500
SEEDS=(42 123 456)
POLICIES=("round-robin" "least-loaded" "weighted")

# Model config paths for roofline mode
MODEL_CONFIG_DIR="$REPO_ROOT/model_configs/llama-3.1-8b-instruct"
HW_CONFIG="$REPO_ROOT/hardware_config.json"

# Beta coefficients from defaults.yaml for llama-3.1-8b, H100, TP=2
# Used by the alpha=0 control (Experiment 2) to isolate alpha's effect.
BETA_COEFFS="6910.420479880494,17.67057489844186,2.8377471109943855"

# Use same KV blocks for both modes (defaults.yaml gives 132139 for blackbox;
# roofline would get the CLI default of 1000000). Explicit value eliminates confound.
KV_BLOCKS=132139

# Generate workload YAML (non-prefix, Gaussian in/out)
WORKLOAD_YAML="$RESULTS_DIR/workload.yaml"
cat > "$WORKLOAD_YAML" <<YAML
version: "1"
seed: 1
category: language
aggregate_rate: ${RATE}.0
num_requests: ${NUM_REQUESTS}

clients:
  - id: "simple-chat"
    tenant_id: "default"
    slo_class: "interactive"
    rate_fraction: 1.0
    streaming: false
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
        min: 32
        max: 256
YAML

echo "=== H19: Roofline vs Blackbox Mode ==="
echo "Instances: $INSTANCES, Requests: $NUM_REQUESTS, Rate: $RATE"
echo "Policies: ${POLICIES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo ""

run_count=0
fail_count=0

for seed in "${SEEDS[@]}"; do
  for policy in "${POLICIES[@]}"; do
    # --- Blackbox mode (default, picks up coefficients from defaults.yaml) ---
    outfile="$RESULTS_DIR/blackbox_${policy}_s${seed}.txt"
    echo "Running blackbox / ${policy} / seed=${seed}..." >&2
    blis_run "$TIMEOUT_STANDARD" "$outfile" \
      --model "$MODEL" \
      --num-instances "$INSTANCES" \
      --routing-policy "$policy" \
      --workload-spec "$WORKLOAD_YAML" \
      --seed "$seed" \
      --total-kv-blocks "$KV_BLOCKS" || true
    run_count=$((run_count + 1))
    if is_timeout "$outfile"; then
      fail_count=$((fail_count + 1))
    fi

    # --- Roofline mode (--model-config-folder triggers roofline fallback) ---
    outfile="$RESULTS_DIR/roofline_${policy}_s${seed}.txt"
    errfile="$RESULTS_DIR/roofline_${policy}_s${seed}_stderr.txt"
    echo "Running roofline / ${policy} / seed=${seed}..." >&2
    blis_run "$TIMEOUT_STANDARD" "$outfile" --stderr "$errfile" \
      --model "$MODEL" \
      --num-instances "$INSTANCES" \
      --routing-policy "$policy" \
      --workload-spec "$WORKLOAD_YAML" \
      --seed "$seed" \
      --model-config-folder "$MODEL_CONFIG_DIR" \
      --hardware-config "$HW_CONFIG" \
      --hardware "H100" \
      --tp 2 \
      --total-kv-blocks "$KV_BLOCKS" || true
    run_count=$((run_count + 1))
    if is_timeout "$outfile"; then
      fail_count=$((fail_count + 1))
    fi
  done
done

# === Experiment 2 (Round 2): Alpha=0 control ===
echo ""
echo "=== Experiment 2: Alpha=0 Control (RCV-4) ==="

for seed in "${SEEDS[@]}"; do
  for policy in "${POLICIES[@]}"; do
    # Blackbox mode with explicit alpha=0, real beta — isolates alpha effect
    outfile="$RESULTS_DIR/alpha0_${policy}_s${seed}.txt"
    echo "Running alpha0-blackbox / ${policy} / seed=${seed}..." >&2
    blis_run "$TIMEOUT_STANDARD" "$outfile" \
      --model "$MODEL" \
      --num-instances "$INSTANCES" \
      --routing-policy "$policy" \
      --workload-spec "$WORKLOAD_YAML" \
      --seed "$seed" \
      --total-kv-blocks "$KV_BLOCKS" \
      --alpha-coeffs "0,0,0" \
      --beta-coeffs "$BETA_COEFFS" || true
    run_count=$((run_count + 1))
    if is_timeout "$outfile"; then
      fail_count=$((fail_count + 1))
    fi
  done
done

echo ""
echo "Completed $run_count runs ($fail_count failures)"

# Run analyzer
echo ""
echo "=== Analysis ==="
python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR"

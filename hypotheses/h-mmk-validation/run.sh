#!/bin/bash
# H-MMK: Cross-Validate DES Against M/M/k Analytical Model
#
# Hypothesis: Under matching assumptions (Poisson arrivals, approximately
# exponential service times, k servers, FCFS), the DES queue length
# distribution and mean wait time should match M/M/k predictions within 5%.
#
# Classification: Statistical / Equivalence
# Family: Structural model
# VV&UQ: Validation
#
# Three sub-experiments:
#   1. M/M/1 (k=1): cleanest comparison, no routing complexity
#   2. k×M/M/1 (k=4, round-robin): tests Poisson splitting + cluster decomposition
#   3. M/M/k approximation (k=4, least-loaded): tests how close JSQ routing
#      gets to the theoretical M/M/k shared-queue model
#
# Each sub-experiment sweeps utilization rho = {0.3, 0.5, 0.7, 0.85}.
# Three seeds per operating point (42, 123, 456).
#
# Design notes:
#   ED-1: Controlled comparison — only routing policy varies between sub-exp 2 and 3
#   ED-2: Rate calibrated from empirical service time measurement (step 0)
#   ED-3: Preconditions — stability (rho < 1), sufficient requests for convergence
#   ED-5: Reproducible — builds binary, runs all variants, no manual steps
#   ED-6: No prior experiment reference (first M/M/k validation)
#
# Reference: https://github.com/inference-sim/inference-sim/issues/319
#
# Usage: ./run.sh [--rebuild]
#   --rebuild  Force rebuild of the binary
#
# Requires: Go 1.24+, Python 3 (scipy optional, falls back to manual KS test)

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
RHOS=(0.3 0.5 0.7 0.85)
NUM_REQUESTS=2000
OUTPUT_TOKEN_MEAN=128  # Exponential output → drives service time variability
INPUT_TOKENS=1         # Constant input=1 → prefill shift <1% of mean service time

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# Generate workload YAML with a given rate
make_workload() {
    local rate=$1
    local seed=$2
    local outfile=$3

    cat > "$outfile" << YAMLEOF
version: "1"
seed: $seed
category: language
aggregate_rate: $rate
num_requests: $NUM_REQUESTS
clients:
  - id: "mmk-client"
    tenant_id: "default"
    slo_class: "batch"
    rate_fraction: 1.0
    streaming: false
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: $INPUT_TOKENS
    output_distribution:
      type: exponential
      params:
        mean: $OUTPUT_TOKEN_MEAN
YAMLEOF
}

echo "============================================================================"
echo "  H-MMK: Cross-Validate DES Against M/M/k Analytical Model"
echo "  Reference: issue #319, docs/standards/experiments.md (cross-validation)"
echo "  Type: Statistical / Equivalence (within 5%)"
echo "  Family: Structural model | VV&UQ: Validation"
echo "============================================================================"
echo ""

# ── Step 0: Calibrate mean service time ──────────────────────────────────────
# Run at very low load (no queueing) with CONSTANT output to measure
# service time precisely. Uses small request count (10) and constant
# output tokens to avoid variance and queueing contamination.

echo "Step 0: Calibrating mean service time..."
echo "  (Constant output=$OUTPUT_TOKEN_MEAN, input=$INPUT_TOKENS, very low load)"

cat > "$RESULTS_DIR/cal_wl.yaml" << YAMLEOF
version: "1"
seed: 42
category: language
aggregate_rate: 0.01
num_requests: 10
clients:
  - id: "calibrate"
    tenant_id: "default"
    slo_class: "batch"
    rate_fraction: 1.0
    streaming: false
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: $INPUT_TOKENS
    output_distribution:
      type: constant
      params:
        value: $OUTPUT_TOKEN_MEAN
YAMLEOF

timeout 120 "$BINARY" run \
    --model "$MODEL" \
    --num-instances 1 \
    --max-num-running-reqs 1 \
    --workload-spec "$RESULTS_DIR/cal_wl.yaml" \
    --seed 42 \
    --scheduler fcfs \
    --admission-policy always-admit \
    --total-kv-blocks 1000000 \
    --log error \
    --results-path "$RESULTS_DIR/calibration.json" \
    2>/dev/null \
    > "$RESULTS_DIR/calibration_stdout.txt"

# Extract mean service time from calibration (E2E at zero load ≈ service time)
# Uses constant output, so all requests have identical service time.
MEAN_SERVICE_MS=$(python3 -c "
import json
data = json.load(open('$RESULTS_DIR/calibration.json'))
reqs = [r for r in data['requests'] if r['e2e_ms'] > 0]
mean_e2e = sum(r['e2e_ms'] for r in reqs) / len(reqs)
print(f'{mean_e2e:.3f}')
")
echo "  Mean service time (constant output=$OUTPUT_TOKEN_MEAN): ${MEAN_SERVICE_MS} ms"

# Compute mu (service rate per server in req/s)
MU=$(python3 -c "print(f'{1000.0 / $MEAN_SERVICE_MS:.6f}')")
echo "  Service rate mu = 1/E[S] = ${MU} req/s"
echo ""

# Pre-compute arrival rates for each rho
echo "  Target utilizations and arrival rates:"
for rho in "${RHOS[@]}"; do
    # For k=1: lambda = rho * mu
    RATE_K1=$(python3 -c "print(f'{$rho * $MU:.4f}')")
    # For k=4: lambda = rho * k * mu
    RATE_K4=$(python3 -c "print(f'{$rho * 4 * $MU:.4f}')")
    echo "    rho=$rho: k=1 rate=${RATE_K1}, k=4 rate=${RATE_K4} req/s"
done
echo ""

# ── Sub-experiment 1: M/M/1 (k=1) ──────────────────────────────────────────
echo "============================================================================"
echo "  Sub-experiment 1: M/M/1 (k=1, single instance)"
echo "  Comparing DES wait times against M/M/1 analytical model"
echo "============================================================================"
echo ""

for rho in "${RHOS[@]}"; do
    RATE=$(python3 -c "print(f'{$rho * $MU:.6f}')")
    for seed in "${SEEDS[@]}"; do
        echo "  Running: rho=$rho rate=$RATE seed=$seed k=1 ..."
        make_workload "$RATE" "$seed" "$RESULTS_DIR/wl_mm1_r${rho}_s${seed}.yaml"
        timeout 300 "$BINARY" run \
            --model "$MODEL" \
            --num-instances 1 \
            --max-num-running-reqs 1 \
            --workload-spec "$RESULTS_DIR/wl_mm1_r${rho}_s${seed}.yaml" \
            --seed "$seed" \
            --scheduler fcfs \
            --admission-policy always-admit \
            --total-kv-blocks 1000000 \
            --log error \
            --results-path "$RESULTS_DIR/mm1_r${rho}_s${seed}.json" \
            2>/dev/null \
            > "$RESULTS_DIR/mm1_r${rho}_s${seed}_stdout.txt" \
            || echo "    WARNING: timeout or error for rho=$rho seed=$seed"
    done
done

echo ""

# ── Sub-experiment 2: k×M/M/1 (k=4, round-robin) ───────────────────────────
echo "============================================================================"
echo "  Sub-experiment 2: k × M/M/1 (k=4, round-robin routing)"
echo "  Poisson splitting: each instance is M/M/1 with rate lambda/4"
echo "============================================================================"
echo ""

for rho in "${RHOS[@]}"; do
    RATE=$(python3 -c "print(f'{$rho * 4 * $MU:.6f}')")
    for seed in "${SEEDS[@]}"; do
        echo "  Running: rho=$rho rate=$RATE seed=$seed k=4 round-robin ..."
        make_workload "$RATE" "$seed" "$RESULTS_DIR/wl_kxmm1_r${rho}_s${seed}.yaml"
        timeout 300 "$BINARY" run \
            --model "$MODEL" \
            --num-instances 4 \
            --max-num-running-reqs 1 \
            --workload-spec "$RESULTS_DIR/wl_kxmm1_r${rho}_s${seed}.yaml" \
            --seed "$seed" \
            --routing-policy round-robin \
            --scheduler fcfs \
            --admission-policy always-admit \
            --total-kv-blocks 1000000 \
            --log error \
            --results-path "$RESULTS_DIR/kxmm1_r${rho}_s${seed}.json" \
            2>/dev/null \
            > "$RESULTS_DIR/kxmm1_r${rho}_s${seed}_stdout.txt" \
            || echo "    WARNING: timeout or error for rho=$rho seed=$seed"
    done
done

echo ""

# ── Sub-experiment 3: M/M/k approximation (k=4, least-loaded) ───────────────
echo "============================================================================"
echo "  Sub-experiment 3: M/M/k approximation (k=4, least-loaded routing)"
echo "  Least-loaded ≈ join-shortest-queue → approximates M/M/k shared queue"
echo "============================================================================"
echo ""

for rho in "${RHOS[@]}"; do
    RATE=$(python3 -c "print(f'{$rho * 4 * $MU:.6f}')")
    for seed in "${SEEDS[@]}"; do
        echo "  Running: rho=$rho rate=$RATE seed=$seed k=4 least-loaded ..."
        make_workload "$RATE" "$seed" "$RESULTS_DIR/wl_mmk_r${rho}_s${seed}.yaml"
        timeout 300 "$BINARY" run \
            --model "$MODEL" \
            --num-instances 4 \
            --max-num-running-reqs 1 \
            --workload-spec "$RESULTS_DIR/wl_mmk_r${rho}_s${seed}.yaml" \
            --seed "$seed" \
            --routing-policy least-loaded \
            --scheduler fcfs \
            --admission-policy always-admit \
            --total-kv-blocks 1000000 \
            --log error \
            --results-path "$RESULTS_DIR/mmk_r${rho}_s${seed}.json" \
            2>/dev/null \
            > "$RESULTS_DIR/mmk_r${rho}_s${seed}_stdout.txt" \
            || echo "    WARNING: timeout or error for rho=$rho seed=$seed"
    done
done

echo ""
echo "============================================================================"
echo "  Analysis"
echo "============================================================================"
echo ""

python3 "$SCRIPT_DIR/analyze.py" \
    --results-dir "$RESULTS_DIR" \
    --mu "$MU" \
    --mean-service-ms "$MEAN_SERVICE_MS" \
    --output-token-mean "$OUTPUT_TOKEN_MEAN" \
    --num-requests "$NUM_REQUESTS"

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis"
echo "============================================================================"

#!/bin/bash
# H-Phase-Structure: TTFT Linearity in Input Tokens, Decode Linearity in Output Tokens
#
# Hypothesis: Prefill cost is proportional to prompt token count and decode cost
# is proportional to generated token count. TTFT should be linear in input_tokens
# (R² > 0.95) with output held constant, and (E2E − TTFT) should be linear in
# output_tokens (R² > 0.95) with input held constant.
#
# Classification: Statistical / Monotonicity (linearity = constant slope)
# Family: Structural model
# VV&UQ: Validation
#
# Two sub-experiments:
#   A. Fix output=128, vary input ∈ {64, 128, 256, 512, 1024} → TTFT vs input_tokens
#   B. Fix input=256, vary output ∈ {64, 128, 256, 512, 1024} → (E2E−TTFT) vs output_tokens
#
# Design notes:
#   ED-1: Controlled comparison — only one token dimension varies per sub-experiment
#   ED-2: Low rate (0.01 req/s) eliminates queueing → isolates latency model
#   ED-3: Precondition — max-num-running-reqs=1, rate << service rate → zero queueing
#   ED-5: Reproducible — builds binary, runs all variants, no manual steps
#   ED-6: Reference: hypotheses/h-mmk-validation/run.sh (calibration approach, beta coefficients)
#         Config diff: this experiment uses single instance, constant distributions on both
#         input and output (H-MMK used constant input only + exponential output).
#         Rate set to 0.01 req/s (same as H-MMK calibration step).
#
# Reference: https://github.com/inference-sim/inference-sim/issues/314
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
RATE=0.01  # Very low rate → zero queueing (P(queue) ≈ 1% at longest service time)
NUM_REQUESTS=20  # Per level. Constant distributions → all requests identical within level.

# Token levels for sweeps
INPUT_LEVELS=(64 128 256 512 1024)
OUTPUT_LEVELS=(64 128 256 512 1024)

# Fixed dimensions
FIXED_OUTPUT=128   # Held constant in Experiment A
FIXED_INPUT=256    # Held constant in Experiment B

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# Generate workload YAML with constant input and output distributions
make_workload() {
    local input_tokens=$1
    local output_tokens=$2
    local seed=$3
    local outfile=$4

    cat > "$outfile" << YAMLEOF
version: "1"
seed: $seed
category: language
aggregate_rate: $RATE
num_requests: $NUM_REQUESTS
clients:
  - id: "phase-structure"
    tenant_id: "default"
    slo_class: "batch"
    rate_fraction: 1.0
    streaming: false
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: $input_tokens
    output_distribution:
      type: constant
      params:
        value: $output_tokens
YAMLEOF
}

run_sim() {
    local results_json=$1
    local stdout_file=$2
    local workload_yaml=$3
    local seed=$4

    timeout 300 "$BINARY" run \
        --model "$MODEL" \
        --num-instances 1 \
        --max-num-running-reqs 1 \
        --workload-spec "$workload_yaml" \
        --seed "$seed" \
        --scheduler fcfs \
        --admission-policy always-admit \
        --total-kv-blocks 1000000 \
        --log error \
        --results-path "$results_json" \
        2>/dev/null \
        > "$stdout_file" \
        || echo "    WARNING: timeout or error"
}

echo "============================================================================"
echo "  H-Phase-Structure: Latency Model Phase Linearity Validation"
echo "  Reference: issue #314, docs/standards/experiments.md (phase structure)"
echo "  Type: Statistical / Monotonicity (R² > 0.95)"
echo "  Family: Structural model | VV&UQ: Validation"
echo "============================================================================"
echo ""
echo "  Config: single instance, max-num-running-reqs=1, rate=${RATE} req/s"
echo "  Requests per level: ${NUM_REQUESTS} (constant distributions → identical)"
echo "  Seeds: ${SEEDS[*]}"
echo ""

# ── Step 0: Calibration sanity check ────────────────────────────────────────
# Run one request at a known config to verify beta coefficient behavior.
# Expected: E2E for (input=1, output=128) ≈ 1126 ms (from H-MMK calibration)

echo "Step 0: Calibration sanity check..."
echo "  (Verifying against H-MMK: input=1, output=128 → E2E ≈ 1126 ms)"

make_workload 1 128 42 "$RESULTS_DIR/cal_wl.yaml"
# Override for calibration: only 10 requests at very low rate
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
        value: 1
    output_distribution:
      type: constant
      params:
        value: 128
YAMLEOF

run_sim "$RESULTS_DIR/cal.json" "$RESULTS_DIR/cal_stdout.txt" "$RESULTS_DIR/cal_wl.yaml" 42

CAL_E2E=$(python3 -c "
import json
data = json.load(open('$RESULTS_DIR/cal.json'))
reqs = [r for r in data['requests'] if r['e2e_ms'] > 0]
mean_e2e = sum(r['e2e_ms'] for r in reqs) / len(reqs)
print(f'{mean_e2e:.3f}')
")
echo "  Measured E2E (input=1, output=128): ${CAL_E2E} ms"
echo "  H-MMK reference: 1126.241 ms"
echo ""

# ── Experiment A: TTFT vs Input Tokens ──────────────────────────────────────
# Fixed output=128, vary input ∈ {64, 128, 256, 512, 1024}

echo "============================================================================"
echo "  Experiment A: TTFT vs Input Tokens"
echo "  Fixed output=${FIXED_OUTPUT}, varying input ∈ {${INPUT_LEVELS[*]}}"
echo "============================================================================"
echo ""

for input in "${INPUT_LEVELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "  Running: input=${input} output=${FIXED_OUTPUT} seed=${seed} ..."
        wl="$RESULTS_DIR/expA_in${input}_s${seed}_wl.yaml"
        make_workload "$input" "$FIXED_OUTPUT" "$seed" "$wl"
        run_sim \
            "$RESULTS_DIR/expA_in${input}_s${seed}.json" \
            "$RESULTS_DIR/expA_in${input}_s${seed}_stdout.txt" \
            "$wl" "$seed"
    done
done

echo ""

# ── Experiment B: (E2E − TTFT) vs Output Tokens ────────────────────────────
# Fixed input=256, vary output ∈ {64, 128, 256, 512, 1024}

echo "============================================================================"
echo "  Experiment B: Decode Time (E2E − TTFT) vs Output Tokens"
echo "  Fixed input=${FIXED_INPUT}, varying output ∈ {${OUTPUT_LEVELS[*]}}"
echo "============================================================================"
echo ""

for output in "${OUTPUT_LEVELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "  Running: input=${FIXED_INPUT} output=${output} seed=${seed} ..."
        wl="$RESULTS_DIR/expB_out${output}_s${seed}_wl.yaml"
        make_workload "$FIXED_INPUT" "$output" "$seed" "$wl"
        run_sim \
            "$RESULTS_DIR/expB_out${output}_s${seed}.json" \
            "$RESULTS_DIR/expB_out${output}_s${seed}_stdout.txt" \
            "$wl" "$seed"
    done
done

echo ""
echo "============================================================================"
echo "  Analysis"
echo "============================================================================"
echo ""

python3 "$SCRIPT_DIR/analyze.py" \
    --results-dir "$RESULTS_DIR" \
    --input-levels "${INPUT_LEVELS[*]}" \
    --output-levels "${OUTPUT_LEVELS[*]}" \
    --fixed-output "$FIXED_OUTPUT" \
    --fixed-input "$FIXED_INPUT" \
    --seeds "${SEEDS[*]}"

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis"
echo "============================================================================"

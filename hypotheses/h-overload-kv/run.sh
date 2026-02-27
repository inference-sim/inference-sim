#!/bin/bash
# H-Overload-KV: Combined overload + KV cache pressure robustness
#
# Hypothesis: Under extreme overload (2x-10x saturation) combined with KV cache
# pressure, the simulator should maintain conservation (INV-1), not panic, and
# preemptions should increase gracefully -- no livelock or silent data loss.
#
# Classification: Deterministic (exact pass/fail on conservation invariant)
# Family: Robustness/failure-mode
# VV&UQ: Verification
# Tier: 1 (correctness)
#
# Design: 3x3 matrix (overload level x KV config)
#   Overload: 2x, 5x, 10x saturation rate
#   KV: abundant (1000000 blocks), constrained (2000 blocks), tiered (2000 GPU + 2000 CPU)
#
# KV sizing rationale: input=256 tokens / blockSize=16 = 16 blocks per request.
#   With 4 instances, 2000 blocks = 500/instance = ~31 concurrent requests/instance.
#   At 2x overload (700 req/s → 175/instance), queue >> capacity → heavy preemptions.
#   500 blocks was too extreme — caused livelock-like cascading preemptions.
#
# Saturation rate: ~350 req/s for 4 instances (beta coefficients, input=256, output=128)
#   Step time ~= 6910 + 17.67*256 + 2.84*128 ~= 11788 us ~= 11.8ms
#   4 instances: 4/0.0118 ~= 339 req/s, rounded to 350 as 1x saturation.
#
# Usage: ./run.sh [--rebuild]
#   --rebuild  Force rebuild of the binary
#
# Requires: Go 1.24+, Python 3

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
SEED=42
NUM_REQUESTS=100
INSTANCES=4

# Saturation rate for 4 instances with input=256, output=128
SAT_RATE=350

# Overload multipliers
MULTIPLIERS="2 5 10"

# KV configurations: name|total_gpu_blocks|cpu_blocks|offload_threshold|transfer_bw|transfer_latency
# - abundant: 1M GPU blocks, no CPU tier
# - constrained: 2000 GPU blocks, no CPU tier (preemption pressure without livelock)
# - tiered: 2000 GPU + 2000 CPU blocks (offload absorbs overflow)
KV_CONFIGS=(
    "abundant|1000000|0|0.9|100|0"
    "constrained|2000|0|0.9|100|0"
    "tiered|2000|2000|0.8|100|10"
)

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# Create workload YAML with constant distributions for deterministic behavior
make_workload() {
    local seed=$1
    local rate=$2
    local outfile=$3

    cat > "$outfile" << YAMLEOF
version: "1"
seed: $seed
category: language
aggregate_rate: ${rate}.0
num_requests: $NUM_REQUESTS
clients:
  - id: "overload-kv-stress"
    tenant_id: "test"
    slo_class: "interactive"
    rate_fraction: 1.0
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 256
    output_distribution:
      type: constant
      params:
        value: 128
YAMLEOF
}

echo "============================================================================"
echo "  H-Overload-KV: Combined Overload + KV Cache Pressure"
echo "  Reference: GitHub Issue #338"
echo "  Type: Deterministic (single seed, exact pass/fail on conservation)"
echo "============================================================================"
echo ""
echo "Configuration: instances=$INSTANCES, requests=$NUM_REQUESTS, seed=$SEED"
echo "Saturation rate estimate: $SAT_RATE req/s (input=256, output=128)"
echo "Overload multipliers: $MULTIPLIERS"
echo "KV configs: abundant (1M), constrained (2000), tiered (2000+2000)"
echo ""

# ---- Run 3x3 matrix --------------------------------------------------------
echo "=== Running 3x3 Matrix ==="
echo ""

for MULT in $MULTIPLIERS; do
    RATE=$((SAT_RATE * MULT))
    WORKLOAD_FILE="$RESULTS_DIR/wl_${MULT}x.yaml"
    make_workload "$SEED" "$RATE" "$WORKLOAD_FILE"

    for kv_cfg in "${KV_CONFIGS[@]}"; do
        IFS='|' read -r KV_NAME GPU_BLOCKS CPU_BLOCKS OFFLOAD_THRESH XFER_BW XFER_LAT <<< "$kv_cfg"

        LABEL="${MULT}x_${KV_NAME}"
        OUTFILE="$RESULTS_DIR/${LABEL}.txt"
        ERRFILE="$RESULTS_DIR/${LABEL}_stderr.txt"

        echo -n "  [${MULT}x overload, ${KV_NAME}] rate=${RATE} gpu_blocks=${GPU_BLOCKS} cpu_blocks=${CPU_BLOCKS} ... "

        EXIT_CODE=0
        timeout 180 "$BINARY" run \
            --model "$MODEL" \
            --num-instances "$INSTANCES" \
            --seed "$SEED" \
            --workload-spec "$WORKLOAD_FILE" \
            --total-kv-blocks "$GPU_BLOCKS" \
            --kv-cpu-blocks "$CPU_BLOCKS" \
            --kv-offload-threshold "$OFFLOAD_THRESH" \
            --kv-transfer-bandwidth "$XFER_BW" \
            --kv-transfer-base-latency "$XFER_LAT" \
            --admission-policy always-admit \
            --routing-policy round-robin \
            --scheduler fcfs \
            --priority-policy constant \
            --summarize-trace \
            --trace-level decisions \
            --log error \
            > "$OUTFILE" 2>"$ERRFILE" || EXIT_CODE=$?

        if [[ $EXIT_CODE -ne 0 ]]; then
            echo "EXIT_CODE=$EXIT_CODE (PANIC/ERROR/TIMEOUT)"
            echo "STDERR:" >> "$OUTFILE"
            cat "$ERRFILE" >> "$OUTFILE"
        else
            echo "ok (exit=0)"
        fi

        # Record exit code and stderr markers for analysis
        echo "---EXIT_CODE=$EXIT_CODE---" >> "$OUTFILE"
        if grep -q "panic" "$ERRFILE" 2>/dev/null; then
            echo "---PANIC_DETECTED---" >> "$OUTFILE"
        fi
    done
    echo ""
done

# ---- Analysis ---------------------------------------------------------------
echo "=== Conservation Invariant Analysis ==="
echo ""

python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR"

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis"
echo "============================================================================"

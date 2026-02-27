#!/bin/bash
# H25: Integration Stress Test
# Verify conservation, determinism, and no-panic under the full policy stack
# Three configurations:
#   Config A: token-bucket admission (high rejection — tests admission pipeline conservation)
#   Config B: always-admit (all requests flow through — tests full pipeline under load)
#   Config C: always-admit + severely constrained KV (forces preemptions — stress path)
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$REPO_ROOT/blis"

# Build if needed
if [[ "${1:-}" == "--rebuild" ]] || [[ ! -x "$BINARY" ]]; then
    echo "Building blis..." >&2
    (cd "$REPO_ROOT" && go build -o blis main.go)
fi

MODEL="meta-llama/llama-3.1-8b-instruct"

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# Generate high-rate workload YAML inline (CLI --rate does NOT override YAML aggregate_rate)
WORKLOAD_YAML="$RESULTS_DIR/stress-workload.yaml"
cat > "$WORKLOAD_YAML" <<'YAML'
version: "1"
seed: 42
category: reasoning
aggregate_rate: 2000.0
num_requests: 500

clients:
  - id: "multi-turn-chat"
    tenant_id: "chat-users"
    slo_class: "interactive"
    rate_fraction: 1.0
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 30
        min: 32
        max: 512
    output_distribution:
      type: gaussian
      params:
        mean: 64
        std_dev: 20
        min: 16
        max: 256
    reasoning:
      reason_ratio_distribution:
        type: gaussian
        params:
          mean: 0
          std_dev: 0
          min: 0
          max: 0
      multi_turn:
        max_rounds: 5
        think_time_us: 500000
        context_growth: accumulate
YAML

echo "================================================================" >&2
echo "H25: Integration Stress Test — Full Policy Stack" >&2
echo "================================================================" >&2

# =====================================================================
# Config A: Token-bucket admission (as specified in hypothesis)
# Expected: high rejection rate, conservation must hold across pipeline
# =====================================================================
CONFIGA_FLAGS=(
    --model "$MODEL"
    --num-instances 4
    --seed 42
    --log error
    --routing-policy weighted
    --routing-scorers "prefix-affinity:3,queue-depth:2,kv-utilization:2"
    --admission-policy token-bucket
    --token-bucket-capacity 500
    --token-bucket-refill-rate 300
    --priority-policy slo-based
    --scheduler priority-fcfs
    --kv-cpu-blocks 200
    --kv-offload-threshold 0.8
    --kv-transfer-bandwidth 50
    --trace-level decisions
    --counterfactual-k 3
    --summarize-trace
    --workload-spec "$WORKLOAD_YAML"
)

echo "" >&2
echo "--- Config A: Token-bucket admission (cap=500, refill=300) ---" >&2

A_RUN1="$RESULTS_DIR/a_run1.txt"
if "$BINARY" run "${CONFIGA_FLAGS[@]}" > "$A_RUN1" 2>/dev/null; then
    echo "  Exit code: 0 (no panic)" >&2
else
    echo "  EXIT CODE: $? (PANIC or ERROR)" >&2
    cat "$A_RUN1" >&2
    exit 1
fi

A_RUN2="$RESULTS_DIR/a_run2.txt"
"$BINARY" run "${CONFIGA_FLAGS[@]}" > "$A_RUN2" 2>/dev/null

if diff -q "$A_RUN1" "$A_RUN2" > /dev/null 2>&1; then
    echo "  Determinism: PASS (byte-identical)" >&2
    A_DETERM="PASS"
else
    echo "  Determinism: FAIL (outputs differ)" >&2
    diff "$A_RUN1" "$A_RUN2" >&2 || true
    A_DETERM="FAIL"
fi

# =====================================================================
# Config B: Always-admit (all requests flow through the full pipeline)
# Same workload, routing, scheduler, KV, tracing — only admission differs
# =====================================================================
CONFIGB_FLAGS=(
    --model "$MODEL"
    --num-instances 4
    --seed 42
    --log error
    --routing-policy weighted
    --routing-scorers "prefix-affinity:3,queue-depth:2,kv-utilization:2"
    --admission-policy always-admit
    --priority-policy slo-based
    --scheduler priority-fcfs
    --kv-cpu-blocks 200
    --kv-offload-threshold 0.8
    --kv-transfer-bandwidth 50
    --trace-level decisions
    --counterfactual-k 3
    --summarize-trace
    --workload-spec "$WORKLOAD_YAML"
)

echo "" >&2
echo "--- Config B: Always-admit (full pipeline stress) ---" >&2

B_RUN1="$RESULTS_DIR/b_run1.txt"
if "$BINARY" run "${CONFIGB_FLAGS[@]}" > "$B_RUN1" 2>/dev/null; then
    echo "  Exit code: 0 (no panic)" >&2
else
    echo "  EXIT CODE: $? (PANIC or ERROR)" >&2
    cat "$B_RUN1" >&2
    exit 1
fi

B_RUN2="$RESULTS_DIR/b_run2.txt"
"$BINARY" run "${CONFIGB_FLAGS[@]}" > "$B_RUN2" 2>/dev/null

if diff -q "$B_RUN1" "$B_RUN2" > /dev/null 2>&1; then
    echo "  Determinism: PASS (byte-identical)" >&2
    B_DETERM="PASS"
else
    echo "  Determinism: FAIL (outputs differ)" >&2
    diff "$B_RUN1" "$B_RUN2" >&2 || true
    B_DETERM="FAIL"
fi

# Per-request data for Config B (the interesting one with all requests flowing)
B_RESULTS_JSON="$RESULTS_DIR/b_results.json"
"$BINARY" run "${CONFIGB_FLAGS[@]}" --results-path "$B_RESULTS_JSON" > /dev/null 2>/dev/null

# =====================================================================
# Config C: Always-admit + constrained KV (forces preemptions)
# Same as Config B but with --total-kv-blocks 800 --block-size-in-tokens 16
# This forces KV eviction/preemption with 500 requests (each ~8 blocks for
# mean input 128 tokens; 125 req/instance * 8 = 1000 blocks needed, only 800
# available per instance, so preemptions are guaranteed — ~70 events).
# Note: --total-kv-blocks <=500 causes impractical preemption cascades (>5 min).
# Round 2 addition per Reviewer B feedback.
# =====================================================================
CONFIGC_FLAGS=(
    --model "$MODEL"
    --num-instances 4
    --seed 42
    --log error
    --routing-policy weighted
    --routing-scorers "prefix-affinity:3,queue-depth:2,kv-utilization:2"
    --admission-policy always-admit
    --priority-policy slo-based
    --scheduler priority-fcfs
    --total-kv-blocks 800
    --block-size-in-tokens 16
    --kv-cpu-blocks 200
    --kv-offload-threshold 0.8
    --kv-transfer-bandwidth 50
    --trace-level decisions
    --counterfactual-k 3
    --summarize-trace
    --workload-spec "$WORKLOAD_YAML"
)

echo "" >&2
echo "--- Config C: Always-admit + constrained KV (total-kv-blocks=800) ---" >&2

C_RUN1="$RESULTS_DIR/c_run1.txt"
C_STDERR1="$RESULTS_DIR/c_stderr1.txt"
EXIT_CODE=0
if timeout 300 "$BINARY" run "${CONFIGC_FLAGS[@]}" > "$C_RUN1" 2>"$C_STDERR1"; then
    echo "  Exit code: 0 (no panic)" >&2
else
    EXIT_CODE=$?
    echo "  EXIT CODE: $EXIT_CODE (PANIC or ERROR or TIMEOUT)" >&2
    echo "  STDERR:" >&2
    tail -20 "$C_STDERR1" >&2
    # Don't exit — we still want to analyze whatever output we got
fi

C_RUN2="$RESULTS_DIR/c_run2.txt"
C_STDERR2="$RESULTS_DIR/c_stderr2.txt"
timeout 300 "$BINARY" run "${CONFIGC_FLAGS[@]}" > "$C_RUN2" 2>"$C_STDERR2" || true

if diff -q "$C_RUN1" "$C_RUN2" > /dev/null 2>&1; then
    echo "  Determinism: PASS (byte-identical)" >&2
    C_DETERM="PASS"
else
    echo "  Determinism: FAIL (outputs differ)" >&2
    diff "$C_RUN1" "$C_RUN2" >&2 || true
    C_DETERM="FAIL"
fi

# Per-request data for Config C
C_RESULTS_JSON="$RESULTS_DIR/c_results.json"
timeout 300 "$BINARY" run "${CONFIGC_FLAGS[@]}" --results-path "$C_RESULTS_JSON" > /dev/null 2>/dev/null || true

# =====================================================================
# Analyze all configurations
# =====================================================================
echo "" >&2
echo "--- Running analysis ---" >&2

echo "=== CONFIG A: TOKEN-BUCKET ===" > "$RESULTS_DIR/combined_output.txt"
python3 "$SCRIPT_DIR/analyze.py" \
    --run1 "$A_RUN1" \
    --run2 "$A_RUN2" \
    --determinism "$A_DETERM" \
    --num-requests 500 \
    --config-label "Config A: Token-Bucket (cap=500, refill=300/s)"

echo ""

echo "=== CONFIG B: ALWAYS-ADMIT ==="
python3 "$SCRIPT_DIR/analyze.py" \
    --run1 "$B_RUN1" \
    --run2 "$B_RUN2" \
    --results-json "$B_RESULTS_JSON" \
    --determinism "$B_DETERM" \
    --num-requests 500 \
    --config-label "Config B: Always-Admit (full pipeline stress)"

echo ""

echo "=== CONFIG C: CONSTRAINED KV ==="
python3 "$SCRIPT_DIR/analyze.py" \
    --run1 "$C_RUN1" \
    --run2 "$C_RUN2" \
    --results-json "$C_RESULTS_JSON" \
    --determinism "$C_DETERM" \
    --num-requests 500 \
    --config-label "Config C: Constrained KV (total-kv-blocks=800, stress path)"

#!/bin/bash
# H5: Token-Bucket Admission Control Under Burst
#
# Hypothesis: During traffic bursts (Gamma arrivals with high CV), a token
# bucket that rejects excess requests should cap queue depth, trading some
# rejected requests for much better tail latency for admitted ones.
#
# Classification: Statistical (Type 2) / Dominance.
# Token-bucket TTFT p99 < always-admit TTFT p99 across all seeds.
#
# Design:
#   - ED-1: Vary exactly one dimension (admission policy)
#   - ED-2: Rate-aware — test at high rate where bursts cause queue buildup
#   - ED-3: Precondition — verify Gamma arrivals produce visible burstiness
#   - ED-4: 3 seeds (42, 123, 456) per configuration
#   - ED-5: Self-contained, builds binary, reproducible
#
# Usage: ./run.sh [--rebuild]
#   --rebuild  Force rebuild of the binary
#
# Requires: Go 1.24+, Python 3

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$REPO_ROOT/simulation_worker"

# Build if needed
if [[ "${1:-}" == "--rebuild" ]] || [[ ! -x "$BINARY" ]]; then
    echo "Building simulation_worker..."
    (cd "$REPO_ROOT" && go build -o simulation_worker main.go)
fi

MODEL="meta-llama/llama-3.1-8b-instruct"
INSTANCES=4
NUM_REQUESTS=500

# Workload YAML with Gamma arrivals (CV=3.5) — bursty traffic pattern
WORKLOAD_YAML="$SCRIPT_DIR/bursty-workload.yaml"
cat > "$WORKLOAD_YAML" <<'YAMLEOF'
version: "1"
seed: 42
category: language
aggregate_rate: 2000
clients:
  - id: bursty-client
    tenant_id: tenant-A
    slo_class: batch
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
YAMLEOF

run_sim() {
    local seed="$1"; shift
    "$BINARY" run \
        --model "$MODEL" \
        --num-instances "$INSTANCES" \
        --num-requests "$NUM_REQUESTS" \
        --workload-spec "$WORKLOAD_YAML" \
        --seed "$seed" \
        --routing-policy least-loaded \
        --scheduler fcfs \
        --priority-policy constant \
        --log error \
        --summarize-trace \
        --trace-level decisions \
        "$@" 2>/dev/null
}

analyze() {
    python3 "$SCRIPT_DIR/analyze.py" "$@"
}

echo "============================================================================"
echo "  H5: Token-Bucket Admission Control Under Burst"
echo "  Reference: docs/plans/research.md, Idea 1, Hypothesis 5"
echo "  Type: Statistical / Dominance"
echo "============================================================================"
echo ""
echo "Workload: Gamma arrivals (CV=3.5), rate=2000, requests=$NUM_REQUESTS, instances=$INSTANCES"
echo ""

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# ── Experiment 1: Core hypothesis (3 seeds) ──────────────────────────────────
# Compare always-admit vs token-bucket under bursty arrivals
# Token-bucket: capacity=500, refill=400 (lower than burst peak)

echo "Experiment 1: Core hypothesis — always-admit vs token-bucket (3 seeds)"
echo ""

for SEED in 42 123 456; do
    echo "  Seed $SEED: always-admit..."
    run_sim "$SEED" \
        --admission-policy always-admit \
        > "$RESULTS_DIR/exp1_always_${SEED}.txt"

    echo "  Seed $SEED: token-bucket (cap=500, refill=400)..."
    run_sim "$SEED" \
        --admission-policy token-bucket \
        --token-bucket-capacity 500 \
        --token-bucket-refill-rate 400 \
        > "$RESULTS_DIR/exp1_bucket_${SEED}.txt"
done

analyze core "$RESULTS_DIR"/exp1_*.txt

# ── Experiment 2: Rate scaling — does burst effect amplify at higher rates? ──
# Compare at low rate (200) where Gamma burstiness shouldn't matter
# vs high rate (2000) where bursts should cause queue buildup

echo ""
echo "Experiment 2: Rate scaling — burst effect at different rates"
echo ""

# Low rate: generate a separate workload YAML per rate
for RATE in 200 500 1000 2000 3000; do
    RATE_YAML="$RESULTS_DIR/workload_rate_${RATE}.yaml"
    cat > "$RATE_YAML" <<RATEYAML
version: "1"
seed: 42
category: language
aggregate_rate: $RATE
clients:
  - id: bursty-client
    tenant_id: tenant-A
    slo_class: batch
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
RATEYAML

    "$BINARY" run \
        --model "$MODEL" \
        --num-instances "$INSTANCES" \
        --num-requests "$NUM_REQUESTS" \
        --workload-spec "$RATE_YAML" \
        --seed 42 \
        --routing-policy least-loaded \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy always-admit \
        --log error \
        --summarize-trace \
        --trace-level decisions \
        2>/dev/null > "$RESULTS_DIR/exp2_always_r${RATE}.txt"

    "$BINARY" run \
        --model "$MODEL" \
        --num-instances "$INSTANCES" \
        --num-requests "$NUM_REQUESTS" \
        --workload-spec "$RATE_YAML" \
        --seed 42 \
        --routing-policy least-loaded \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy token-bucket \
        --token-bucket-capacity 500 \
        --token-bucket-refill-rate 400 \
        --log error \
        --summarize-trace \
        --trace-level decisions \
        2>/dev/null > "$RESULTS_DIR/exp2_bucket_r${RATE}.txt"
done

analyze rate-scaling "$RESULTS_DIR"/exp2_*.txt

# ── Experiment 3: Token-bucket tuning — capacity/refill sensitivity ──────────
# Vary bucket parameters to understand the sensitivity curve

echo ""
echo "Experiment 3: Token-bucket parameter sensitivity (rate=2000, seed=42)"
echo ""

for CAP_REFILL in "100:100" "250:200" "500:400" "1000:600" "2000:1000"; do
    CAP="${CAP_REFILL%%:*}"
    REFILL="${CAP_REFILL##*:}"

    run_sim 42 \
        --admission-policy token-bucket \
        --token-bucket-capacity "$CAP" \
        --token-bucket-refill-rate "$REFILL" \
        > "$RESULTS_DIR/exp3_c${CAP}_r${REFILL}.txt"
done

# Also include always-admit baseline for comparison
run_sim 42 \
    --admission-policy always-admit \
    > "$RESULTS_DIR/exp3_always.txt"

analyze tuning "$RESULTS_DIR"/exp3_*.txt

# ── Experiment 4: Calibrated bucket (Opus 4.6 review) ────────────────────────
# Previous experiments used cap=500 < mean_input=512 — structurally impossible.
# This experiment uses properly calibrated parameters where the bucket CAN
# admit average-sized requests: cap=100000, refill matched to target throughput.

echo ""
echo "Experiment 4: Calibrated bucket — cap >> mean_input (3 seeds)"
echo "  (Addresses reviewer feedback: cap=500 < mean_input=512 is structural rejection)"
echo ""

for SEED in 42 123 456; do
    echo "  Seed $SEED: always-admit..."
    run_sim "$SEED" \
        --admission-policy always-admit \
        > "$RESULTS_DIR/exp4_always_${SEED}.txt"

    echo "  Seed $SEED: calibrated bucket (cap=100000, refill=600000)..."
    run_sim "$SEED" \
        --admission-policy token-bucket \
        --token-bucket-capacity 100000 \
        --token-bucket-refill-rate 600000 \
        > "$RESULTS_DIR/exp4_calibrated_${SEED}.txt"
done

analyze calibrated "$RESULTS_DIR"/exp4_*.txt

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis"
echo "============================================================================"

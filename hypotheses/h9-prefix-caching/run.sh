#!/bin/bash
# H9: Prefix Caching Effectiveness
#
# Hypothesis: TTFT should decrease monotonically as prefix_length increases
# (holding total input constant at ~768 tokens), because more cached blocks
# means fewer new tokens to prefill.
#
# Mechanism under test:
#   sim/kvcache.go:126-138   — GetCachedBlocks() hash matching
#   sim/simulator.go:478-479 — numNewTokens reduction from cached blocks
#
# Experiment 1: Core monotonicity (single instance, 5 prefix lengths, 3 seeds)
# Experiment 2: Cluster scale with prefix-affinity routing (4 instances)
# Experiment 3: Cache capacity stress test (varying total-kv-blocks)
#
# Usage: ./run.sh [--rebuild]

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
SEEDS=(42 123 456)

# Total input tokens target: ~768 per request
# prefix_length + user_part = ~768, so user_part = 768 - prefix_length
PREFIXES=(0 64 128 256 512)
TOTAL_INPUT=768

analyze() {
    python3 "$SCRIPT_DIR/analyze.py" "$@"
}

# Create workload YAML for a given prefix_length
# The prefix is prepended to sampled input (generator.go:152-154),
# so we set input_distribution mean = total - prefix_length
make_workload() {
    local prefix_len=$1
    local seed=$2
    local outfile=$3
    local user_part=$((TOTAL_INPUT - prefix_len))

    if [[ $prefix_len -eq 0 ]]; then
        # No prefix group — all unique tokens
        cat > "$outfile" << YAMLEOF
version: "1"
seed: $seed
category: language
aggregate_rate: 100.0
num_requests: 200
clients:
  - id: "no-prefix"
    tenant_id: "test"
    slo_class: "interactive"
    rate_fraction: 1.0
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: $user_part
        std_dev: 10
        min: $((user_part - 32))
        max: $((user_part + 32))
    output_distribution:
      type: gaussian
      params:
        mean: 64
        std_dev: 10
        min: 32
        max: 128
YAMLEOF
    else
        cat > "$outfile" << YAMLEOF
version: "1"
seed: $seed
category: language
aggregate_rate: 100.0
num_requests: 200
clients:
  - id: "prefix-${prefix_len}"
    tenant_id: "test"
    slo_class: "interactive"
    rate_fraction: 1.0
    streaming: true
    prefix_group: "shared-prefix"
    prefix_length: $prefix_len
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: $user_part
        std_dev: 10
        min: $((user_part - 32))
        max: $((user_part + 32))
    output_distribution:
      type: gaussian
      params:
        mean: 64
        std_dev: 10
        min: 32
        max: 128
YAMLEOF
    fi
}

echo "============================================================================"
echo "  H9: Prefix Caching Effectiveness"
echo "  Hypothesis: TTFT decreases monotonically with increasing prefix_length"
echo "  Reference: docs/plans/research.md, sim/kvcache.go:126-138"
echo "============================================================================"
echo ""

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# ── Experiment 1: Core monotonicity (single instance) ────────────────────────

echo "Experiment 1: Core Monotonicity (single instance)"
echo "  Config: 1 instance, 200 requests, rate=100, total_input≈${TOTAL_INPUT}"
echo "  Prefix lengths: ${PREFIXES[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo ""

for seed in "${SEEDS[@]}"; do
    for pfx in "${PREFIXES[@]}"; do
        make_workload "$pfx" "$seed" "$RESULTS_DIR/wl_p${pfx}_${seed}.yaml"
        "$BINARY" run \
            --model "$MODEL" \
            --num-instances 1 \
            --seed "$seed" \
            --workload-spec "$RESULTS_DIR/wl_p${pfx}_${seed}.yaml" \
            --log error \
            2>/dev/null \
            > "$RESULTS_DIR/exp1_p${pfx}_${seed}.txt"
    done
done

analyze monotonicity "$RESULTS_DIR"/exp1_*.txt

# ── Experiment 2: Cluster with prefix-affinity routing ────────────────────────

echo ""
echo "============================================================================"
echo "Experiment 2: Cluster Scale with Prefix-Affinity Routing"
echo "  Config: 4 instances, 200 requests, rate=100, weighted routing"
echo "  Scorers: prefix-affinity:3,queue-depth:2"
echo "  Prefix lengths: ${PREFIXES[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo ""

for seed in "${SEEDS[@]}"; do
    for pfx in "${PREFIXES[@]}"; do
        make_workload "$pfx" "$seed" "$RESULTS_DIR/wl2_p${pfx}_${seed}.yaml"
        "$BINARY" run \
            --model "$MODEL" \
            --num-instances 4 \
            --seed "$seed" \
            --routing-policy weighted \
            --routing-scorers "prefix-affinity:3,queue-depth:2" \
            --workload-spec "$RESULTS_DIR/wl2_p${pfx}_${seed}.yaml" \
            --log error \
            --summarize-trace \
            --trace-level decisions \
            2>/dev/null \
            > "$RESULTS_DIR/exp2_p${pfx}_${seed}.txt"
    done
done

analyze cluster "$RESULTS_DIR"/exp2_*.txt

# ── Experiment 3: Cache capacity stress test ──────────────────────────────────

echo ""
echo "============================================================================"
echo "Experiment 3: Cache Capacity Stress Test"
echo "  Config: 1 instance, 200 requests, prefix_length=256, varying cache size"
echo "  Cache sizes (total-kv-blocks): 50, 100, 500, 5000, 1000000"
echo "  Seeds: ${SEEDS[*]}"
echo ""

CACHE_SIZES=(50 100 500 5000 1000000)
PFX_FIXED=256

for seed in "${SEEDS[@]}"; do
    make_workload "$PFX_FIXED" "$seed" "$RESULTS_DIR/wl3_${seed}.yaml"
    for blocks in "${CACHE_SIZES[@]}"; do
        "$BINARY" run \
            --model "$MODEL" \
            --num-instances 1 \
            --seed "$seed" \
            --total-kv-blocks "$blocks" \
            --workload-spec "$RESULTS_DIR/wl3_${seed}.yaml" \
            --log error \
            2>/dev/null \
            > "$RESULTS_DIR/exp3_b${blocks}_${seed}.txt"
    done
done

analyze capacity "$RESULTS_DIR"/exp3_*.txt

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis and root cause"
echo "============================================================================"

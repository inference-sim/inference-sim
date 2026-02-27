#!/bin/bash
# H8: KV Cache Pressure
#
# Hypothesis: Reducing total KV blocks should increase preemption
# frequency and worsen tail latency (monotonically).
#
# Type: Statistical / Monotonicity
# Mechanism under test:
#   sim/simulator.go:375-408 — preempt() evicts running requests when KV is full
#   sim/simulator.go:436,455 — makeRunningBatch calls preempt() on allocation failure
#
# Experiment 1: Monotonicity (5 block counts × 3 seeds)
# Experiment 2: Conservation check (INV-1 at all block counts)
#
# Design notes:
#   ED-1: Controlled comparison — only total-kv-blocks varies
#   ED-2: Rate=2000 creates enough concurrent KV pressure for preemptions
#   ED-3: Precondition — feasibility testing confirmed transition at 2100-2500 blocks
#   ED-4: Both YAML seed and CLI seed vary together (same value)
#   ED-5: Reproducible — run.sh builds binary and runs all variants
#
# Usage: ./run.sh [--rebuild]

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
BLOCK_COUNTS=(5000 3000 2200 2100 2000)

analyze() {
    python3 "$SCRIPT_DIR/analyze.py" "$@"
}

# Create workload YAML for a given seed
# High rate + medium-length tokens to create KV pressure
make_workload() {
    local seed=$1
    local outfile=$2

    cat > "$outfile" << YAMLEOF
version: "1"
seed: $seed
category: language
aggregate_rate: 2000.0
num_requests: 200
clients:
  - id: "kv-stress"
    tenant_id: "test"
    slo_class: "interactive"
    rate_fraction: 1.0
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 512
        std_dev: 50
        min: 256
        max: 768
    output_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 50
        min: 128
        max: 512
YAMLEOF
}

echo "============================================================================"
echo "  H8: KV Cache Pressure"
echo "  Hypothesis: Reducing total-kv-blocks monotonically increases preemption"
echo "              rate and worsens TTFT p99 / E2E p99"
echo "  Type: Statistical / Monotonicity"
echo "  Reference: docs/plans/research.md, sim/simulator.go:375-408"
echo "============================================================================"
echo ""

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# ── Experiment 1: Monotonicity ───────────────────────────────────────────────

echo "Experiment 1: KV Block Pressure Monotonicity"
echo "  Config: 4 instances, 200 requests, rate=2000, block_size=16"
echo "  Block counts: ${BLOCK_COUNTS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo ""

for seed in "${SEEDS[@]}"; do
    make_workload "$seed" "$RESULTS_DIR/wl_${seed}.yaml"
    for blocks in "${BLOCK_COUNTS[@]}"; do
        echo "  Running: seed=$seed blocks=$blocks ..."
        timeout 120 "$BINARY" run \
            --model "$MODEL" \
            --num-instances 4 \
            --total-kv-blocks "$blocks" \
            --block-size-in-tokens 16 \
            --seed "$seed" \
            --workload-spec "$RESULTS_DIR/wl_${seed}.yaml" \
            --log error \
            2>/dev/null \
            > "$RESULTS_DIR/exp1_b${blocks}_s${seed}.txt" \
            || echo "    WARNING: timeout or error for blocks=$blocks seed=$seed"
    done
done

echo ""
analyze monotonicity "$RESULTS_DIR"/exp1_*.txt

# ── Experiment 2: Conservation Check ─────────────────────────────────────────

echo ""
echo "============================================================================"
echo "Experiment 2: Conservation Invariant (INV-1) Under KV Pressure"
echo "  Verifying: injected == completed + still_queued + still_running"
echo ""

analyze conservation "$RESULTS_DIR"/exp1_*.txt

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis and root cause"
echo "============================================================================"

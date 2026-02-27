#!/bin/bash
# H11: Token Budget
#
# Hypothesis: Batch formation with larger token budgets should improve
# throughput but worsen ITL (inter-token latency).
#
# Type: Statistical / Monotonicity
# Family: Performance-regime
# Mechanism under test:
#   sim/simulator.go:355-463 -- makeRunningBatch() allocates a per-step token budget
#   sim/latency_model.go:35-51 -- BlackboxLatencyModel.StepTime() scales with batch tokens
#
# Experiment 1: Monotonicity (5 token budgets x 3 seeds)
#   - Throughput should increase with larger token budget
#   - ITL should increase with larger token budget (longer steps)
# Experiment 2: Conservation check (INV-1 at all budgets)
#
# Design notes:
#   ED-1: Controlled comparison -- only max-num-scheduled-tokens varies
#   ED-2: Constant input/output lengths eliminate distribution noise
#   ED-3: 5 sweep points cover 16x range (512 to 8192)
#   ED-4: 3 seeds (42, 123, 456) for statistical robustness
#   ED-5: Reproducible -- run.sh builds binary and runs all variants
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
TOKEN_BUDGETS=(512 1024 2048 4096 8192)
SEEDS=(42 123 456)

analyze() {
    python3 "$SCRIPT_DIR/analyze.py" "$@"
}

# Create workload YAML for a given seed
# Constant distribution eliminates variance from input/output length
make_workload() {
    local seed=$1
    local outfile=$2

    cat > "$outfile" << YAMLEOF
version: "1"
seed: $seed
category: language
aggregate_rate: 1000.0
num_requests: 500
clients:
  - id: "uniform"
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
echo "  H11: Token Budget"
echo "  Hypothesis: Larger token budgets improve throughput but worsen ITL"
echo "  Type: Statistical / Monotonicity"
echo "  Reference: sim/simulator.go:355-463, sim/latency_model.go:35-51"
echo "============================================================================"
echo ""

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# -- Experiment 1: Monotonicity -------------------------------------------

echo "Experiment 1: Token Budget Monotonicity"
echo "  Config: 4 instances, round-robin, always-admit, fcfs"
echo "  500 requests, rate=1000, constant input=256 output=128"
echo "  Token budgets: ${TOKEN_BUDGETS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo ""

for seed in "${SEEDS[@]}"; do
    make_workload "$seed" "$RESULTS_DIR/wl_${seed}.yaml"
    for budget in "${TOKEN_BUDGETS[@]}"; do
        echo "  Running: seed=$seed budget=$budget ..."
        timeout 120 "$BINARY" run \
            --model "$MODEL" \
            --num-instances 4 \
            --max-num-scheduled-tokens "$budget" \
            --seed "$seed" \
            --workload-spec "$RESULTS_DIR/wl_${seed}.yaml" \
            --log error \
            2>/dev/null \
            > "$RESULTS_DIR/exp1_t${budget}_s${seed}.txt" \
            || echo "    WARNING: timeout or error for budget=$budget seed=$seed"
    done
done

echo ""
analyze monotonicity "$RESULTS_DIR"/exp1_*.txt

# -- Experiment 2: Conservation Check (INV-1) -----------------------------

echo ""
echo "============================================================================"
echo "Experiment 2: Conservation Invariant (INV-1) Under Token Budget Sweep"
echo "  Verifying: injected == completed + still_queued + still_running"
echo ""

analyze conservation "$RESULTS_DIR"/exp1_*.txt

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis and root cause"
echo "============================================================================"

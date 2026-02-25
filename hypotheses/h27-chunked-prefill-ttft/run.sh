#!/bin/bash
# H27: Chunked Prefill Reduces Short-Request TTFT in Bimodal Workloads
#
# Hypothesis: Enabling chunked prefill (--long-prefill-token-threshold=256) reduces
# TTFT p99 for short requests (64 input tokens) by at least 30% in a bimodal workload
# (50% short at 64 tokens, 50% long at 2048 tokens) at moderate load (4 instances,
# rate near 50% saturation), because per-step time drops from ~43ms to ~11ms allowing
# short requests to be scheduled sooner.
#
# Classification: Statistical / Dominance
# Family: Scheduling & Batch Formation
# VV&UQ: Validation
# Tier: 2 (behavioral comparison)
#
# Design:
#   - ED-1: Vary exactly one dimension (--long-prefill-token-threshold: 0 vs 256)
#           A: chunking disabled (threshold=0, default)
#           B: chunking enabled (threshold=256)
#   - ED-2: Control — both configs use identical workload, instances, routing, KV
#   - ED-3: Precondition — bimodal workload: 50% short (64 tokens), 50% long (2048 tokens)
#   - ED-4: 3 seeds (42, 123, 456)
#   - ED-5: Self-contained, builds binary, reproducible
#   - ED-6: No prior experiment reference — first chunked prefill investigation
#
# Rate sizing rationale (bimodal workload):
#   Short requests (50%): input=64, output=128
#     stepTime = 6910 + 17.67*64 + 2.84*128 = 6910 + 1131 + 364 = ~8405 us ~= 8.4ms
#   Long requests (50%): input=2048, output=128
#     stepTime = 6910 + 17.67*2048 + 2.84*128 = 6910 + 36177 + 364 = ~43451 us ~= 43.5ms
#   Average step time ~= 0.5*8.4 + 0.5*43.5 = ~25.9ms
#   Per-instance capacity ~= 1/0.0259 ~= 38.6 req/s
#   4 instances: capacity ~= 154.4 req/s
#   50% saturation: rate ~= 77 req/s (use 78)
#
# With chunked prefill (threshold=256):
#   Long request (2048 tokens) is split into ceil(2048/256) = 8 prefill chunks.
#   Each chunk step: ~6910 + 17.67*256 + 2.84*0 = ~11434 us ~= 11.4ms
#   Short requests can interleave between chunks, reducing HOL blocking.
#
# Usage: ./run.sh [--rebuild]
#
# Requires: Go 1.24+, Python 3

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"

setup_experiment "${1:-}"

INSTANCES=4
SEEDS=(42 123 456)
RATE=78

# -- Generate bimodal workload YAML --
make_bimodal_yaml() {
    local outfile=$1
    cat > "$outfile" << 'YAMLEOF'
version: "1"
seed: 42
category: language
aggregate_rate: 78.0
num_requests: 200
clients:
  - id: "short-requests"
    tenant_id: "tenant-A"
    slo_class: "interactive"
    rate_fraction: 0.5
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 64
    output_distribution:
      type: constant
      params:
        value: 128
  - id: "long-requests"
    tenant_id: "tenant-B"
    slo_class: "interactive"
    rate_fraction: 0.5
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 2048
    output_distribution:
      type: constant
      params:
        value: 128
YAMLEOF
}

WORKLOAD_YAML="$RESULTS_DIR/bimodal_workload.yaml"
make_bimodal_yaml "$WORKLOAD_YAML"

# Preflight KV check: long requests need ceil(2048/16)=128 blocks each
preflight_kv_check 132139 16 2048

echo "=== H27: Chunked Prefill TTFT Experiment ==="
echo "Instances: $INSTANCES, Rate: $RATE req/s, Requests: 200"
echo "Config A: chunking disabled (threshold=0)"
echo "Config B: chunking enabled (threshold=256)"
echo ""

# -- Run experiments --
for seed in "${SEEDS[@]}"; do
    echo "--- Seed $seed ---"

    # Config A: chunking disabled (threshold=0)
    results_a="$RESULTS_DIR/config_a_seed${seed}_results.json"
    echo "  Running Config A (no chunking, seed=$seed)..."
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/config_a_seed${seed}.out" \
        --model "$MODEL" \
        --num-instances "$INSTANCES" \
        --workload-spec "$WORKLOAD_YAML" \
        --long-prefill-token-threshold 0 \
        --seed "$seed" \
        --results-path "$results_a" \
        --routing-policy least-loaded

    # Config B: chunking enabled (threshold=256)
    results_b="$RESULTS_DIR/config_b_seed${seed}_results.json"
    echo "  Running Config B (chunking=256, seed=$seed)..."
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/config_b_seed${seed}.out" \
        --model "$MODEL" \
        --num-instances "$INSTANCES" \
        --workload-spec "$WORKLOAD_YAML" \
        --long-prefill-token-threshold 256 \
        --seed "$seed" \
        --results-path "$results_b" \
        --routing-policy least-loaded

    echo ""
done

# -- Verify no timeouts --
echo "=== Checking for timeouts ==="
any_timeout=false
for seed in "${SEEDS[@]}"; do
    for config in a b; do
        if is_timeout "$RESULTS_DIR/config_${config}_seed${seed}.out"; then
            echo "TIMEOUT/ERROR in config_${config}_seed${seed}"
            any_timeout=true
        fi
    done
done

if $any_timeout; then
    echo "ERROR: One or more runs timed out. Results are unreliable."
    exit 1
fi
echo "All runs completed successfully."

# -- Run analysis --
echo ""
echo "=== Analysis ==="
python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR"

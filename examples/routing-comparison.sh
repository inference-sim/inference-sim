#!/bin/bash
# BLIS Routing Policy Comparison
#
# This script demonstrates how routing policy and scorer weight choices
# dramatically affect performance — especially tail latency (TTFT p99).
#
# The key insight: at high request rates (5000 req/s), the scorer
# dimension you choose matters enormously because different metrics
# update at different speeds:
#
#   - queue-depth:  uses EffectiveLoad (QueueDepth + BatchSize + PendingRequests)
#                   PendingRequests updates INSTANTLY when a request is routed,
#                   so the next routing decision immediately sees the change.
#                   Result: even load distribution, low tail latency.
#
#   - kv-utilization: uses KVUtilization (fraction of KV blocks in use)
#                     KV blocks are only allocated during batch formation,
#                     which happens on a step timer (~9ms for 8B models).
#                     At 5000 req/s, ~45 routing decisions happen between
#                     KV updates, all seeing the same stale utilization.
#                     Result: severely skewed distribution, 3x worse tail latency.
#
# Expected output (llama-3.1-8b-instruct, 4 instances, 1000 requests):
#
#   | Configuration            | Distribution          | TTFT p99 (ms) |
#   |--------------------------|-----------------------|---------------|
#   | round-robin              | 250/250/250/250       | ~2,626        |
#   | least-loaded             | 251/250/250/249       | ~2,598        |
#   | weighted queue-depth:1   | 251/250/250/249       | ~2,598        |
#   | weighted kv-utilization:1| 333/423/47/197        | ~7,870        |
#   | weighted (default)       | 252/250/249/249       | ~2,634        |
#
#   The kv-utilization-only configuration produces 3x worse TTFT p99
#   because KV state is a lagging indicator — it creates a stale-signal
#   feedback loop that concentrates requests on fewer instances.
#
# Usage:
#   chmod +x examples/routing-comparison.sh
#   ./examples/routing-comparison.sh
#
# Prerequisites:
#   go build -o simulation_worker main.go

set -e

BINARY=${BLIS_BINARY:-./simulation_worker}
MODEL="meta-llama/llama-3.1-8b-instruct"
INSTANCES=4
REQUESTS=1000
RATE=5000

if [ ! -f "$BINARY" ]; then
    echo "Building BLIS..."
    go build -o simulation_worker main.go
    BINARY=./simulation_worker
fi

echo "================================================================"
echo "  BLIS Routing Policy Comparison"
echo "  Model: $MODEL | Instances: $INSTANCES | Requests: $REQUESTS | Rate: $RATE req/s"
echo "================================================================"
echo ""

run_experiment() {
    local label="$1"
    shift
    echo "--- $label ---"
    $BINARY run --model "$MODEL" --num-instances $INSTANCES \
        --num-requests $REQUESTS --rate $RATE \
        --trace-level decisions --summarize-trace --log error "$@" 2>/dev/null \
        | grep -E "(\"ttft_p99|\"ttft_mean|\"responses_per_sec|Target Dist|  instance)" \
        | tail -8
    echo ""
}

run_experiment "round-robin" \
    --routing-policy round-robin

run_experiment "least-loaded" \
    --routing-policy least-loaded

run_experiment "weighted (queue-depth:1 — fast signal)" \
    --routing-policy weighted --routing-scorers "queue-depth:1"

run_experiment "weighted (kv-utilization:1 — lagging signal)" \
    --routing-policy weighted --routing-scorers "kv-utilization:1"

run_experiment "weighted (default: queue-depth:2,kv-utilization:2,load-balance:1)" \
    --routing-policy weighted

echo "================================================================"
echo "  Key takeaway: kv-utilization alone produces ~3x worse TTFT p99"
echo "  because KV state lags behind queue state at high request rates."
echo "  The default profile includes queue-depth to prevent this."
echo "================================================================"

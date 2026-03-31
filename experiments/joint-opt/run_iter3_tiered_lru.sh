#!/usr/bin/env bash
set -euo pipefail
# Iteration 3: H-main for SLO-aware tiered LRU KV eviction.
# Note: Tiered LRU is structural (always active in this build).
# H-main comparison: these results vs Iteration 2 treatment results.
# Run from repo root after building with tiered-LRU changes.

cd "$(dirname "$0")/../.."

MODEL="${MODEL:-qwen/qwen3-14b}"
SEEDS=(42 123 456)
RESULTS_DIR="experiments/joint-opt/results/iter3"
mkdir -p "$RESULTS_DIR"

for SEED in "${SEEDS[@]}"; do
  echo "=== Iter 3 (tiered-LRU + slo-priority-preemption) — seed $SEED ==="
  ./blis run \
    --model        "$MODEL" \
    --seed         "$SEED" \
    --workload-spec experiments/joint-opt/workload-mixed.yaml \
    --admission-policy   tier-shed \
    --routing-policy     weighted \
    --routing-scorers    "prefix-affinity:4,queue-depth:3" \
    --priority-policy    slo-based \
    --scheduler          priority-fcfs \
    --batch-formation    slo-priority-preemption \
    > "$RESULTS_DIR/seed${SEED}.json"
done
echo "Done. Compare vs experiments/joint-opt/results/iter2/treatment_seed*.json"

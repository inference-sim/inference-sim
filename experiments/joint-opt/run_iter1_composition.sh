#!/usr/bin/env bash
set -euo pipefail
# Iteration 1: Joint composition validation.
# Run with: MODEL=qwen/qwen3-14b bash experiments/joint-opt/run_iter1_composition.sh
#
# Purpose: This IS the joint composition — all four components active together:
#   - SLO-aware routing (pa:4,qd:3)
#   - SLO-based priority (priority-fcfs)
#   - Tier-shed admission
#   - vllm batch formation (LIFO baseline)
#
# Identical flags to Iteration 0. The purpose of this iteration is to validate
# the joint behavior explicitly and confirm reproducibility.

cd "$(dirname "$0")/../.."   # repo root

MODEL="${MODEL:-qwen/qwen3-14b}"
SEEDS=(42 123 456)
RESULTS_DIR="experiments/joint-opt/results/iter1"
mkdir -p "$RESULTS_DIR"

for SEED in "${SEEDS[@]}"; do
  echo "=== Iter 1 composition — seed $SEED ==="
  ./blis run \
    --model        "$MODEL" \
    --seed         "$SEED" \
    --workload-spec experiments/joint-opt/workload-mixed.yaml \
    --admission-policy   tier-shed \
    --routing-policy     weighted \
    --routing-scorers    "prefix-affinity:4,queue-depth:3" \
    --priority-policy    slo-based \
    --scheduler          priority-fcfs \
    --batch-formation    vllm \
    > "$RESULTS_DIR/seed${SEED}.json"
  echo "  → $RESULTS_DIR/seed${SEED}.json"
done
echo "Done. Aggregate: jq -s '[.[].slo_metrics]' $RESULTS_DIR/*.json"

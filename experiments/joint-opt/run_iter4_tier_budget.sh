#!/usr/bin/env bash
set -euo pipefail
# Iteration 4: H-main and fraction ablation for TierBudgetBatchFormation.
# Treatment: --batch-formation tier-budget --tier-budget-critical-frac 0.5
# Ablation: --tier-budget-critical-frac 0.333 (equal-share)

cd "$(dirname "$0")/../.."

MODEL="${MODEL:-qwen/qwen3-14b}"
SEEDS=(42 123 456)
RESULTS_DIR="experiments/joint-opt/results/iter4"
mkdir -p "$RESULTS_DIR"

for SEED in "${SEEDS[@]}"; do
  echo "=== Iter 4 treatment (tier-budget f_c=0.5) — seed $SEED ==="
  ./blis run \
    --model        "$MODEL" \
    --seed         "$SEED" \
    --workload-spec experiments/joint-opt/workload-mixed.yaml \
    --admission-policy         tier-shed \
    --routing-policy           weighted \
    --routing-scorers          "prefix-affinity:4,queue-depth:3" \
    --priority-policy          slo-based \
    --scheduler                priority-fcfs \
    --batch-formation          tier-budget \
    --tier-budget-critical-frac 0.50 \
    --tier-budget-standard-frac 0.70 \
    > "$RESULTS_DIR/treatment_seed${SEED}.json"

  echo "=== Iter 4 ablation (equal-share f_c=0.333) — seed $SEED ==="
  ./blis run \
    --model        "$MODEL" \
    --seed         "$SEED" \
    --workload-spec experiments/joint-opt/workload-mixed.yaml \
    --admission-policy         tier-shed \
    --routing-policy           weighted \
    --routing-scorers          "prefix-affinity:4,queue-depth:3" \
    --priority-policy          slo-based \
    --scheduler                priority-fcfs \
    --batch-formation          tier-budget \
    --tier-budget-critical-frac 0.333 \
    --tier-budget-standard-frac 0.50 \
    > "$RESULTS_DIR/ablation_seed${SEED}.json"
done
echo "Done."

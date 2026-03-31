#!/usr/bin/env bash
set -euo pipefail
# Iteration 0: Measure baseline compound strategy performance.
# Run with: MODEL=qwen/qwen3-14b bash experiments/joint-opt/run_iter0_baseline.sh
#
# Purpose: Establish the baseline compound (pa:4,qd:3 + priority-fcfs + tier-shed)
# under the mixed workload. Used to calibrate aggregate_rate in workload-mixed.yaml.

cd "$(dirname "$0")/../.."   # repo root

MODEL="${MODEL:-qwen/qwen3-14b}"
SEEDS=(42 123 456)
RESULTS_DIR="experiments/joint-opt/results/iter0"
mkdir -p "$RESULTS_DIR"

for SEED in "${SEEDS[@]}"; do
  echo "=== Iter 0 baseline — seed $SEED ==="
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

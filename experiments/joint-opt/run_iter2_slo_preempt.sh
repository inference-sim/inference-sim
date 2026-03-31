#!/usr/bin/env bash
set -euo pipefail
# Iteration 2: H-main and ablation for SLO-priority preemption ordering.
# Treatment: --batch-formation slo-priority-preemption
# Ablation:  --batch-formation vllm  (LIFO, everything else same)

cd "$(dirname "$0")/../.."

MODEL="${MODEL:-qwen/qwen3-14b}"
SEEDS=(42 123 456)
RESULTS_DIR="experiments/joint-opt/results/iter2"
mkdir -p "$RESULTS_DIR"

for SEED in "${SEEDS[@]}"; do
  echo "=== Iter 2 treatment (slo-priority-preemption) — seed $SEED ==="
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
    > "$RESULTS_DIR/treatment_seed${SEED}.json"

  echo "=== Iter 2 ablation (vllm/LIFO) — seed $SEED ==="
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
    > "$RESULTS_DIR/ablation_seed${SEED}.json"
done
echo "Done. Compare: jq '.slo_metrics.ttft_p99_by_class' $RESULTS_DIR/*.json"

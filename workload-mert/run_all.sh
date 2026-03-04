#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="meta-llama/llama-3.1-8b-instruct"
NUM_INSTANCES=4
POLICY_CONFIG='admission:
  policy: always-admit
priority:
  policy: constant
routing:
  policy: weighted
  scorers:
  - name: prefix-affinity
    weight: 1.0
  - name: load-balance
    weight: 1.0
scheduler: fcfs'

WORKLOADS=(
  workload-mert/workload_v2_cache_warmup.yaml
  workload-mert/workload_v2_load_spikes.yaml
  workload-mert/workload_v2_multiturn.yaml
)

POLICY_FILE=$(mktemp /tmp/blis-policy-XXXXXX.yaml)
echo "$POLICY_CONFIG" > "$POLICY_FILE"
trap 'rm -f "$POLICY_FILE"' EXIT

TOTAL_START=$SECONDS

for workload in "${WORKLOADS[@]}"; do
  name=$(basename "$workload" .yaml)
  echo "========================================="
  echo "Running: $name"
  echo "========================================="
  START=$SECONDS
  ./blis run \
    --model "$MODEL" \
    --num-instances "$NUM_INSTANCES" \
    --policy-config "$POLICY_FILE" \
    --workload-spec "$workload"
  ELAPSED=$((SECONDS - START))
  echo "Duration: ${ELAPSED}s"
  echo ""
done

TOTAL_ELAPSED=$((SECONDS - TOTAL_START))
echo "========================================="
echo "Total duration: ${TOTAL_ELAPSED}s"
echo "========================================="

#!/bin/bash
# Iteration 7: Comprehensive robustness sweep
BIN="./strategy-evolution/simulation_worker"
P="python3 strategy-evolution/parse_results.py"
BASE_ARGS="--model meta-llama/llama-3.1-8b-instruct --num-instances 8 --kv-offload-threshold 0.9 --long-prefill-token-threshold 256 --seed 42"
BASELINE_POLICY="--routing-policy weighted --routing-scorers prefix-affinity:3,queue-depth:2,kv-utilization:2 --scheduler fcfs --priority-policy constant"
BEST_POLICY="--routing-policy weighted --routing-scorers prefix-affinity:3,queue-depth:2,kv-utilization:2 --scheduler priority-fcfs --priority-policy slo-tiered --slo-priority-bridge --slo-prefill --slo-prefill-critical 0 --slo-prefill-sheddable 0"

echo "=============================================="
echo "  SECTION 1: KV PRESSURE SWEEP"
echo "=============================================="
for blocks in 132139 20000 10000 5000; do
  cpu_blocks=$((blocks / 3))
  echo ""
  echo "--- KV blocks=$blocks (cpu=$cpu_blocks) ---"
  echo -n "  BL: "
  $BIN run $BASE_ARGS $BASELINE_POLICY --total-kv-blocks $blocks --kv-cpu-blocks $cpu_blocks --workload-spec strategy-evolution/lib/mixed-production-workload.yaml 2>/dev/null | $P
  echo -n "  ST: "
  $BIN run $BASE_ARGS $BEST_POLICY --total-kv-blocks $blocks --kv-cpu-blocks $cpu_blocks --workload-spec strategy-evolution/lib/mixed-production-workload.yaml 2>/dev/null | $P
done

echo ""
echo "=============================================="
echo "  SECTION 2: WORKLOAD SHAPE VARIANTS"
echo "=============================================="

echo ""
echo "--- Orthogonal (reference) ---"
echo -n "  BL: "
$BIN run $BASE_ARGS $BASELINE_POLICY --kv-cpu-blocks 44000 --workload-spec strategy-evolution/lib/mixed-production-workload.yaml 2>/dev/null | $P
echo -n "  ST: "
$BIN run $BASE_ARGS $BEST_POLICY --kv-cpu-blocks 44000 --workload-spec strategy-evolution/lib/mixed-production-workload.yaml 2>/dev/null | $P

echo ""
echo "--- Non-orthogonal (different token dists per SLO) ---"
echo -n "  BL: "
$BIN run $BASE_ARGS $BASELINE_POLICY --kv-cpu-blocks 44000 --workload-spec strategy-evolution/lib/workload-nonorthogonal.yaml 2>/dev/null | $P
echo -n "  ST: "
$BIN run $BASE_ARGS $BEST_POLICY --kv-cpu-blocks 44000 --workload-spec strategy-evolution/lib/workload-nonorthogonal.yaml 2>/dev/null | $P

echo ""
echo "--- Asymmetric (5% crit, 80% sheddable) ---"
echo -n "  BL: "
$BIN run $BASE_ARGS $BASELINE_POLICY --kv-cpu-blocks 44000 --workload-spec strategy-evolution/lib/workload-asymmetric.yaml 2>/dev/null | $P
echo -n "  ST: "
$BIN run $BASE_ARGS $BEST_POLICY --kv-cpu-blocks 44000 --workload-spec strategy-evolution/lib/workload-asymmetric.yaml 2>/dev/null | $P

echo ""
echo "--- Multi-prefix (3 different prefix groups) ---"
echo -n "  BL: "
$BIN run $BASE_ARGS $BASELINE_POLICY --kv-cpu-blocks 44000 --workload-spec strategy-evolution/lib/workload-multiprefix.yaml 2>/dev/null | $P
echo -n "  ST: "
$BIN run $BASE_ARGS $BEST_POLICY --kv-cpu-blocks 44000 --workload-spec strategy-evolution/lib/workload-multiprefix.yaml 2>/dev/null | $P

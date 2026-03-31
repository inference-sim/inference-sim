#!/usr/bin/env bash
# run.sh — Joint SLO Optimization experiment runner
#
# USAGE
#   ./run.sh calibrate            Measure saturation throughput, update workload.yaml
#   ./run.sh iter0                Baseline measurement (joint compound, 3 seeds)
#   ./run.sh iter1                Joint composition vs BLIS defaults
#   ./run.sh iter2                SLO-priority preemption + ablation
#   ./run.sh iter3                Tiered-LRU comparison (needs two binary builds)
#   ./run.sh iter4                Tier-budget formation + fraction sweep
#   ./run.sh all                  Run iter0 → iter4 in sequence
#
# PREREQUISITES
#   1. Build blis: go build -o blis main.go  (from repo root)
#   2. Set MODEL env var: export MODEL=qwen/qwen3-14b
#   3. Run ./run.sh calibrate to set aggregate_rate in workload.yaml
#
# All results are written to results/iter{N}/*.json
# All commands are run from the repo root (cd up two levels from this script).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-qwen/qwen3-14b}"
SEEDS=(42 123 456)
WORKLOAD="$SCRIPT_DIR/workload.yaml"
RESULTS="$SCRIPT_DIR/results"
BLIS="${BLIS:-./blis}"

# ── Shared flag sets ────────────────────────────────────────────────────────

# The joint compound: best known from prior tracks
COMPOUND_FLAGS=(
  --admission-policy   tier-shed
  --routing-policy     weighted
  --routing-scorers    "prefix-affinity:4,queue-depth:3"
  --priority-policy    slo-based
  --scheduler          priority-fcfs
  --batch-formation    vllm
)

# BLIS defaults (for H-main comparison in Iter 1)
DEFAULT_FLAGS=(
  --admission-policy   always-admit
  --routing-policy     round-robin
  --priority-policy    constant
  --scheduler          fcfs
  --batch-formation    vllm
)

# ── Helpers ─────────────────────────────────────────────────────────────────

run_seeds() {
  local label="$1"; shift
  local out_dir="$1"; shift
  mkdir -p "$out_dir"
  for SEED in "${SEEDS[@]}"; do
    echo "  [seed $SEED] $label"
    "$BLIS" run \
      --model "$MODEL" \
      --seed "$SEED" \
      --workload-spec "$WORKLOAD" \
      "$@" \
      > "$out_dir/seed${SEED}.json"
  done
  echo "  → $out_dir/"
}

# ── calibrate ───────────────────────────────────────────────────────────────
# Runs a binary search over aggregate_rate to find the saturation point.
# Saturation = highest rate at which all requests complete (no timeouts, queue < 5%).
# Updates aggregate_rate in workload.yaml to 2× saturation.

do_calibrate() {
  echo "=== Calibration: finding saturation throughput ==="
  echo ""
  echo "NOTE: Calibration requires manual binary search."
  echo "Suggested procedure:"
  echo ""
  echo "  1. Edit workload.yaml: set aggregate_rate to a low value (e.g. 50)"
  echo "  2. Run: $BLIS run --model $MODEL --seed 42 --workload-spec $WORKLOAD ${COMPOUND_FLAGS[*]}"
  echo "  3. Check stdout for timeout_count and queue depth"
  echo "  4. Double aggregate_rate and repeat until timeouts appear"
  echo "  5. Binary search between last-good and first-bad values"
  echo "  6. Record saturation_rate = last rate with 0 timeouts"
  echo "  7. Set aggregate_rate = 2 × saturation_rate in workload.yaml"
  echo ""
  echo "Quick single-rate probe:"
  echo "  RATE=100 $0 probe"
  echo ""
  echo "After calibration, record the saturation rate in FINDINGS.md."
}

do_probe() {
  local rate="${RATE:-100}"
  echo "=== Probe: aggregate_rate=$rate ==="
  # Temporarily patch workload rate
  local tmp=$(mktemp)
  sed "s/^aggregate_rate:.*/aggregate_rate: $rate/" "$WORKLOAD" > "$tmp"
  "$BLIS" run \
    --model "$MODEL" --seed 42 \
    --workload-spec "$tmp" \
    "${COMPOUND_FLAGS[@]}" \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
m = d.get('metrics', d)
print(f'  rate={$rate}  completed={m.get(\"completed_requests\",\"?\")}  timeouts={m.get(\"timed_out_requests\",0)}  p99={m.get(\"ttft_p99_ms\",\"?\")}ms')
"
  rm -f "$tmp"
}

# ── iter0: baseline measurement ─────────────────────────────────────────────

do_iter0() {
  echo "=== Iteration 0: Baseline measurement ==="
  run_seeds "baseline (joint compound)" "$RESULTS/iter0" "${COMPOUND_FLAGS[@]}"
  echo "Done. Analyze: python3 analyze.py iter0"
}

# ── iter1: joint composition vs BLIS defaults ────────────────────────────────

do_iter1() {
  echo "=== Iteration 1: Joint composition validation ==="

  echo "--- H-main: compound vs BLIS defaults ---"
  run_seeds "compound" "$RESULTS/iter1/compound" "${COMPOUND_FLAGS[@]}"
  run_seeds "blis-defaults" "$RESULTS/iter1/blis-defaults" "${DEFAULT_FLAGS[@]}"

  echo "--- H-ablation: routing ---"
  run_seeds "abl-routing (round-robin)" "$RESULTS/iter1/abl-routing" \
    --admission-policy tier-shed \
    --routing-policy round-robin \
    --priority-policy slo-based \
    --scheduler priority-fcfs \
    --batch-formation vllm

  echo "--- H-ablation: scheduling ---"
  run_seeds "abl-scheduling (fcfs)" "$RESULTS/iter1/abl-scheduling" \
    --admission-policy tier-shed \
    --routing-policy weighted \
    --routing-scorers "prefix-affinity:4,queue-depth:3" \
    --priority-policy constant \
    --scheduler fcfs \
    --batch-formation vllm

  echo "--- H-ablation: no-chunk ---"
  # no-chunk is controlled by longprefill-token-threshold; 0 = always chunk
  run_seeds "abl-nochunk (chunked for all)" "$RESULTS/iter1/abl-nochunk" \
    "${COMPOUND_FLAGS[@]}"
    # NOTE: Add --longprefill-token-threshold 0 when that flag is available

  echo "--- H-ablation: admission ---"
  run_seeds "abl-admission (always-admit)" "$RESULTS/iter1/abl-admission" \
    --admission-policy always-admit \
    --routing-policy weighted \
    --routing-scorers "prefix-affinity:4,queue-depth:3" \
    --priority-policy slo-based \
    --scheduler priority-fcfs \
    --batch-formation vllm

  echo "--- H-control-negative: uniform SLO ---"
  # All-standard workload: patch slo_class to standard in a temp spec
  local tmp=$(mktemp --suffix=.yaml)
  sed 's/slo_class: "[^"]*"/slo_class: "standard"/g' "$WORKLOAD" > "$tmp"
  mkdir -p "$RESULTS/iter1/ctrl-uniform-slo"
  for SEED in "${SEEDS[@]}"; do
    echo "  [seed $SEED] ctrl-uniform-slo"
    "$BLIS" run --model "$MODEL" --seed "$SEED" --workload-spec "$tmp" \
      "${COMPOUND_FLAGS[@]}" \
      > "$RESULTS/iter1/ctrl-uniform-slo/seed${SEED}.json"
  done
  rm -f "$tmp"

  echo "Done. Analyze: python3 analyze.py iter1"
}

# ── iter2: SLO-priority preemption ──────────────────────────────────────────

do_iter2() {
  echo "=== Iteration 2: SLO-priority preemption ordering ==="

  echo "--- H-main: SLO-priority vs LIFO ---"
  run_seeds "slo-priority-preemption" "$RESULTS/iter2/treatment" \
    --admission-policy tier-shed \
    --routing-policy weighted \
    --routing-scorers "prefix-affinity:4,queue-depth:3" \
    --priority-policy slo-based \
    --scheduler priority-fcfs \
    --batch-formation slo-priority-preemption

  run_seeds "lifo-ablation" "$RESULTS/iter2/ablation" \
    "${COMPOUND_FLAGS[@]}"  # vllm = LIFO, same as iter1

  echo "--- H-control-negative: abundant KV ---"
  # Increase KV blocks to 4× default to make preemption rare
  run_seeds "abundant-kv" "$RESULTS/iter2/ctrl-abundant-kv" \
    --admission-policy tier-shed \
    --routing-policy weighted \
    --routing-scorers "prefix-affinity:4,queue-depth:3" \
    --priority-policy slo-based \
    --scheduler priority-fcfs \
    --batch-formation slo-priority-preemption \
    --kv-blocks 524288   # 4× default 131072; adjust per hardware

  echo "Done. Analyze: python3 analyze.py iter2"
}

# ── iter3: tiered LRU (structural — needs two builds) ───────────────────────

do_iter3() {
  echo "=== Iteration 3: Tiered LRU KV eviction ==="
  echo ""
  echo "NOTE: This iteration compares two binary builds:"
  echo "  BLIS_NEW (with tiered-LRU, PR #901) = \$BLIS (current)"
  echo "  BLIS_OLD (pre-PR-#901 build)        = set \$BLIS_OLD env var"
  echo ""

  BLIS_OLD="${BLIS_OLD:-}"
  if [[ -z "$BLIS_OLD" ]]; then
    echo "BLIS_OLD not set. Build the pre-PR-#901 binary and set:"
    echo "  export BLIS_OLD=/path/to/blis-without-tiered-lru"
    echo "  ./run.sh iter3"
    exit 1
  fi

  echo "--- Treatment: tiered-LRU build (PR #901) ---"
  run_seeds "tiered-lru" "$RESULTS/iter3/treatment" \
    --admission-policy tier-shed \
    --routing-policy weighted \
    --routing-scorers "prefix-affinity:4,queue-depth:3" \
    --priority-policy slo-based \
    --scheduler priority-fcfs \
    --batch-formation slo-priority-preemption

  echo "--- Ablation: single-list LRU build (pre-PR-#901) ---"
  mkdir -p "$RESULTS/iter3/ablation"
  for SEED in "${SEEDS[@]}"; do
    echo "  [seed $SEED] single-list-lru"
    "$BLIS_OLD" run \
      --model "$MODEL" --seed "$SEED" --workload-spec "$WORKLOAD" \
      --admission-policy tier-shed \
      --routing-policy weighted \
      --routing-scorers "prefix-affinity:4,queue-depth:3" \
      --priority-policy slo-based \
      --scheduler priority-fcfs \
      --batch-formation slo-priority-preemption \
      > "$RESULTS/iter3/ablation/seed${SEED}.json"
  done

  echo "Done. Analyze: python3 analyze.py iter3"
}

# ── iter4: tier-budget batch formation ──────────────────────────────────────

do_iter4() {
  echo "=== Iteration 4: Admission-feedback batch formation ==="

  echo "--- H-main: tier-budget f_c=0.50 ---"
  run_seeds "tier-budget-fc050" "$RESULTS/iter4/treatment" \
    --admission-policy tier-shed \
    --routing-policy weighted \
    --routing-scorers "prefix-affinity:4,queue-depth:3" \
    --priority-policy slo-based \
    --scheduler priority-fcfs \
    --batch-formation tier-budget \
    --tier-budget-critical-frac 0.50 \
    --tier-budget-standard-frac 0.70

  echo "--- H-ablation: equal-share f_c=0.333 ---"
  run_seeds "tier-budget-fc033" "$RESULTS/iter4/ablation-equal-share" \
    --admission-policy tier-shed \
    --routing-policy weighted \
    --routing-scorers "prefix-affinity:4,queue-depth:3" \
    --priority-policy slo-based \
    --scheduler priority-fcfs \
    --batch-formation tier-budget \
    --tier-budget-critical-frac 0.333 \
    --tier-budget-standard-frac 0.50

  echo "--- H-robustness: fraction sweep ---"
  for FC in 0.20 0.30 0.40 0.50 0.60 0.70; do
    FS="0.70"
    run_seeds "fc=${FC}" "$RESULTS/iter4/sweep-fc${FC/./}" \
      --admission-policy tier-shed \
      --routing-policy weighted \
      --routing-scorers "prefix-affinity:4,queue-depth:3" \
      --priority-policy slo-based \
      --scheduler priority-fcfs \
      --batch-formation tier-budget \
      --tier-budget-critical-frac "$FC" \
      --tier-budget-standard-frac "$FS"
  done

  echo "Done. Analyze: python3 analyze.py iter4"
}

# ── all ─────────────────────────────────────────────────────────────────────

do_all() {
  do_iter0
  do_iter1
  do_iter2
  # iter3 requires BLIS_OLD — skipped in batch run
  echo "Skipping iter3 (requires BLIS_OLD; run manually)"
  do_iter4
  echo ""
  echo "All iterations complete. Run: python3 analyze.py all"
}

# ── dispatch ────────────────────────────────────────────────────────────────

CMD="${1:-help}"
case "$CMD" in
  calibrate) do_calibrate ;;
  probe)     do_probe ;;
  iter0)     do_iter0 ;;
  iter1)     do_iter1 ;;
  iter2)     do_iter2 ;;
  iter3)     do_iter3 ;;
  iter4)     do_iter4 ;;
  all)       do_all ;;
  *)
    echo "Usage: $0 {calibrate|probe|iter0|iter1|iter2|iter3|iter4|all}"
    echo ""
    echo "Environment variables:"
    echo "  MODEL=qwen/qwen3-14b   Model to simulate"
    echo "  BLIS=./blis            Path to blis binary (default: ./blis in repo root)"
    echo "  BLIS_OLD=./blis-old    Pre-PR-#901 binary for iter3 ablation"
    echo "  RATE=100               Rate for probe subcommand"
    exit 1
    ;;
esac

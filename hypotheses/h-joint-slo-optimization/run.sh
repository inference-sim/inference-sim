#!/usr/bin/env bash
# run.sh — Joint SLO Optimization experiment runner
#
# USAGE
#   ./run.sh [--rebuild] <command>
#
# COMMANDS
#   calibrate     Probe multiple rates to find saturation throughput
#   iter0         Baseline: joint compound, 3 seeds
#   iter1         Joint composition vs BLIS defaults + ablations
#   iter2         SLO-priority preemption ordering
#   iter3         Tiered-LRU KV eviction (requires BLIS_OLD env var)
#   iter4         Tier-budget batch formation + fraction sweep
#   all           iter0 → iter2 → iter4 (iter3 requires separate binary)
#
# ENVIRONMENT VARIABLES
#   BLIS        Path to implementation binary (default: auto-built from repo root)
#   BLIS_OLD    Pre-tiered-LRU binary for iter3 ablation (required for iter3)
#   MODEL       Model name (default: qwen/qwen3-14b)
#
# OUTPUT
#   results/iter{N}/{arm}/seed{S}.txt — raw blis output per run
#
# NOTE: Build the binary from the joint-slo-optimization branch, not main.
#   The blis binary on main lacks --batch-formation and tier-shed policy.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use shared harness
source "$SCRIPT_DIR/../lib/harness.sh"

# ── Configuration ────────────────────────────────────────────────────────────

MODEL="${MODEL:-qwen/qwen3-14b}"
SEEDS=(42 123 456)
WORKLOAD="$SCRIPT_DIR/workload.yaml"
RESULTS_BASE="$SCRIPT_DIR/results"
BLIS="${BLIS:-$REPO_ROOT/blis}"

# Cluster setup matching prior Strategy Evolution tracks (h-compound-strategy)
INSTANCES=4
TP=2
HARDWARE=H100
NUM_REQUESTS=1500

# Rate: set to 2× saturation. Run ./run.sh calibrate first to measure saturation.
# Prior track (h-compound-strategy): rate=300 for 120% with input_mean=256, 4 instances.
# This workload: input_mean=512 — saturation roughly halved. Start with rate=200.
RATE=200   # UPDATE after calibration: set to 2 × measured_saturation

# Policy configs
POLICY_COMPOUND="$SCRIPT_DIR/policy-compound.yaml"
POLICY_DEFAULTS="$SCRIPT_DIR/policy-defaults.yaml"

# Common flags for all runs
BASE_FLAGS=(
  --model      "$MODEL"
  --tp         $TP
  --hardware   "$HARDWARE"
  --num-instances $INSTANCES
  --num-requests  $NUM_REQUESTS
  --routing-scorers "prefix-affinity:4,queue-depth:3"
  --log error
)

# ── Helper: run one arm across all seeds ─────────────────────────────────────

run_arm() {
  local label="$1"   # e.g. "iter0/compound"
  shift
  # Remaining args are blis flags
  local out_dir="$RESULTS_BASE/$label"
  mkdir -p "$out_dir"
  for SEED in "${SEEDS[@]}"; do
    local outfile="$out_dir/seed${SEED}.txt"
    echo "  [seed $SEED] $label"
    blis_run $TIMEOUT_EXTENDED "$outfile" \
      "${BASE_FLAGS[@]}" \
      --seed "$SEED" \
      --workload-spec "$WORKLOAD" \
      "$@"
  done
  echo "  → $out_dir/"
}

# ── calibrate ─────────────────────────────────────────────────────────────────

do_calibrate() {
  echo "=== Calibration: finding saturation throughput ==="
  echo "Testing rates: 50 100 150 200 250 300 req/s"
  echo ""

  mkdir -p "$RESULTS_BASE/calibrate"
  for RATE_PROBE in 50 100 150 200 250 300; do
    local outfile="$RESULTS_BASE/calibrate/rate${RATE_PROBE}.txt"
    echo -n "  rate=$RATE_PROBE ... "
    blis_run $TIMEOUT_QUICK "$outfile" \
      "${BASE_FLAGS[@]}" \
      --seed 42 \
      --num-requests 200 \
      --workload-spec "$WORKLOAD" \
      --policy-config "$POLICY_COMPOUND" \
      --rate "$RATE_PROBE" 2>/dev/null || true

    # Parse completed vs injected
    if [[ -f "$outfile" ]]; then
      python3 - "$outfile" <<'PYEOF'
import sys, re
t = open(sys.argv[1]).read()
m_comp = re.search(r'"completed_requests":\s*(\d+)', t)
m_inj  = re.search(r'"injected_requests":\s*(\d+)', t)
m_tout = re.search(r'"timed_out_requests":\s*(\d+)', t)
comp = int(m_comp.group(1)) if m_comp else 0
inj  = int(m_inj.group(1))  if m_inj  else 0
tout = int(m_tout.group(1)) if m_tout else 0
util = f"{comp/inj*100:.0f}%" if inj > 0 else "?"
print(f"completed={comp}/{inj} ({util}), timeouts={tout}")
PYEOF
    else
      echo "NO OUTPUT"
    fi
  done

  echo ""
  echo "Find the last rate where timeouts=0 and completed=injected."
  echo "That is ~saturation. Set RATE=$(awk 'BEGIN{print 2*that}') in this script."
  echo "Also update aggregate_rate in workload.yaml to the same value."
}

# ── iter0: baseline measurement ──────────────────────────────────────────────

do_iter0() {
  echo "=== Iteration 0: Baseline measurement ==="
  run_arm "iter0/compound" \
    --policy-config "$POLICY_COMPOUND" \
    --routing-scorers "prefix-affinity:4,queue-depth:3" \
    --batch-formation vllm
  echo "Analyze: python3 $SCRIPT_DIR/analyze.py iter0 $RESULTS_BASE"
}

# ── iter1: joint composition vs BLIS defaults ─────────────────────────────────

do_iter1() {
  echo "=== Iteration 1: Joint composition validation ==="

  echo "--- H-main: compound ---"
  run_arm "iter1/compound" \
    --policy-config "$POLICY_COMPOUND" \
    --routing-scorers "prefix-affinity:4,queue-depth:3" \
    --batch-formation vllm

  echo "--- H-main: BLIS defaults ---"
  run_arm "iter1/blis-defaults" \
    --policy-config "$POLICY_DEFAULTS" \
    --batch-formation vllm

  echo "--- H-ablation: routing (round-robin, keep admission+scheduling) ---"
  run_arm "iter1/abl-routing" \
    --policy-config "$POLICY_COMPOUND" \
    --routing-policy round-robin \
    --batch-formation vllm

  echo "--- H-ablation: scheduling (fcfs, keep rest) ---"
  # Override scheduler via CLI flag (overrides policy-config)
  run_arm "iter1/abl-scheduling" \
    --policy-config "$POLICY_COMPOUND" \
    --routing-scorers "prefix-affinity:4,queue-depth:3" \
    --scheduler fcfs \
    --batch-formation vllm

  echo "--- H-ablation: admission (always-admit, keep routing+scheduling) ---"
  run_arm "iter1/abl-admission" \
    --policy-config "$POLICY_DEFAULTS" \
    --routing-policy weighted \
    --routing-scorers "prefix-affinity:4,queue-depth:3" \
    --priority-policy slo-based \
    --scheduler priority-fcfs \
    --batch-formation vllm

  echo "--- H-control-negative: uniform SLO ---"
  # Patch workload: set all slo_class to standard
  local uniform_wl
  uniform_wl=$(mktemp --suffix=.yaml)
  sed 's/slo_class: "[^"]*"/slo_class: "standard"/g' "$WORKLOAD" > "$uniform_wl"
  mkdir -p "$RESULTS_BASE/iter1/ctrl-uniform-slo"
  for SEED in "${SEEDS[@]}"; do
    echo "  [seed $SEED] ctrl-uniform-slo"
    blis_run $TIMEOUT_EXTENDED "$RESULTS_BASE/iter1/ctrl-uniform-slo/seed${SEED}.txt" \
      "${BASE_FLAGS[@]}" --seed "$SEED" --workload-spec "$uniform_wl" \
      --policy-config "$POLICY_COMPOUND" \
      --routing-scorers "prefix-affinity:4,queue-depth:3" \
      --batch-formation vllm
  done
  rm -f "$uniform_wl"

  echo "Analyze: python3 $SCRIPT_DIR/analyze.py iter1 $RESULTS_BASE"
}

# ── iter2: SLO-priority preemption ordering ───────────────────────────────────

do_iter2() {
  echo "=== Iteration 2: SLO-priority preemption ordering ==="

  echo "--- H-main: treatment (slo-priority-preemption) ---"
  run_arm "iter2/treatment" \
    --policy-config "$POLICY_COMPOUND" \
    --routing-scorers "prefix-affinity:4,queue-depth:3" \
    --batch-formation slo-priority-preemption

  echo "--- H-main: ablation (vllm/LIFO) ---"
  run_arm "iter2/ablation" \
    --policy-config "$POLICY_COMPOUND" \
    --routing-scorers "prefix-affinity:4,queue-depth:3" \
    --batch-formation vllm

  echo "Analyze: python3 $SCRIPT_DIR/analyze.py iter2 $RESULTS_BASE"
}

# ── iter3: tiered LRU (requires two binary builds) ────────────────────────────

do_iter3() {
  echo "=== Iteration 3: Tiered-LRU KV eviction ==="
  echo ""

  BLIS_OLD="${BLIS_OLD:-}"
  if [[ -z "$BLIS_OLD" ]]; then
    echo "ERROR: BLIS_OLD not set. Build a pre-PR-#901 binary and export BLIS_OLD=<path>"
    echo "  git -C $REPO_ROOT checkout <pre-PR-901-sha> -- sim/kv/cache.go"
    echo "  go build -o /tmp/blis-old $REPO_ROOT/main.go"
    echo "  export BLIS_OLD=/tmp/blis-old"
    exit 1
  fi

  echo "--- Treatment: tiered-LRU build (slo-priority-preemption + tiered-LRU) ---"
  run_arm "iter3/treatment" \
    --policy-config "$POLICY_COMPOUND" \
    --routing-scorers "prefix-affinity:4,queue-depth:3" \
    --batch-formation slo-priority-preemption

  echo "--- Ablation: single-list LRU build (pre-PR-#901) ---"
  mkdir -p "$RESULTS_BASE/iter3/ablation"
  local saved_blis="$BINARY"
  BINARY="$BLIS_OLD"
  for SEED in "${SEEDS[@]}"; do
    echo "  [seed $SEED] single-list-lru (BLIS_OLD)"
    blis_run $TIMEOUT_EXTENDED "$RESULTS_BASE/iter3/ablation/seed${SEED}.txt" \
      --model "$MODEL" --tp $TP --hardware "$HARDWARE" --num-instances $INSTANCES \
      --num-requests $NUM_REQUESTS --seed "$SEED" \
      --workload-spec "$WORKLOAD" \
      --admission-policy always-admit \
      --routing-policy weighted \
      --routing-scorers "prefix-affinity:4,queue-depth:3" \
      --priority-policy slo-based \
      --scheduler priority-fcfs \
      --batch-formation vllm \
      --log error
  done
  BINARY="$saved_blis"

  echo "Analyze: python3 $SCRIPT_DIR/analyze.py iter3 $RESULTS_BASE"
}

# ── iter4: tier-budget batch formation ────────────────────────────────────────

do_iter4() {
  echo "=== Iteration 4: Admission-feedback batch formation ==="

  echo "--- H-main: tier-budget f_c=0.50 ---"
  run_arm "iter4/treatment" \
    --policy-config "$POLICY_COMPOUND" \
    --routing-scorers "prefix-affinity:4,queue-depth:3" \
    --batch-formation tier-budget \
    --tier-budget-critical-frac 0.50 \
    --tier-budget-standard-frac 0.70

  echo "--- H-ablation: equal-share f_c=0.333 ---"
  run_arm "iter4/abl-equal-share" \
    --policy-config "$POLICY_COMPOUND" \
    --routing-scorers "prefix-affinity:4,queue-depth:3" \
    --batch-formation tier-budget \
    --tier-budget-critical-frac 0.333 \
    --tier-budget-standard-frac 0.50

  echo "--- H-robustness: fraction sweep ---"
  for FC in 0.20 0.30 0.40 0.50 0.60 0.70; do
    run_arm "iter4/sweep-fc${FC/./}" \
      --policy-config "$POLICY_COMPOUND" \
      --routing-scorers "prefix-affinity:4,queue-depth:3" \
      --batch-formation tier-budget \
      --tier-budget-critical-frac "$FC" \
      --tier-budget-standard-frac 0.70
  done

  echo "Analyze: python3 $SCRIPT_DIR/analyze.py iter4 $RESULTS_BASE"
}

# ── all ───────────────────────────────────────────────────────────────────────

do_all() {
  do_iter0
  do_iter1
  do_iter2
  echo "(Skipping iter3 — requires BLIS_OLD; run manually with: BLIS_OLD=<path> ./run.sh iter3)"
  do_iter4
  echo ""
  echo "All iterations complete. Run: python3 $SCRIPT_DIR/analyze.py all $RESULTS_BASE"
}

# ── Build BINARY if needed ────────────────────────────────────────────────────

# Override harness BINARY with implementation branch binary
# The joint-slo-optimization worktree has --batch-formation; main does not.
IMPL_WORKTREE="$REPO_ROOT/.worktrees/joint-slo-optimization"
if [[ -x "$IMPL_WORKTREE/blis" ]]; then
  BINARY="$IMPL_WORKTREE/blis"
elif [[ -x "$REPO_ROOT/blis" ]]; then
  BINARY="$REPO_ROOT/blis"
else
  echo "Building blis from implementation branch..."
  if [[ -d "$IMPL_WORKTREE" ]]; then
    (cd "$IMPL_WORKTREE" && go build -o blis main.go)
    BINARY="$IMPL_WORKTREE/blis"
  else
    (cd "$REPO_ROOT" && go build -o blis main.go)
    BINARY="$REPO_ROOT/blis"
  fi
fi
echo "Using binary: $BINARY" >&2

# ── Dispatch ──────────────────────────────────────────────────────────────────

# Parse --rebuild flag
REBUILD=""
if [[ "${1:-}" == "--rebuild" ]]; then
  REBUILD="--rebuild"
  shift
fi
setup_experiment "$REBUILD"

CMD="${1:-help}"
case "$CMD" in
  calibrate) do_calibrate ;;
  iter0)     do_iter0 ;;
  iter1)     do_iter1 ;;
  iter2)     do_iter2 ;;
  iter3)     do_iter3 ;;
  iter4)     do_iter4 ;;
  all)       do_all ;;
  *)
    echo "Usage: $0 [--rebuild] {calibrate|iter0|iter1|iter2|iter3|iter4|all}"
    echo ""
    echo "Environment variables:"
    echo "  BLIS=<path>      Override blis binary (default: auto-detected from worktree)"
    echo "  BLIS_OLD=<path>  Pre-PR-#901 binary for iter3 ablation"
    echo "  MODEL=<name>     Model name (default: qwen/qwen3-14b)"
    exit 1
    ;;
esac

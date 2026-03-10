#!/bin/bash
# Strategy Evolution: Deadline-Aware SLO Scheduling
#
# Tests DeadlineAwarePriority (hyperbolic urgency from per-SLO-class TTFT deadlines)
# against three baselines: B0 (FCFS), B1 (age-only), B2 (static class weights).
#
# 6 hypothesis arms:
#   H-main:           B0, B1, B2, Treatment at 30%, 80%, 120% capacity x 3 seeds (36 runs)
#   H-ablation:       Treatment vs Treatment-uniform-deadline at 80%, 120% x 3 seeds (12 runs)
#   H-zero-sum:       Uses H-main data at 120% (no extra runs)
#   H-control-neg:    Treatment-uniform-SLO vs Treatment-differentiated at 30% x 3 seeds (6 runs)
#   H-robustness:     Treatment vs B2 at CV=1.5, 2.0, 3.5 at 120% x 3 seeds (18 runs)
#   H-single-turn:    Treatment vs B2 single-turn at 120% x 3 seeds (6 runs)
#
# Total: ~78 unique simulation runs
#
# Usage: ./run.sh [--rebuild]
#   --rebuild  Force rebuild of the binary

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"

setup_experiment "${1:-}"

# Must run from repo root so defaults.yaml is found by the simulator
cd "$REPO_ROOT"

# ── Parameters ──────────────────────────────────────────────────────────────
# Capacity estimate: ~250 req/s for 4 instances (conservative, accounting for
# multi-turn context accumulation with moderate caching).
# Rate sweep: 30% (~75), 80% (~200), 120% (~300)
RATE_30=75
RATE_80=200
RATE_120=300
NUM_REQUESTS=1500
NUM_INSTANCES=4
SEEDS=(42 123 456)

# Common CLI flags for all runs
COMMON_FLAGS=(
    --model "$MODEL"
    --tp 2
    --hardware H100
    --num-instances $NUM_INSTANCES
    --num-requests $NUM_REQUESTS
    --routing-policy weighted
    --routing-scorers "prefix-affinity:3,queue-depth:2"
    --log error
)

# ── Policy config files (written to RESULTS_DIR) ───────────────────────────

# B2: Static class weight policy bundle
B2_CONFIG="$RESULTS_DIR/b2_policy.yaml"
cat > "$B2_CONFIG" <<'EOF'
priority:
  policy: static-class-weight
  class_weights:
    critical: 10.0
    standard: 5.0
    sheddable: 1.0
EOF

# Treatment: Deadline-aware priority policy bundle
TREATMENT_CONFIG="$RESULTS_DIR/treatment_policy.yaml"
cat > "$TREATMENT_CONFIG" <<'EOF'
priority:
  policy: deadline-aware
  class_weights:
    critical: 10.0
    standard: 5.0
    sheddable: 1.0
  deadlines:
    critical: 100000
    standard: 500000
    sheddable: 2000000
  epsilon: 0.01
EOF

# Ablation: Uniform deadline (all 500ms)
ABLATION_UNIFORM_DL_CONFIG="$RESULTS_DIR/ablation_uniform_dl_policy.yaml"
cat > "$ABLATION_UNIFORM_DL_CONFIG" <<'EOF'
priority:
  policy: deadline-aware
  class_weights:
    critical: 10.0
    standard: 5.0
    sheddable: 1.0
  deadlines:
    critical: 500000
    standard: 500000
    sheddable: 500000
  epsilon: 0.01
EOF

# ── Workload YAML generation ────────────────────────────────────────────────
# Generate workload spec YAML for a given rate, seed, CV, and SLO configuration.
# Args: $1=rate, $2=seed, $3=cv, $4=slo_mode (mixed|uniform), $5=multi_turn (yes|no)
generate_workload() {
    local rate="$1"
    local seed="$2"
    local cv="$3"
    local slo_mode="$4"
    local multi_turn="$5"
    local outfile="$6"

    local crit_class="critical"
    local std_class="standard"
    local shed_class="sheddable"
    if [[ "$slo_mode" == "uniform" ]]; then
        crit_class="standard"
        std_class="standard"
        shed_class="standard"
    fi

    local mt_block=""
    if [[ "$multi_turn" == "yes" ]]; then
        mt_block="    reasoning:
      multi_turn:
        max_rounds: 3
        think_time_us: 500000
        context_growth: accumulate"
    fi

    cat > "$outfile" <<YAMLEOF
version: "2"
aggregate_rate: ${rate}
seed: ${seed}
num_requests: ${NUM_REQUESTS}
clients:
  - id: critical-client
    rate_fraction: 0.2
    slo_class: ${crit_class}
    arrival:
      process: gamma
      cv: ${cv}
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 64
        min: 32
        max: 1024
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 32
        min: 16
        max: 512
${mt_block:+${mt_block}}
  - id: standard-client
    rate_fraction: 0.4
    slo_class: ${std_class}
    arrival:
      process: gamma
      cv: ${cv}
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 64
        min: 32
        max: 1024
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 32
        min: 16
        max: 512
${mt_block:+${mt_block}}
  - id: sheddable-client
    rate_fraction: 0.4
    slo_class: ${shed_class}
    arrival:
      process: gamma
      cv: ${cv}
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 64
        min: 32
        max: 1024
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 32
        min: 16
        max: 512
${mt_block:+${mt_block}}
YAMLEOF
}

# ── Output directory structure ──────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"/{h-main,h-ablation,h-control-neg,h-robustness,h-single-turn}

echo "============================================================================"
echo "  Strategy Evolution: Deadline-Aware SLO Scheduling"
echo "  Rates: ${RATE_30} (30%), ${RATE_80} (80%), ${RATE_120} (120%) req/s"
echo "  Seeds: ${SEEDS[*]}"
echo "  Requests: ${NUM_REQUESTS} per run"
echo "============================================================================"
echo ""

# Track run count for progress
RUN=0
TOTAL_RUNS=78

progress() {
    RUN=$((RUN + 1))
    echo "  [${RUN}/${TOTAL_RUNS}] $1"
}

# ── H-main: B0, B1, B2, Treatment at 30%, 80%, 120% x 3 seeds ─────────────
echo "=== H-main: Core comparison (36 runs) ==="
echo ""

for RATE_LABEL in 30 80 120; do
    eval "RATE=\$RATE_${RATE_LABEL}"
    for SEED in "${SEEDS[@]}"; do
        # Generate workload YAML (shared across all configs at this rate/seed)
        WORKLOAD_FILE="$RESULTS_DIR/h-main/workload_r${RATE_LABEL}_s${SEED}.yaml"
        generate_workload "$RATE" "$SEED" "2.0" "mixed" "yes" "$WORKLOAD_FILE"

        # B0: FCFS + Constant priority
        progress "H-main B0 rate=${RATE_LABEL}% seed=${SEED}"
        blis_run $TIMEOUT_EXTENDED "$RESULTS_DIR/h-main/b0_r${RATE_LABEL}_s${SEED}.txt" \
            "${COMMON_FLAGS[@]}" \
            --workload-spec "$WORKLOAD_FILE" \
            --scheduler fcfs \
            --priority-policy constant \
            --seed "$SEED" || true

        # B1: PriorityFCFS + SLO-based (age-only)
        progress "H-main B1 rate=${RATE_LABEL}% seed=${SEED}"
        blis_run $TIMEOUT_EXTENDED "$RESULTS_DIR/h-main/b1_r${RATE_LABEL}_s${SEED}.txt" \
            "${COMMON_FLAGS[@]}" \
            --workload-spec "$WORKLOAD_FILE" \
            --scheduler priority-fcfs \
            --priority-policy slo-based \
            --seed "$SEED" || true

        # B2: PriorityFCFS + Static class weight
        # --scheduler required because bundle only sets priority, not scheduler
        progress "H-main B2 rate=${RATE_LABEL}% seed=${SEED}"
        blis_run $TIMEOUT_EXTENDED "$RESULTS_DIR/h-main/b2_r${RATE_LABEL}_s${SEED}.txt" \
            "${COMMON_FLAGS[@]}" \
            --workload-spec "$WORKLOAD_FILE" \
            --scheduler priority-fcfs \
            --policy-config "$B2_CONFIG" \
            --seed "$SEED" || true

        # Treatment: PriorityFCFS + Deadline-aware
        # --scheduler required because bundle only sets priority, not scheduler
        progress "H-main Treatment rate=${RATE_LABEL}% seed=${SEED}"
        blis_run $TIMEOUT_EXTENDED "$RESULTS_DIR/h-main/treatment_r${RATE_LABEL}_s${SEED}.txt" \
            "${COMMON_FLAGS[@]}" \
            --workload-spec "$WORKLOAD_FILE" \
            --scheduler priority-fcfs \
            --policy-config "$TREATMENT_CONFIG" \
            --seed "$SEED" || true
    done
done
echo ""

# ── H-ablation: Treatment vs Uniform-deadline at 80%, 120% x 3 seeds ───────
echo "=== H-ablation: Per-class deadline differentiation (12 runs) ==="
echo ""

for RATE_LABEL in 80 120; do
    eval "RATE=\$RATE_${RATE_LABEL}"
    for SEED in "${SEEDS[@]}"; do
        WORKLOAD_FILE="$RESULTS_DIR/h-ablation/workload_r${RATE_LABEL}_s${SEED}.yaml"
        generate_workload "$RATE" "$SEED" "2.0" "mixed" "yes" "$WORKLOAD_FILE"

        # Treatment (differentiated deadlines) — reuse from h-main if same rate
        progress "H-ablation differentiated rate=${RATE_LABEL}% seed=${SEED}"
        blis_run $TIMEOUT_EXTENDED "$RESULTS_DIR/h-ablation/diff_r${RATE_LABEL}_s${SEED}.txt" \
            "${COMMON_FLAGS[@]}" \
            --workload-spec "$WORKLOAD_FILE" \
            --scheduler priority-fcfs \
            --policy-config "$TREATMENT_CONFIG" \
            --seed "$SEED" || true

        # Ablation: Uniform deadline (500ms for all classes)
        progress "H-ablation uniform-deadline rate=${RATE_LABEL}% seed=${SEED}"
        blis_run $TIMEOUT_EXTENDED "$RESULTS_DIR/h-ablation/uniform_dl_r${RATE_LABEL}_s${SEED}.txt" \
            "${COMMON_FLAGS[@]}" \
            --workload-spec "$WORKLOAD_FILE" \
            --scheduler priority-fcfs \
            --policy-config "$ABLATION_UNIFORM_DL_CONFIG" \
            --seed "$SEED" || true
    done
done
echo ""

# ── H-control-negative: Uniform SLO vs Differentiated at 30% x 3 seeds ─────
echo "=== H-control-negative: Mechanism specificity (6 runs) ==="
echo ""

for SEED in "${SEEDS[@]}"; do
    # Uniform SLO workload (all clients labeled 'standard')
    UNIFORM_WORKLOAD="$RESULTS_DIR/h-control-neg/workload_uniform_s${SEED}.yaml"
    generate_workload "$RATE_30" "$SEED" "2.0" "uniform" "yes" "$UNIFORM_WORKLOAD"

    # Differentiated SLO workload (mixed classes)
    DIFF_WORKLOAD="$RESULTS_DIR/h-control-neg/workload_diff_s${SEED}.yaml"
    generate_workload "$RATE_30" "$SEED" "2.0" "mixed" "yes" "$DIFF_WORKLOAD"

    # Treatment with uniform SLO labels
    progress "H-control-neg uniform-SLO seed=${SEED}"
    blis_run $TIMEOUT_EXTENDED "$RESULTS_DIR/h-control-neg/uniform_slo_s${SEED}.txt" \
        "${COMMON_FLAGS[@]}" \
        --workload-spec "$UNIFORM_WORKLOAD" \
        --scheduler priority-fcfs \
        --policy-config "$TREATMENT_CONFIG" \
        --seed "$SEED" || true

    # Treatment with differentiated SLO labels
    progress "H-control-neg diff-SLO seed=${SEED}"
    blis_run $TIMEOUT_EXTENDED "$RESULTS_DIR/h-control-neg/diff_slo_s${SEED}.txt" \
        "${COMMON_FLAGS[@]}" \
        --workload-spec "$DIFF_WORKLOAD" \
        --scheduler priority-fcfs \
        --policy-config "$TREATMENT_CONFIG" \
        --seed "$SEED" || true
done
echo ""

# ── H-robustness-burst: Treatment vs B2 at CV=1.5, 2.0, 3.5 at 120% x 3 seeds
echo "=== H-robustness-burst: Burst intensity scaling (18 runs) ==="
echo ""

for CV in 1.5 2.0 3.5; do
    # Sanitize CV for filename (replace . with p)
    CV_LABEL=$(echo "$CV" | tr '.' 'p')
    for SEED in "${SEEDS[@]}"; do
        WORKLOAD_FILE="$RESULTS_DIR/h-robustness/workload_cv${CV_LABEL}_s${SEED}.yaml"
        generate_workload "$RATE_120" "$SEED" "$CV" "mixed" "yes" "$WORKLOAD_FILE"

        # Treatment
        progress "H-robustness Treatment CV=${CV} seed=${SEED}"
        blis_run $TIMEOUT_EXTENDED "$RESULTS_DIR/h-robustness/treatment_cv${CV_LABEL}_s${SEED}.txt" \
            "${COMMON_FLAGS[@]}" \
            --workload-spec "$WORKLOAD_FILE" \
            --scheduler priority-fcfs \
            --policy-config "$TREATMENT_CONFIG" \
            --seed "$SEED" || true

        # B2: Static class weight
        progress "H-robustness B2 CV=${CV} seed=${SEED}"
        blis_run $TIMEOUT_EXTENDED "$RESULTS_DIR/h-robustness/b2_cv${CV_LABEL}_s${SEED}.txt" \
            "${COMMON_FLAGS[@]}" \
            --workload-spec "$WORKLOAD_FILE" \
            --scheduler priority-fcfs \
            --policy-config "$B2_CONFIG" \
            --seed "$SEED" || true
    done
done
echo ""

# ── H-single-turn: Treatment vs B2 single-turn at 120% x 3 seeds ───────────
echo "=== H-single-turn: Multi-turn confound isolation (6 runs) ==="
echo ""

for SEED in "${SEEDS[@]}"; do
    WORKLOAD_FILE="$RESULTS_DIR/h-single-turn/workload_s${SEED}.yaml"
    generate_workload "$RATE_120" "$SEED" "2.0" "mixed" "no" "$WORKLOAD_FILE"

    # Treatment
    progress "H-single-turn Treatment seed=${SEED}"
    blis_run $TIMEOUT_EXTENDED "$RESULTS_DIR/h-single-turn/treatment_s${SEED}.txt" \
        "${COMMON_FLAGS[@]}" \
        --workload-spec "$WORKLOAD_FILE" \
        --scheduler priority-fcfs \
        --policy-config "$TREATMENT_CONFIG" \
        --seed "$SEED" || true

    # B2: Static class weight
    progress "H-single-turn B2 seed=${SEED}"
    blis_run $TIMEOUT_EXTENDED "$RESULTS_DIR/h-single-turn/b2_s${SEED}.txt" \
        "${COMMON_FLAGS[@]}" \
        --workload-spec "$WORKLOAD_FILE" \
        --scheduler priority-fcfs \
        --policy-config "$B2_CONFIG" \
        --seed "$SEED" || true
done
echo ""

# ── Copy results to persistent location ─────────────────────────────────────
PERSIST_DIR="$SCRIPT_DIR/results"
echo "============================================================================"
echo "  Copying results to $PERSIST_DIR"
echo "============================================================================"
rm -rf "$PERSIST_DIR"
cp -r "$RESULTS_DIR" "$PERSIST_DIR"

echo ""
echo "  Total runs: ${RUN}"
echo ""

# ── Analysis ────────────────────────────────────────────────────────────────
echo "============================================================================"
echo "  Analysis"
echo "============================================================================"
echo ""

python3 "$SCRIPT_DIR/analyze.py" "$PERSIST_DIR"

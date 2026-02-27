#!/bin/bash
# H-Step-Quantum: Does reducing step-time quantum shrink DES-to-M/M/1 wait-time divergence?
#
# Hypothesis: Reducing the DES step-time quantum (by scaling beta coefficients)
# should proportionally reduce the DES-to-M/M/1 mean wait time divergence.
# At rho=0.7, the W_q error (currently ~60% with ~6.9ms steps) should scale
# linearly with step_time / mean_service_time, approaching 0% as step time -> 0.
#
# Classification: Statistical / Monotonicity
# Family: Structural model
# VV&UQ: Validation
#
# Design:
#   - Three beta coefficient scalings: 1x (baseline), 0.1x, 0.01x
#   - Alpha coefficients held constant (controls overhead, not step quantum)
#   - Single instance (k=1, M/M/1 comparison — cleanest)
#   - Utilization sweep: rho = {0.3, 0.5, 0.7, 0.9}
#   - Each beta scaling requires re-calibration of mu (service rate)
#   - Workload: Poisson arrivals, constant input=1, exponential output mean=128
#   - Seeds: 42, 123, 456
#
# ED-1: Controlled comparison — only beta coefficients vary between configurations
# ED-2: Rate calibrated per beta scaling from empirical service time
# ED-3: Preconditions — stability (rho < 1), constant output for calibration
# ED-5: Reproducible — builds binary, runs all variants, no manual steps
# ED-6: Reference: hypotheses/h-mmk-validation/run.sh
#   Config diff: beta-coeffs and alpha-coeffs now explicit (h-mmk used defaults);
#   all other flags identical to h-mmk sub-experiment 1 (k=1, fcfs, always-admit)
#
# Reference: https://github.com/inference-sim/inference-sim/issues/329
#
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$REPO_ROOT/blis"

# Build if needed
if [[ "${1:-}" == "--rebuild" ]] || [[ ! -x "$BINARY" ]]; then
    echo "Building blis..."
    (cd "$REPO_ROOT" && go build -o blis main.go)
fi

MODEL="meta-llama/llama-3.1-8b-instruct"
SEEDS=(42 123 456)
RHOS=(0.3 0.5 0.7 0.9)
NUM_REQUESTS=2000
OUTPUT_TOKEN_MEAN=128
INPUT_TOKENS=1

# Default alpha coefficients (held constant across all beta scalings)
ALPHA_COEFFS="1601.35,3.51,1805.54"

# Beta coefficient scalings to test
# Baseline: [6910.42, 17.67, 2.84] → step_time ≈ 6913 us per decode step
# 0.1x:    [691.042, 1.767, 0.284] → step_time ≈ 691 us per decode step
# 0.01x:   [69.1042, 0.1767, 0.0284] → step_time ≈ 69 us per decode step
BETA_SCALES=(1.0 0.1 0.01)
BETA_BASE_0=6910.42
BETA_BASE_1=17.67
BETA_BASE_2=2.84

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# Generate workload YAML with a given rate
make_workload() {
    local rate=$1
    local seed=$2
    local outfile=$3

    cat > "$outfile" << YAMLEOF
version: "1"
seed: $seed
category: language
aggregate_rate: $rate
num_requests: $NUM_REQUESTS
clients:
  - id: "step-quantum-client"
    tenant_id: "default"
    slo_class: "batch"
    rate_fraction: 1.0
    streaming: false
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: $INPUT_TOKENS
    output_distribution:
      type: exponential
      params:
        mean: $OUTPUT_TOKEN_MEAN
YAMLEOF
}

echo "============================================================================"
echo "  H-Step-Quantum: Step-Time Quantum vs DES-M/M/1 Divergence"
echo "  Reference: issue #329, H-MMK findings (PR #325)"
echo "  Type: Statistical / Monotonicity"
echo "  Family: Structural model | VV&UQ: Validation"
echo "============================================================================"
echo ""

# ── Step 0: Calibrate mean service time for each beta scaling ─────────────
# Use constant output and very low load to measure service time precisely.

echo "Step 0: Calibrating service times for each beta scaling..."
echo ""

# Store calibration results in files (avoid bash 4+ associative arrays)
CAL_DIR="$RESULTS_DIR/calibration"
mkdir -p "$CAL_DIR"

for scale in "${BETA_SCALES[@]}"; do
    BETA_0=$(python3 -c "print(f'{$BETA_BASE_0 * $scale:.4f}')")
    BETA_1=$(python3 -c "print(f'{$BETA_BASE_1 * $scale:.4f}')")
    BETA_2=$(python3 -c "print(f'{$BETA_BASE_2 * $scale:.4f}')")
    BETA_COEFFS="$BETA_0,$BETA_1,$BETA_2"

    # Compute expected step time per decode step (single request)
    STEP_US=$(python3 -c "print(f'{$BETA_BASE_0 * $scale + $BETA_BASE_2 * $scale:.2f}')")

    echo "  Beta scale ${scale}x: beta=[$BETA_COEFFS], expected step_time=${STEP_US} us"

    cat > "$RESULTS_DIR/cal_wl_${scale}.yaml" << YAMLEOF
version: "1"
seed: 42
category: language
aggregate_rate: 0.01
num_requests: 10
clients:
  - id: "calibrate"
    tenant_id: "default"
    slo_class: "batch"
    rate_fraction: 1.0
    streaming: false
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: $INPUT_TOKENS
    output_distribution:
      type: constant
      params:
        value: $OUTPUT_TOKEN_MEAN
YAMLEOF

    timeout 120 "$BINARY" run \
        --model "$MODEL" \
        --num-instances 1 \
        --max-num-running-reqs 1 \
        --workload-spec "$RESULTS_DIR/cal_wl_${scale}.yaml" \
        --seed 42 \
        --scheduler fcfs \
        --admission-policy always-admit \
        --total-kv-blocks 1000000 \
        --beta-coeffs "$BETA_COEFFS" \
        --alpha-coeffs "$ALPHA_COEFFS" \
        --log error \
        --results-path "$RESULTS_DIR/cal_${scale}.json" \
        2>/dev/null \
        > "$RESULTS_DIR/cal_${scale}_stdout.txt"

    MEAN_SERVICE_MS=$(python3 -c "
import json
data = json.load(open('$RESULTS_DIR/cal_${scale}.json'))
reqs = [r for r in data['requests'] if r['e2e_ms'] > 0]
mean_e2e = sum(r['e2e_ms'] for r in reqs) / len(reqs)
print(f'{mean_e2e:.3f}')
")
    MU=$(python3 -c "print(f'{1000.0 / $MEAN_SERVICE_MS:.6f}')")

    echo "    Mean service time: ${MEAN_SERVICE_MS} ms, mu = ${MU} req/s"

    # Store calibration results in files
    echo "$MU" > "$CAL_DIR/mu_${scale}"
    echo "$MEAN_SERVICE_MS" > "$CAL_DIR/svc_${scale}"
    echo "$STEP_US" > "$CAL_DIR/step_${scale}"
done

echo ""

# ── Main experiment: M/M/1 comparison at each beta scaling ────────────────

for scale in "${BETA_SCALES[@]}"; do
    BETA_0=$(python3 -c "print(f'{$BETA_BASE_0 * $scale:.4f}')")
    BETA_1=$(python3 -c "print(f'{$BETA_BASE_1 * $scale:.4f}')")
    BETA_2=$(python3 -c "print(f'{$BETA_BASE_2 * $scale:.4f}')")
    BETA_COEFFS="$BETA_0,$BETA_1,$BETA_2"
    MU=$(cat "$CAL_DIR/mu_${scale}")
    SVC_MS=$(cat "$CAL_DIR/svc_${scale}")
    STEP_US=$(cat "$CAL_DIR/step_${scale}")

    echo "============================================================================"
    echo "  Beta scale ${scale}x: step_time=${STEP_US} us"
    echo "    mu=${MU} req/s, service_time=${SVC_MS} ms"
    echo "    beta=[$BETA_COEFFS], alpha=[$ALPHA_COEFFS]"
    echo "============================================================================"
    echo ""

    for rho in "${RHOS[@]}"; do
        RATE=$(python3 -c "print(f'{$rho * $MU:.6f}')")
        for seed in "${SEEDS[@]}"; do
            echo "  Running: scale=${scale} rho=$rho rate=$RATE seed=$seed ..."
            make_workload "$RATE" "$seed" "$RESULTS_DIR/wl_s${scale}_r${rho}_s${seed}.yaml"
            timeout 300 "$BINARY" run \
                --model "$MODEL" \
                --num-instances 1 \
                --max-num-running-reqs 1 \
                --workload-spec "$RESULTS_DIR/wl_s${scale}_r${rho}_s${seed}.yaml" \
                --seed "$seed" \
                --scheduler fcfs \
                --admission-policy always-admit \
                --total-kv-blocks 1000000 \
                --beta-coeffs "$BETA_COEFFS" \
                --alpha-coeffs "$ALPHA_COEFFS" \
                --log error \
                --results-path "$RESULTS_DIR/s${scale}_r${rho}_s${seed}.json" \
                2>/dev/null \
                > "$RESULTS_DIR/s${scale}_r${rho}_s${seed}_stdout.txt" \
                || echo "    WARNING: timeout or error for scale=$scale rho=$rho seed=$seed"
        done
    done

    echo ""
done

echo "============================================================================"
echo "  Analysis"
echo "============================================================================"
echo ""

# Read calibration data from files
MU_1=$(cat "$CAL_DIR/mu_1.0")
MU_01=$(cat "$CAL_DIR/mu_0.1")
MU_001=$(cat "$CAL_DIR/mu_0.01")
SVC_1=$(cat "$CAL_DIR/svc_1.0")
SVC_01=$(cat "$CAL_DIR/svc_0.1")
SVC_001=$(cat "$CAL_DIR/svc_0.01")
STEP_1=$(cat "$CAL_DIR/step_1.0")
STEP_01=$(cat "$CAL_DIR/step_0.1")
STEP_001=$(cat "$CAL_DIR/step_0.01")

# Pass calibration data to analyzer
python3 "$SCRIPT_DIR/analyze.py" \
    --results-dir "$RESULTS_DIR" \
    --beta-scales "1.0,0.1,0.01" \
    --mu-values "${MU_1},${MU_01},${MU_001}" \
    --service-ms-values "${SVC_1},${SVC_01},${SVC_001}" \
    --step-us-values "${STEP_1},${STEP_01},${STEP_001}" \
    --num-requests "$NUM_REQUESTS"

echo ""

# ── Control experiment: alpha=0 (Round 2, RCV-4) ──────────────────────────
# Disables alpha overhead to confirm it is the root cause of divergence.
# With alpha=0, E2E = step_total, so M/M/1 comparison uses the correct mu.

echo "============================================================================"
echo "  Control Experiment: alpha=[0,0,0] with beta at 1.0x"
echo "  Purpose: confirm alpha/beta split as root cause of divergence"
echo "============================================================================"
echo ""

# Calibrate with alpha=0
ALPHA_ZERO="0,0,0"
BETA_BASELINE="$BETA_BASE_0,$BETA_BASE_1,$BETA_BASE_2"

cat > "$RESULTS_DIR/cal_wl_a0.yaml" << YAMLEOF
version: "1"
seed: 42
category: language
aggregate_rate: 0.01
num_requests: 10
clients:
  - id: "calibrate"
    tenant_id: "default"
    slo_class: "batch"
    rate_fraction: 1.0
    streaming: false
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: $INPUT_TOKENS
    output_distribution:
      type: constant
      params:
        value: $OUTPUT_TOKEN_MEAN
YAMLEOF

timeout 120 "$BINARY" run \
    --model "$MODEL" \
    --num-instances 1 \
    --max-num-running-reqs 1 \
    --workload-spec "$RESULTS_DIR/cal_wl_a0.yaml" \
    --seed 42 \
    --scheduler fcfs \
    --admission-policy always-admit \
    --total-kv-blocks 1000000 \
    --beta-coeffs "$BETA_BASELINE" \
    --alpha-coeffs "$ALPHA_ZERO" \
    --log error \
    --results-path "$RESULTS_DIR/cal_a0.json" \
    2>/dev/null \
    > "$RESULTS_DIR/cal_a0_stdout.txt"

MU_A0=$(python3 -c "
import json
data = json.load(open('$RESULTS_DIR/cal_a0.json'))
reqs = [r for r in data['requests'] if r['e2e_ms'] > 0]
mean_e2e = sum(r['e2e_ms'] for r in reqs) / len(reqs)
print(f'{1000.0 / mean_e2e:.6f}')
")
SVC_A0=$(python3 -c "
import json
data = json.load(open('$RESULTS_DIR/cal_a0.json'))
reqs = [r for r in data['requests'] if r['e2e_ms'] > 0]
mean_e2e = sum(r['e2e_ms'] for r in reqs) / len(reqs)
print(f'{mean_e2e:.3f}')
")
echo "  Alpha=0 calibration: service_time=${SVC_A0} ms, mu=${MU_A0} req/s"
echo ""

for rho in "${RHOS[@]}"; do
    RATE=$(python3 -c "print(f'{$rho * $MU_A0:.6f}')")
    for seed in "${SEEDS[@]}"; do
        echo "  Running: alpha=0 rho=$rho rate=$RATE seed=$seed ..."
        make_workload "$RATE" "$seed" "$RESULTS_DIR/wl_a0_r${rho}_s${seed}.yaml"
        timeout 300 "$BINARY" run \
            --model "$MODEL" \
            --num-instances 1 \
            --max-num-running-reqs 1 \
            --workload-spec "$RESULTS_DIR/wl_a0_r${rho}_s${seed}.yaml" \
            --seed "$seed" \
            --scheduler fcfs \
            --admission-policy always-admit \
            --total-kv-blocks 1000000 \
            --beta-coeffs "$BETA_BASELINE" \
            --alpha-coeffs "$ALPHA_ZERO" \
            --log error \
            --results-path "$RESULTS_DIR/a0_r${rho}_s${seed}.json" \
            2>/dev/null \
            > "$RESULTS_DIR/a0_r${rho}_s${seed}_stdout.txt" \
            || echo "    WARNING: timeout or error for alpha=0 rho=$rho seed=$seed"
    done
done

echo ""
echo "  Alpha=0 control results:"
python3 -c "
import json, os

mu = $MU_A0
rhos = [0.3, 0.5, 0.7, 0.9]
seeds = [42, 123, 456]

def mm1_wq(lam, mu):
    rho = lam / mu
    if rho >= 1: return float('inf')
    return rho / (mu * (1 - rho))

print(f'  {\"rho\":<8} {\"W_q Ana (ms)\":>14} {\"W_q DES (ms)\":>14} {\"Error\":>10}')
print(f'  {\"-\"*8} {\"-\"*14} {\"-\"*14} {\"-\"*10}')

for rho in rhos:
    lam = rho * mu
    wq_ana = mm1_wq(lam, mu) * 1000
    waits = []
    for seed in seeds:
        fpath = os.path.join('$RESULTS_DIR', f'a0_r{rho}_s{seed}.json')
        if not os.path.exists(fpath):
            continue
        data = json.load(open(fpath))
        completed = [r for r in data['requests'] if r['e2e_ms'] > 0]
        waits.extend([r['scheduling_delay_ms'] / 1000.0 for r in completed])
    if waits:
        wq_des = sum(waits) / len(waits)
        err = (wq_des - wq_ana) / wq_ana * 100 if wq_ana > 0 else 0
        print(f'  {rho:<8} {wq_ana:>14.2f} {wq_des:>14.2f} {err:>+9.1f}%')
"

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis"
echo "============================================================================"

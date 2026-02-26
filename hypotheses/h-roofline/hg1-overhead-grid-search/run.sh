#!/bin/bash
# HG1: perLayerOverhead Calibration via Grid Search with Train/Test Split
#
# Hypothesis: A data-driven grid search over perLayerOverhead (0-500μs) can
# find a value that generalizes across model families without overfitting.
# The analytical 100μs/layer from H2b overshoots for 7B-TP1.
#
# Design: Two-phase sweep with 9-train / 4-test split.
#   Phase 1 (coarse): 0 to 500μs in 25μs steps → 21 values
#   Phase 2 (fine): ±25μs around coarse optimum in 5μs steps → 11 values
#   Validation: fine optimum against 4 test experiments
#
# All runs include H1 correction (bwEfficiencyFactor=0.82).
#
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/../../lib/harness.sh"
setup_experiment "${1:-}"

GT_DIR="$REPO_ROOT/eval/ground_truth"
MODEL_CONFIGS="$REPO_ROOT/model_configs"
BENCH_DATA="$REPO_ROOT/bench_data"

# --- Precondition checks (ED-3) ---
for prereq in "$GT_DIR" "$MODEL_CONFIGS" "$BENCH_DATA"; do
    if [[ ! -d "$prereq" ]]; then
        echo "ERROR: required directory missing: $prereq" >&2
        exit 1
    fi
done

# --- Train/Test split (idea4 design) ---
# Train (9): all 5 model families represented; mix of chat/code/summarization
TRAIN_EXPERIMENTS=(
    "jan30-llama2-7b-tp1-chatsweep|llama-2-7b-hf|1"
    "jan30-llama2-7b-tp2-codesweep|llama-2-7b-hf|2"
    "jan30-llama2-7b-tp4-chatsweep|llama-2-7b-hf|4"
    "20260210-codellama-34b-tp2-chatsweep|codellama-34b-instruct-hf|2"
    "20260210-codellama-34b-tp2-codesweep|codellama-34b-instruct-hf|2"
    "20260210-llama2-70b-tp4-chatsweep|llama-2-70b-hf|4"
    "20260210-qwen3-14b-tp1-codesweep|qwen3-14b|1"
    "20260210-qwen3-14b-tp2-chatsweep|qwen3-14b|2"
    "dec17-tp1-qwen7-summarization|qwen2.5-7b-instruct|1"
)

# Test (4): llama2-7b at all 3 TP levels + llama2-70b; unseen workload-TP combos
TEST_EXPERIMENTS=(
    "jan30-llama2-7b-tp1-codesweep|llama-2-7b-hf|1"
    "jan30-llama2-7b-tp2-chatsweep|llama-2-7b-hf|2"
    "jan30-llama2-7b-tp4-codesweep|llama-2-7b-hf|4"
    "20260210-llama2-70b-tp4-codesweep|llama-2-70b-hf|4"
)

ALL_EXPERIMENTS=("${TRAIN_EXPERIMENTS[@]}" "${TEST_EXPERIMENTS[@]}")

# --- Helper: extract workload params from profile.yaml ---
extract_profile() {
    local profile_path="$1"
    python3 -c "
import yaml, json, sys
with open('$profile_path') as f:
    p = yaml.safe_load(f)
d = p.get('data', {})
out = {
    'num_requests': p.get('max-requests', 50),
    'prefix_tokens': d.get('prefix_tokens', 0),
    'prompt_tokens': d.get('prompt_tokens', 512),
    'prompt_tokens_stdev': d.get('prompt_tokens_stdev', 256),
    'prompt_tokens_min': d.get('prompt_tokens_min', 2),
    'prompt_tokens_max': d.get('prompt_tokens_max', 7000),
    'output_tokens': d.get('output_tokens', 512),
    'output_tokens_stdev': d.get('output_tokens_stdev', 256),
    'output_tokens_min': d.get('output_tokens_min', 2),
    'output_tokens_max': d.get('output_tokens_max', 7000),
}
json.dump(out, sys.stdout)
"
}

# --- Helper: extract synchronous rate from guidellm-results.json ---
extract_sync_rate() {
    local results_path="$1"
    python3 -c "
import json, sys
with open('$results_path') as f:
    data = json.load(f)
b0 = data['benchmarks'][0]
n = len(b0['requests']['successful'])
dur = b0.get('duration', 0)
if dur > 0:
    print(f'{n / dur:.6f}')
else:
    print('1.0')
"
}

# --- Run one experiment at a given perLayerOverhead ---
run_one() {
    local exp_name="$1"
    local config_folder="$2"
    local tp="$3"
    local overhead="$4"
    local output_tag="$5"

    local gt_exp_dir="$GT_DIR/$exp_name"
    local profile_path="$gt_exp_dir/profile.yaml"
    local guidellm_path="$gt_exp_dir/guidellm-results.json"

    if [[ ! -f "$profile_path" ]] || [[ ! -f "$guidellm_path" ]]; then
        echo "  SKIP: missing ground truth for $exp_name" >&2
        return 1
    fi

    # Extract workload parameters
    local params_json
    params_json=$(extract_profile "$profile_path")
    local num_requests prefix_tokens prompt_tokens prompt_tokens_stdev
    local prompt_tokens_min prompt_tokens_max output_tokens output_tokens_stdev
    local output_tokens_min output_tokens_max sync_rate

    num_requests=$(echo "$params_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['num_requests'])")
    prefix_tokens=$(echo "$params_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['prefix_tokens'])")
    prompt_tokens=$(echo "$params_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['prompt_tokens'])")
    prompt_tokens_stdev=$(echo "$params_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['prompt_tokens_stdev'])")
    prompt_tokens_min=$(echo "$params_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['prompt_tokens_min'])")
    prompt_tokens_max=$(echo "$params_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['prompt_tokens_max'])")
    output_tokens=$(echo "$params_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['output_tokens'])")
    output_tokens_stdev=$(echo "$params_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['output_tokens_stdev'])")
    output_tokens_min=$(echo "$params_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['output_tokens_min'])")
    output_tokens_max=$(echo "$params_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['output_tokens_max'])")
    sync_rate=$(extract_sync_rate "$guidellm_path")

    # Write hardware config with this overhead value
    local hw_config="$RESULTS_DIR/hw_${output_tag}.json"
    cat > "$hw_config" << HWEOF
{
    "H100": {
        "TFlopsPeak": 989.5,
        "BwPeakTBs": 3.35,
        "bwEfficiencyFactor": 0.82,
        "perLayerOverhead": $overhead
    }
}
HWEOF

    local results_file="$RESULTS_DIR/${exp_name}_${output_tag}.json"
    local stdout_file="$RESULTS_DIR/${exp_name}_${output_tag}.txt"

    blis_run "$TIMEOUT_STANDARD" "$stdout_file" \
        --hardware-config "$hw_config" \
        --model "$config_folder" \
        --model-config-folder "$MODEL_CONFIGS/$config_folder" \
        --bench-data-path "$BENCH_DATA" \
        --hardware H100 --tp "$tp" \
        --workload distribution \
        --seed 42 \
        --num-requests "$num_requests" \
        --rate "$sync_rate" \
        --prefix-tokens "$prefix_tokens" \
        --prompt-tokens "$prompt_tokens" \
        --prompt-tokens-stdev "$prompt_tokens_stdev" \
        --prompt-tokens-min "$prompt_tokens_min" \
        --prompt-tokens-max "$prompt_tokens_max" \
        --output-tokens "$output_tokens" \
        --output-tokens-stdev "$output_tokens_stdev" \
        --output-tokens-min "$output_tokens_min" \
        --output-tokens-max "$output_tokens_max" \
        --results-path "$results_file" || true

    if [[ -f "$results_file" ]]; then
        return 0
    else
        return 1
    fi
}

echo "=========================================="
echo "  HG1: perLayerOverhead Grid Search"
echo "=========================================="
echo ""
echo "Phase 1 (coarse): 0 to 500μs in 25μs steps (21 values)"
echo "Phase 2 (fine): ±25μs around optimum in 5μs steps (11 values)"
echo "Train: 9 experiments, Test: 4 experiments"
echo "All runs: bwEfficiencyFactor=0.82 (H1 correction)"
echo ""

# ===========================
# PHASE 1: Coarse sweep
# ===========================
echo "--- Phase 1: Coarse Sweep (0-500μs, step=25) ---"
echo ""

for overhead in $(seq 0 25 500); do
    tag="coarse_${overhead}"
    echo "  perLayerOverhead=${overhead}μs ..."

    for entry in "${TRAIN_EXPERIMENTS[@]}"; do
        IFS='|' read -r exp_name config_folder tp <<< "$entry"
        run_one "$exp_name" "$config_folder" "$tp" "$overhead" "$tag" 2>/dev/null || true
    done
done

echo ""
echo "Phase 1 complete. Running coarse analysis..."
echo ""

# Run coarse analysis to find optimum
COARSE_OPTIMUM=$(python3 "$SCRIPT_DIR/analyze.py" \
    --phase coarse \
    --results-dir "$RESULTS_DIR" \
    --gt-dir "$GT_DIR" \
    --step 25 --start 0 --end 500)

echo "Coarse optimum: ${COARSE_OPTIMUM}μs"
echo ""

# ===========================
# PHASE 2: Fine sweep
# ===========================
FINE_START=$((COARSE_OPTIMUM - 25))
FINE_END=$((COARSE_OPTIMUM + 25))
if [[ $FINE_START -lt 0 ]]; then FINE_START=0; fi

echo "--- Phase 2: Fine Sweep (${FINE_START}-${FINE_END}μs, step=5) ---"
echo ""

for overhead in $(seq "$FINE_START" 5 "$FINE_END"); do
    tag="fine_${overhead}"
    echo "  perLayerOverhead=${overhead}μs ..."

    for entry in "${TRAIN_EXPERIMENTS[@]}"; do
        IFS='|' read -r exp_name config_folder tp <<< "$entry"
        run_one "$exp_name" "$config_folder" "$tp" "$overhead" "$tag" 2>/dev/null || true
    done
done

echo ""
echo "Phase 2 complete. Running fine analysis..."
echo ""

FINE_OPTIMUM=$(python3 "$SCRIPT_DIR/analyze.py" \
    --phase fine \
    --results-dir "$RESULTS_DIR" \
    --gt-dir "$GT_DIR" \
    --step 5 --start "$FINE_START" --end "$FINE_END")

echo "Fine-grained optimum: ${FINE_OPTIMUM}μs"
echo ""

# ===========================
# PHASE 3: Validation on test set
# ===========================
echo "--- Phase 3: Test Set Validation ---"
echo ""
echo "Running test experiments at optimum=${FINE_OPTIMUM}μs and baseline (H2b=100μs)..."

for entry in "${TEST_EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_name config_folder tp <<< "$entry"
    echo "  $exp_name ..."
    run_one "$exp_name" "$config_folder" "$tp" "$FINE_OPTIMUM" "test_optimum" 2>/dev/null || true
    run_one "$exp_name" "$config_folder" "$tp" 100 "test_h2b" 2>/dev/null || true
    run_one "$exp_name" "$config_folder" "$tp" 0 "test_nooverhead" 2>/dev/null || true
done

echo ""

# Also run train set at the fine optimum for the complete picture
echo "Running train experiments at optimum=${FINE_OPTIMUM}μs and baseline..."
for entry in "${TRAIN_EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_name config_folder tp <<< "$entry"
    run_one "$exp_name" "$config_folder" "$tp" "$FINE_OPTIMUM" "final_optimum" 2>/dev/null || true
    run_one "$exp_name" "$config_folder" "$tp" 100 "final_h2b" 2>/dev/null || true
    run_one "$exp_name" "$config_folder" "$tp" 0 "final_nooverhead" 2>/dev/null || true
done

echo ""
echo "=========================================="
echo "  Full Analysis"
echo "=========================================="
echo ""

python3 "$SCRIPT_DIR/analyze.py" \
    --phase final \
    --results-dir "$RESULTS_DIR" \
    --gt-dir "$GT_DIR" \
    --optimum "$FINE_OPTIMUM" \
    --coarse-optimum "$COARSE_OPTIMUM"

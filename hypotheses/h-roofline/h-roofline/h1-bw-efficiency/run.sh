#!/bin/bash
# H1 Bandwidth Efficiency Validation
#
# Hypothesis: The simulator systematically underestimates latency for
# memory-bound steps because it uses theoretical peak HBM bandwidth
# instead of achievable (~80-82%) sustained bandwidth.
#
# Tests: Run BLIS roofline v2 against all 13 ground truth experiments in
# two configurations (baseline = raw peak BW, treatment = bwEfficiencyFactor=0.82)
# and compare predicted TTFT, TPOT, E2E against measured values.
#
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$SCRIPT_DIR/../../../lib/harness.sh"
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

# --- Hardware configs: baseline (no BW correction) vs treatment (0.82) ---
cat > "$RESULTS_DIR/hw_baseline.json" << 'EOF'
{
    "H100": {
        "TFlopsPeak": 989.5,
        "BwPeakTBs": 3.35
    }
}
EOF

cat > "$RESULTS_DIR/hw_treatment.json" << 'EOF'
{
    "H100": {
        "TFlopsPeak": 989.5,
        "BwPeakTBs": 3.35,
        "bwEfficiencyFactor": 0.82
    }
}
EOF

# --- Experiment matrix ---
# Format: experiment_name|model_config_folder|tp
EXPERIMENTS=(
    "jan30-llama2-7b-tp1-chatsweep|llama-2-7b-hf|1"
    "jan30-llama2-7b-tp1-codesweep|llama-2-7b-hf|1"
    "jan30-llama2-7b-tp2-chatsweep|llama-2-7b-hf|2"
    "jan30-llama2-7b-tp2-codesweep|llama-2-7b-hf|2"
    "jan30-llama2-7b-tp4-chatsweep|llama-2-7b-hf|4"
    "jan30-llama2-7b-tp4-codesweep|llama-2-7b-hf|4"
    "20260210-codellama-34b-tp2-chatsweep|codellama-34b-instruct-hf|2"
    "20260210-codellama-34b-tp2-codesweep|codellama-34b-instruct-hf|2"
    "20260210-llama2-70b-tp4-chatsweep|llama-2-70b-hf|4"
    "20260210-llama2-70b-tp4-codesweep|llama-2-70b-hf|4"
    "20260210-qwen3-14b-tp1-codesweep|qwen3-14b|1"
    "20260210-qwen3-14b-tp2-chatsweep|qwen3-14b|2"
    "dec17-tp1-qwen7-summarization|qwen2.5-7b-instruct|1"
)

# --- Helper: extract workload params from profile.yaml via Python ---
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
# Uses benchmarks[0] (synchronous mode): effective_rate = num_successful / duration
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

echo "=========================================="
echo "  H1: Bandwidth Efficiency Validation"
echo "=========================================="
echo ""
echo "Baseline: raw peak BW (bwEfficiencyFactor=0)"
echo "Treatment: sustained BW (bwEfficiencyFactor=0.82)"
echo "Ground truth: lowest-QPS (synchronous) benchmark from GuideLLM"
echo ""

# Track successes and failures
SUCCESS_COUNT=0
FAIL_COUNT=0

for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_name config_folder tp <<< "$entry"
    gt_exp_dir="$GT_DIR/$exp_name"

    echo "--- $exp_name (tp=$tp, config=$config_folder) ---"

    # Validate ground truth directory
    if [[ ! -d "$gt_exp_dir" ]]; then
        echo "  SKIP: ground truth dir missing: $gt_exp_dir" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    profile_path="$gt_exp_dir/profile.yaml"
    guidellm_path="$gt_exp_dir/guidellm-results.json"
    if [[ ! -f "$profile_path" ]] || [[ ! -f "$guidellm_path" ]]; then
        echo "  SKIP: missing profile.yaml or guidellm-results.json" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # Extract workload parameters
    params_json=$(extract_profile "$profile_path")
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

    # Extract synchronous rate
    sync_rate=$(extract_sync_rate "$guidellm_path")

    echo "  requests=$num_requests rate=$sync_rate prefix=$prefix_tokens prompt=$prompt_tokens output=$output_tokens"

    # Common BLIS flags
    # --model is required by CLI even in roofline mode; use config folder name
    common_flags=(
        --model "$config_folder"
        --model-config-folder "$MODEL_CONFIGS/$config_folder"
        --bench-data-path "$BENCH_DATA"
        --hardware H100 --tp "$tp"
        --workload distribution
        --seed 42
        --num-requests "$num_requests"
        --rate "$sync_rate"
        --prefix-tokens "$prefix_tokens"
        --prompt-tokens "$prompt_tokens"
        --prompt-tokens-stdev "$prompt_tokens_stdev"
        --prompt-tokens-min "$prompt_tokens_min"
        --prompt-tokens-max "$prompt_tokens_max"
        --output-tokens "$output_tokens"
        --output-tokens-stdev "$output_tokens_stdev"
        --output-tokens-min "$output_tokens_min"
        --output-tokens-max "$output_tokens_max"
        --results-path
    )

    # Run baseline
    baseline_results="$RESULTS_DIR/${exp_name}_baseline.json"
    baseline_out="$RESULTS_DIR/${exp_name}_baseline.txt"
    echo "  Running baseline..."
    blis_run "$TIMEOUT_STANDARD" "$baseline_out" \
        --hardware-config "$RESULTS_DIR/hw_baseline.json" \
        "${common_flags[@]}" "$baseline_results" || true

    # Run treatment
    treatment_results="$RESULTS_DIR/${exp_name}_treatment.json"
    treatment_out="$RESULTS_DIR/${exp_name}_treatment.txt"
    echo "  Running treatment..."
    blis_run "$TIMEOUT_STANDARD" "$treatment_out" \
        --hardware-config "$RESULTS_DIR/hw_treatment.json" \
        "${common_flags[@]}" "$treatment_results" || true

    # Check outcomes
    if [[ -f "$baseline_results" ]] && [[ -f "$treatment_results" ]]; then
        echo "  OK: both runs completed"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "  WARN: one or both runs did not produce results" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
done

echo "=========================================="
echo "  Run Summary: $SUCCESS_COUNT succeeded, $FAIL_COUNT failed"
echo "=========================================="
echo ""

# --- Run analysis ---
echo "Running analysis..."
python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR" "$GT_DIR"

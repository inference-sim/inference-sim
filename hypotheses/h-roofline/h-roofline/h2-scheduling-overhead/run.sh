#!/bin/bash
# H2 Scheduling Overhead Validation
#
# Hypothesis: The simulator underestimates step time by a fixed additive
# amount representing CPU-side vLLM scheduling overhead (~5ms decode,
# ~30ms prefill) that the roofline model does not capture.
#
# Tests: Run BLIS roofline v2 against all 13 ground truth experiments in
# two configurations:
#   Baseline: H1 BW correction only (bwEfficiencyFactor=0.82)
#   Treatment: H1 + InferSim-scale overheads (TOverheadMicros=5000, PrefillOverheadMicros=30000)
#
# Reference: hypotheses/h-roofline/h1-bw-efficiency/run.sh (H1 baseline config)
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

# --- Hardware configs ---
# Baseline: H1 correction only (same as H1 treatment)
cat > "$RESULTS_DIR/hw_baseline.json" << 'EOF'
{
    "H100": {
        "TFlopsPeak": 989.5,
        "BwPeakTBs": 3.35,
        "bwEfficiencyFactor": 0.82
    }
}
EOF

# Treatment: H1 + InferSim-equivalent fixed 5ms decode overhead.
# The engine applies: perLayerOverhead × num_layers / tp.
# To produce a fixed 5ms overhead regardless of model, we generate
# per-experiment configs where perLayerOverhead = 5000 / (num_layers / tp).
# This preserves the H2 hypothesis: "fixed additive overhead independent of compute."

FIXED_DECODE_OVERHEAD_US=5000  # InferSim: 5ms decode (models/model.py:229)

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

# --- Helper: extract num_hidden_layers from model config.json ---
extract_num_layers() {
    local config_path="$1"
    python3 -c "
import json, sys
with open('$config_path') as f:
    cfg = json.load(f)
if 'text_config' in cfg:
    cfg = cfg['text_config']
print(cfg.get('num_hidden_layers', 0))
"
}

# --- Helper: generate treatment config with fixed-equivalent overhead ---
# Engine formula: overhead = perLayerOverhead × num_layers / tp
# To get fixed 5ms: perLayerOverhead = 5000 / (num_layers / tp) = 5000 * tp / num_layers
generate_fixed_overhead_hw_config() {
    local num_layers="$1"
    local tp="$2"
    local output_path="$3"

    local per_layer_us=$(python3 -c "print(round($FIXED_DECODE_OVERHEAD_US * $tp / $num_layers, 1))")

    cat > "$output_path" << EOFHW
{
    "H100": {
        "TFlopsPeak": 989.5,
        "BwPeakTBs": 3.35,
        "bwEfficiencyFactor": 0.82,
        "perLayerOverhead": $per_layer_us
    }
}
EOFHW
    echo "  Fixed 5ms equivalent: perLayerOverhead=${per_layer_us}μs (${num_layers} layers, TP=${tp})"
}

echo "=========================================="
echo "  H2: Scheduling Overhead Validation"
echo "=========================================="
echo ""
echo "Baseline: H1 BW correction only (bwEfficiencyFactor=0.82)"
echo "Treatment: H1 + fixed 5ms overhead (via per-experiment perLayerOverhead)"
echo "Ground truth: synchronous benchmark from GuideLLM"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0

for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_name config_folder tp <<< "$entry"
    gt_exp_dir="$GT_DIR/$exp_name"

    echo "--- $exp_name (tp=$tp, config=$config_folder) ---"

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

    # Extract num_hidden_layers from model config
    model_config_json="$MODEL_CONFIGS/$config_folder/config.json"
    if [[ ! -f "$model_config_json" ]]; then
        echo "  SKIP: missing model config: $model_config_json" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    num_layers=$(extract_num_layers "$model_config_json")

    # Generate per-experiment treatment config (fixed 5ms equivalent)
    treatment_hw_config="$RESULTS_DIR/hw_treatment_${exp_name}.json"
    generate_fixed_overhead_hw_config "$num_layers" "$tp" "$treatment_hw_config"

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

    sync_rate=$(extract_sync_rate "$guidellm_path")
    echo "  requests=$num_requests rate=$sync_rate prefix=$prefix_tokens prompt=$prompt_tokens output=$output_tokens"

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

    # Run baseline (H1 only)
    baseline_results="$RESULTS_DIR/${exp_name}_baseline.json"
    baseline_out="$RESULTS_DIR/${exp_name}_baseline.txt"
    echo "  Running baseline (H1 only)..."
    blis_run "$TIMEOUT_STANDARD" "$baseline_out" \
        --hardware-config "$RESULTS_DIR/hw_baseline.json" \
        "${common_flags[@]}" "$baseline_results" || true

    # Run treatment (H1 + H2 fixed 5ms overhead)
    treatment_results="$RESULTS_DIR/${exp_name}_treatment.json"
    treatment_out="$RESULTS_DIR/${exp_name}_treatment.txt"
    echo "  Running treatment (H1 + fixed 5ms)..."
    blis_run "$TIMEOUT_STANDARD" "$treatment_out" \
        --hardware-config "$treatment_hw_config" \
        "${common_flags[@]}" "$treatment_results" || true

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

echo "Running analysis..."
python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR" "$GT_DIR"

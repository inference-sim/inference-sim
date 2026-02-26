#!/bin/bash
# H5 Mixed-Batch Additive Model Validation
#
# Hypothesis: The roofline mixed-batch model should combine prefill and decode
# times using an additive formula (GEMM(totalBatch) + PrefillAttn + DecodeAttn),
# verified by E2E MAPE improvement over ground truth and smoothness of the
# latency curve across prefill/decode ratios.
#
# Family: Structural model
# VV&UQ: Validation
# Type: Deterministic
#
# Part A: Synthetic ratio sweep (simulator-internal, no ground truth)
# Part B: Ground truth comparison across 13 experiments at all QPS sweep points
#
# ED-6 Config diff vs H1: Adds PerLayerCPUOverhead=100 (H2b) and
#   mixedBatchMode={weighted-average, smooth-wa, additive}.
#   H3 (fuseQKV) omitted — 0.0pp effect. H4 (perComponentRoofline) omitted — +0.3pp.
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

# --- Hardware configs: three treatment arms ---
# Baseline: weighted-average (default, no mixedBatchMode field)
cat > "$RESULTS_DIR/hw_baseline.json" << 'EOF'
{
    "H100": {
        "TFlopsPeak": 989.5,
        "BwPeakTBs": 3.35,
        "bwEfficiencyFactor": 0.82,
        "perLayerOverhead": 100
    }
}
EOF

# Treatment 1: smooth weighted-average (no branch thresholds)
cat > "$RESULTS_DIR/hw_smooth_wa.json" << 'EOF'
{
    "H100": {
        "TFlopsPeak": 989.5,
        "BwPeakTBs": 3.35,
        "bwEfficiencyFactor": 0.82,
        "perLayerOverhead": 100,
        "mixedBatchMode": "smooth-wa"
    }
}
EOF

# Treatment 2: additive model
cat > "$RESULTS_DIR/hw_additive.json" << 'EOF'
{
    "H100": {
        "TFlopsPeak": 989.5,
        "BwPeakTBs": 3.35,
        "bwEfficiencyFactor": 0.82,
        "perLayerOverhead": 100,
        "mixedBatchMode": "additive"
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

# --- Helper: extract ALL benchmark rates from guidellm-results.json ---
# Returns JSON array of {index, strategy_type, rate, num_successful} per benchmark
extract_all_rates() {
    local results_path="$1"
    python3 -c "
import json, sys
with open('$results_path') as f:
    data = json.load(f)
out = []
for i, b in enumerate(data['benchmarks']):
    strategy = b.get('config', {}).get('strategy', {})
    stype = strategy.get('type_', 'unknown')
    n = len(b.get('requests', {}).get('successful', []))
    dur = b.get('duration', 0)
    if stype == 'synchronous':
        rate = n / dur if dur > 0 else 1.0
    elif stype == 'constant':
        rate = strategy.get('rate', n / dur if dur > 0 else 1.0)
    elif stype == 'throughput':
        rate = n / dur if dur > 0 else 10.0
    else:
        rate = n / dur if dur > 0 else 1.0
    out.append({'index': i, 'type': stype, 'rate': rate, 'n': n})
json.dump(out, sys.stdout)
"
}

echo "=========================================="
echo "  H5: Mixed-Batch Additive Model Validation"
echo "=========================================="
echo ""
echo "Baseline:    weighted-average (current, with branch thresholds)"
echo "Control:     smooth-wa (no branch thresholds, still weighted average)"
echo "Treatment:   additive (GEMM(totalBatch) + PrefillAttn + DecodeAttn)"
echo "All include: H1 (BwEfficiencyFactor=0.82) + H2b (PerLayerCPUOverhead=100)"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0

for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_name config_folder tp <<< "$entry"
    gt_exp_dir="$GT_DIR/$exp_name"

    echo "=== $exp_name (tp=$tp, config=$config_folder) ==="

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

    # Extract all benchmark rates
    all_rates_json=$(extract_all_rates "$guidellm_path")
    num_benchmarks=$(echo "$all_rates_json" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))")

    echo "  requests=$num_requests prefix=$prefix_tokens prompt=$prompt_tokens output=$output_tokens"
    echo "  benchmarks=$num_benchmarks"

    for bi in $(seq 0 $((num_benchmarks - 1))); do
        bench_rate=$(echo "$all_rates_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[$bi][\"rate\"]:.6f}')" 2>/dev/null)
        bench_type=$(echo "$all_rates_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[$bi]['type'])" 2>/dev/null)

        echo "  --- benchmark[$bi] type=$bench_type rate=$bench_rate ---"

        # Common BLIS flags for this benchmark point
        common_flags=(
            --model "$config_folder"
            --model-config-folder "$MODEL_CONFIGS/$config_folder"
            --bench-data-path "$BENCH_DATA"
            --hardware H100 --tp "$tp"
            --workload distribution
            --seed 42
            --num-requests "$num_requests"
            --rate "$bench_rate"
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

        # Run all three arms
        for arm in baseline smooth_wa additive; do
            hw_file="$RESULTS_DIR/hw_${arm}.json"
            results_file="$RESULTS_DIR/${exp_name}_b${bi}_${arm}.json"
            out_file="$RESULTS_DIR/${exp_name}_b${bi}_${arm}.txt"

            blis_run "$TIMEOUT_STANDARD" "$out_file" \
                --hardware-config "$hw_file" \
                "${common_flags[@]}" "$results_file" || true
        done

        # Verify all three arms produced results
        all_ok=true
        for arm in baseline smooth_wa additive; do
            if [[ ! -f "$RESULTS_DIR/${exp_name}_b${bi}_${arm}.json" ]]; then
                all_ok=false
            fi
        done

        if $all_ok; then
            echo "    OK: all arms completed"
        else
            echo "    WARN: one or more arms missing results" >&2
        fi
    done

    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    echo ""
done

echo "=========================================="
echo "  Run Summary: $SUCCESS_COUNT experiments, $FAIL_COUNT skipped"
echo "=========================================="
echo ""

# --- Run analysis ---
echo "Running analysis..."
python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR" "$GT_DIR"

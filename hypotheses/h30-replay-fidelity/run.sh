#!/usr/bin/env bash
set -euo pipefail

# H30: Three-way comparison — CrossModel vs Blackbox (per-model) vs Real vLLM
# Uses ./blis run CLI for both backends. Single instance, no routing.

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
REPLAY_DATA="${REPO_ROOT}/training/replay_data"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

[ -f "./blis" ] || go build -o blis main.go

# Iter 2 per-model blackbox coefficients (from training/ledger.md)
declare -A BB_BETA=(
    ["llama-2-7b"]="6179,2.065,121.342"
    ["llama-2-70b"]="16536,3.622,42.500"
    ["mixtral-8x7b"]="17560,5.028,18.132"
    ["codellama-34b"]="13549,2.020,35.910"
)
declare -A BB_ALPHA=(
    ["llama-2-7b"]="9680,0,0"
    ["llama-2-70b"]="17660,0,0"
    ["mixtral-8x7b"]="16081,0,0"
    ["codellama-34b"]="14986,0,50.8"
)
declare -A CFG_DIR=(
    ["llama-2-7b"]="Llama-2-7b-hf"
    ["llama-2-70b"]="Llama-2-70b-hf"
    ["mixtral-8x7b"]="Mixtral-8x7B-v0.1"
    ["codellama-34b"]="CodeLlama-34b-Instruct-hf"
)

TRAIN_EXPERIMENTS=(
    "20260217-231439-llama-2-7b-tp1-general"
    "20260217-155451-llama-2-7b-tp1-codegen"
    "20260217-162547-llama-2-7b-tp1-roleplay"
    "20260217-202857-llama-2-70b-tp4-general"
    "20260217-203421-llama-2-70b-hf-tp4-codegen"
    "20260218-084319-llama-2-70b-tp4-roleplay"
    "20260218-130541-mixtral-8x7b-v0-1-tp2-general"
    "20260218-120914-mixtral-8x7b-v0-1-tp2-codegen"
    "20260218-141024-mixtral-8x7b-v0-1-tp2-roleplay"
    "20260218-150304-codellama-34b-tp2-general"
)

echo "=== H30: Three-Way Comparison — Training Set ==="

run_experiment() {
    local exp="$1" backend="$2" suffix="$3"
    local GT="${REPLAY_DATA}/${exp}_ground_truth.json"
    local SPEC="${REPLAY_DATA}/${exp}.yaml"
    [ ! -f "${GT}" ] && return 1

    local MODEL TP KV MAX_S MAX_T N_REQ HOR M_SHORT
    read -r MODEL TP KV MAX_S MAX_T N_REQ HOR M_SHORT < <(python3 -c "
import json; d=json.load(open('${GT}'))
c=d['config']
print(c['model'], c['tensor_parallelism'], c['kv_blocks_total_gpu'],
      c['max_num_seqs'], c['max_num_batched_tokens'], c['total_requests'],
      c['horizon_us'], d['model_short'])
")

    local V2="${OUTPUT_DIR}/${exp}_v2.yaml"
    [ ! -f "${V2}" ] && ./blis convert inference-perf --spec "${SPEC}" > "${V2}" 2>/dev/null

    local EXTRA_FLAGS=""
    if [ "${backend}" = "crossmodel" ]; then
        EXTRA_FLAGS="--latency-model crossmodel --hardware H100 --model-config-folder training/model_configs/${CFG_DIR[${M_SHORT}]}"
    else
        EXTRA_FLAGS="--latency-model blackbox --beta-coeffs ${BB_BETA[${M_SHORT}]} --alpha-coeffs ${BB_ALPHA[${M_SHORT}]}"
    fi

    ./blis run \
        --model "${MODEL}" ${EXTRA_FLAGS} \
        --tp "${TP}" --total-kv-blocks "${KV}" --block-size-in-tokens 16 \
        --max-num-running-reqs "${MAX_S}" --max-num-scheduled-tokens "${MAX_T}" \
        --num-instances 1 --num-requests "${N_REQ}" --horizon "${HOR}" \
        --workload-spec "${V2}" \
        --results-path "${OUTPUT_DIR}/${exp}_${suffix}.json" \
        > /dev/null 2> "${OUTPUT_DIR}/${exp}_${suffix}.log"
}

for exp in "${TRAIN_EXPERIMENTS[@]}"; do
    GT="${REPLAY_DATA}/${exp}_ground_truth.json"
    [ ! -f "${GT}" ] && { echo "SKIP: ${exp}"; continue; }
    M_SHORT=$(python3 -c "import json; print(json.load(open('${GT}'))['model_short'])")

    echo "  ${exp}: crossmodel..."
    run_experiment "${exp}" "crossmodel" "cm" || true
    echo "  ${exp}: blackbox (Iter 2 per-model)..."
    run_experiment "${exp}" "blackbox" "bb" || true
done

echo ""
echo "=== Comparison Table ==="
python3 "${SCRIPT_DIR}/analyze.py" \
    --output-dir "${OUTPUT_DIR}" \
    --ground-truth-dir "${REPLAY_DATA}" \
    --split train

echo "=== Done ==="

#!/usr/bin/env bash
set -euo pipefail

# H31: CrossModel Generalization to Near-Saturation Reasoning
# Uses ./blis run CLI (same approach as H30), NOT the Go replay binary.

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
REPLAY_DATA="${REPO_ROOT}/training/replay_data"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

echo "=== H31: CrossModel Generalization to Reasoning ==="

# Ensure specs exist
if [ ! -d "${REPLAY_DATA}" ] || [ -z "$(ls ${REPLAY_DATA}/*.yaml 2>/dev/null)" ]; then
    python3 training/generate_replay_specs.py
fi
[ -f "./blis" ] || go build -o blis main.go

# Model config directories
declare -A CFG=(
    ["llama-2-7b"]="Llama-2-7b-hf"
    ["llama-2-70b"]="Llama-2-70b-hf"
    ["mixtral-8x7b"]="Mixtral-8x7B-v0.1"
    ["codellama-34b"]="CodeLlama-34b-Instruct-hf"
)

EXPERIMENTS=(
    "20260218-160939-codellama-34b-tp2-reasoning"
    "20260218-065057-llama-2-70b-hf-tp4-reasoning"
    "20260217-170634-llama-2-7b-tp1-reasoning"
    "20260218-135247-mixtral-8x7b-v0-1-tp2-reasoning"
)

for exp in "${EXPERIMENTS[@]}"; do
    GT="${REPLAY_DATA}/${exp}_ground_truth.json"
    SPEC="${REPLAY_DATA}/${exp}.yaml"
    [ ! -f "${GT}" ] && { echo "SKIP: ${exp}"; continue; }

    read -r MODEL TP KV MAX_S MAX_T N_REQ HOR M_SHORT < <(python3 -c "
import json; d=json.load(open('${GT}'))
c=d['config']
print(c['model'], c['tensor_parallelism'], c['kv_blocks_total_gpu'],
      c['max_num_seqs'], c['max_num_batched_tokens'], c['total_requests'],
      c['horizon_us'], d['model_short'])
")

    echo "  ${M_SHORT}/${exp##*-}: TP=${TP}, KV=${KV}, reqs=${N_REQ}"

    V2="${OUTPUT_DIR}/${exp}_v2.yaml"
    ./blis convert inference-perf --spec "${SPEC}" > "${V2}" 2>/dev/null

    timeout 300 ./blis run \
        --model "${MODEL}" --latency-model crossmodel --hardware H100 \
        --tp "${TP}" --total-kv-blocks "${KV}" --block-size-in-tokens 16 \
        --max-num-running-reqs "${MAX_S}" --max-num-scheduled-tokens "${MAX_T}" \
        --num-instances 1 --num-requests "${N_REQ}" --horizon "${HOR}" \
        --model-config-folder "training/model_configs/${CFG[${M_SHORT}]}" \
        --workload-spec "${V2}" \
        --results-path "${OUTPUT_DIR}/${exp}_results.json" \
        > /dev/null 2> "${OUTPUT_DIR}/${exp}_stderr.log" \
    || echo "    TIMEOUT/ERROR (exit $?)"
done

python3 "${SCRIPT_DIR}/analyze.py" \
    --replay-dir "${OUTPUT_DIR}" --ground-truth-dir "${REPLAY_DATA}"
echo "=== H31 Complete ==="

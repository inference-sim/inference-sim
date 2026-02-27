#!/bin/bash
# H-Arrival-Generators: Validate Arrival Sampler Distributions
#
# Hypothesis: For each arrival sampler (Poisson, Gamma CV=1.5, Gamma CV=3.5,
# Weibull CV=1.5, Weibull CV=3.5), generating 10K+ inter-arrival times should
# yield (a) sample mean within 5% of theoretical mean, (b) sample CV within
# 10% of theoretical CV, and (c) KS test p > 0.05 against the theoretical CDF.
#
# Classification: Statistical / Equivalence (sample stats vs theoretical)
# Family: Workload/arrival
# VV&UQ: Verification
#
# Design notes:
#   ED-1: Each sampler tested independently with identical rate and request count
#   ED-2: Rate=100 req/s chosen for practical relevance (typical BLIS experiments)
#   ED-3: Precondition — minimal workload (input=1, output=1) eliminates simulation noise
#   ED-5: Reproducible — builds binary, installs scipy in temp venv, no manual steps
#   ED-6: No prior experiment reference (first arrival generator validation)
#
# Reference: https://github.com/inference-sim/inference-sim/issues/312
#
# Usage: ./run.sh [--rebuild]
#
# Requires: Go 1.24+, Python 3 with venv support (scipy installed automatically)

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
RATE=100        # req/s — practical rate for BLIS experiments
NUM_REQUESTS=10001  # 10,001 requests → 10,000 inter-arrival time gaps

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# Set up Python venv with scipy for KS tests
echo "Setting up Python environment with scipy..."
VENV="$RESULTS_DIR/venv"
python3 -m venv "$VENV" 2>/dev/null
"$VENV/bin/pip" install --quiet scipy 2>&1 | tail -1
PYTHON="$VENV/bin/python3"
echo "  scipy installed in temporary venv"
echo ""

# Sampler configurations: name, process, cv (empty for poisson)
SAMPLERS=(
    "poisson:poisson:"
    "gamma_cv1.5:gamma:1.5"
    "gamma_cv3.5:gamma:3.5"
    "weibull_cv1.5:weibull:1.5"
    "weibull_cv3.5:weibull:3.5"
)

# Generate workload YAML for a specific arrival process
make_workload() {
    local process=$1
    local cv=$2  # empty string for poisson
    local seed=$3
    local outfile=$4

    local arrival_block
    if [[ -z "$cv" ]]; then
        arrival_block="      process: $process"
    else
        arrival_block="      process: $process
      cv: $cv"
    fi

    cat > "$outfile" << YAMLEOF
version: "1"
seed: $seed
category: language
aggregate_rate: $RATE
num_requests: $NUM_REQUESTS
clients:
  - id: "arrival-test"
    tenant_id: "default"
    slo_class: "batch"
    rate_fraction: 1.0
    streaming: false
    arrival:
$arrival_block
    input_distribution:
      type: constant
      params:
        value: 1
    output_distribution:
      type: constant
      params:
        value: 1
YAMLEOF
}

run_sim() {
    local results_json=$1
    local stdout_file=$2
    local workload_yaml=$3
    local seed=$4

    timeout 300 "$BINARY" run \
        --model "$MODEL" \
        --num-instances 1 \
        --workload-spec "$workload_yaml" \
        --seed "$seed" \
        --scheduler fcfs \
        --admission-policy always-admit \
        --total-kv-blocks 1000000 \
        --log error \
        --results-path "$results_json" \
        2>/dev/null \
        > "$stdout_file" \
        || echo "    WARNING: timeout or error"
}

echo "============================================================================"
echo "  H-Arrival-Generators: Validate Arrival Sampler Distributions"
echo "  Reference: issue #312, docs/standards/experiments.md (workload/arrival)"
echo "  Type: Statistical / Equivalence (KS test p > 0.05)"
echo "  Family: Workload/arrival | VV&UQ: Verification"
echo "============================================================================"
echo ""
echo "  Config: rate=${RATE} req/s, ${NUM_REQUESTS} requests, 5 samplers, 3 seeds"
echo "  Seeds: ${SEEDS[*]}"
echo ""

for sampler_spec in "${SAMPLERS[@]}"; do
    IFS=':' read -r name process cv <<< "$sampler_spec"

    echo "-- Sampler: ${name} -------------------------------------------------------"
    for seed in "${SEEDS[@]}"; do
        echo "  Running: ${name} seed=${seed} ..."
        wl="$RESULTS_DIR/${name}_s${seed}_wl.yaml"
        make_workload "$process" "$cv" "$seed" "$wl"
        run_sim \
            "$RESULTS_DIR/${name}_s${seed}.json" \
            "$RESULTS_DIR/${name}_s${seed}_stdout.txt" \
            "$wl" "$seed"
    done
    echo ""
done

echo "============================================================================"
echo "  Analysis"
echo "============================================================================"
echo ""

"$PYTHON" "$SCRIPT_DIR/analyze.py" \
    --results-dir "$RESULTS_DIR" \
    --rate "$RATE" \
    --seeds "${SEEDS[*]}"

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis"
echo "============================================================================"

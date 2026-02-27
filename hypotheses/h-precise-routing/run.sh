#!/bin/bash
# Iteration 20: Precise KV Routing
# Tests whether eviction-aware prefix routing beats approximate LRU routing
# under high prefix cardinality and KV memory pressure.
set -euo pipefail

source "$(dirname "$0")/../lib/harness.sh"
setup_experiment --rebuild

RD="$(dirname "$0")/results"
mkdir -p "$RD"

SEEDS=(42 123 7777)
INSTANCES=8
BLOCK_SIZE=16
HORIZON=10000000  # 10s in microseconds

# Helper: generate workload YAML with N prefix groups
gen_workload() {
    local ngroups=$1
    local rate=$2
    local seed=$3
    local outfile="$RESULTS_DIR/workload_g${ngroups}_r${rate}_s${seed}.yaml"

    local frac
    frac=$(python3 -c "print(round(1.0/$ngroups, 6))")

    cat > "$outfile" <<YAMLEOF
version: 2
aggregate_rate: $rate
seed: $seed
clients:
YAMLEOF

    for g in $(seq 1 "$ngroups"); do
        cat >> "$outfile" <<YAMLEOF
  - id: "group_${g}"
    rate_fraction: $frac
    slo_class: standard
    prefix_group: "prefix_g${g}"
    prefix_length: 512
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 512
    output_distribution:
      type: constant
      params:
        value: 128
YAMLEOF
    done
    echo "$outfile"
}

# Run a single experiment
run_one() {
    local policy=$1
    local scorers=$2
    local precise=$3
    local ngroups=$4
    local rate=$5
    local kvblocks=$6
    local seed=$7
    local tag=$8
    local outfile="$RD/${tag}_seed${seed}.json"

    local workload
    workload=$(gen_workload "$ngroups" "$rate" "$seed")

    local extra_flags=""
    if [[ "$precise" == "true" ]]; then
        extra_flags="$extra_flags --precise-kv-routing"
    fi
    if [[ -n "$scorers" ]]; then
        extra_flags="$extra_flags --routing-scorers $scorers"
    fi

    echo "  Running: ${tag} seed=${seed}" >&2
    # shellcheck disable=SC2086
    blis_run "$TIMEOUT_STANDARD" "$outfile" \
        --model "$MODEL" \
        --num-instances "$INSTANCES" \
        --total-kv-blocks "$kvblocks" \
        --block-size-in-tokens "$BLOCK_SIZE" \
        --routing-policy "$policy" \
        --horizon "$HORIZON" \
        --seed "$seed" \
        --workload-spec "$workload" \
        $extra_flags || true
}

echo "========================================"
echo "Iter 20: Precise KV Routing"
echo "  Instances: $INSTANCES, BlockSize: $BLOCK_SIZE"
echo "========================================"

# === Main experiment: vary KV blocks and prefix groups ===
# Fixed rate=400 (within capacity: 8 inst × ~57 req/s = 460 capacity)
# Moderate load creates enough traffic for cache contention without overload queue noise
RATE=400

for kvblocks in 5000 2000 1000; do
    for ngroups in 4 10 20; do
        echo ""
        echo "--- KV=$kvblocks, groups=$ngroups, rate=$RATE ---"
        for seed in "${SEEDS[@]}"; do
            tag_base="kv${kvblocks}_g${ngroups}"

            run_one "round-robin" "" "false" "$ngroups" "$RATE" "$kvblocks" "$seed" "rr_${tag_base}"
            run_one "weighted" "prefix-affinity:4,queue-depth:3" "false" "$ngroups" "$RATE" "$kvblocks" "$seed" "approx_${tag_base}"
            run_one "weighted" "prefix-affinity:4,queue-depth:3" "true" "$ngroups" "$RATE" "$kvblocks" "$seed" "precise_${tag_base}"
        done
    done
done

# === Overload experiment: rate=800 (1.74x overload) with tight KV ===
echo ""
echo "=== Overload: rate=800, KV=2000, 20 groups ==="
for seed in "${SEEDS[@]}"; do
    run_one "round-robin" "" "false" 20 800 2000 "$seed" "rr_overload"
    run_one "weighted" "prefix-affinity:4,queue-depth:3" "false" 20 800 2000 "$seed" "approx_overload"
    run_one "weighted" "prefix-affinity:4,queue-depth:3" "true" 20 800 2000 "$seed" "precise_overload"
done

echo ""
echo "=== Experiment complete ==="

#!/bin/bash
# Iteration 21: Prefill-Decode Disaggregation
#
# Simulates P/D disaggregation by running separate prefill-only and decode-only
# clusters, then comparing TTFT vs a shared (co-located) cluster.
#
# Design: For a fair comparison at total N instances:
#   - Shared: N instances, rate=R, full lifecycle (input=1024, output=256)
#   - Disaggregated: N/2 prefill instances, rate=R, output=1 (TTFT measurement)
#                    N/2 decode instances, rate=R, input=16, output=256
#   - Disaggregated TTFT = prefill cluster TTFT (no decode interference)
#
# The key metric is TTFT P99 improvement, since disaggregation's primary benefit
# is eliminating head-of-line blocking from decode steps during prefill scheduling.
set -euo pipefail

source "$(dirname "$0")/../lib/harness.sh"
setup_experiment --rebuild

RD="$(dirname "$0")/results"
mkdir -p "$RD"

SEEDS=(42 123 7777)
BLOCK_SIZE=16
HORIZON=10000000  # 10s

# Workload generation
gen_workload() {
    local ngroups=$1
    local rate=$2
    local seed=$3
    local input_tokens=$4
    local output_tokens=$5
    local prefix_len=$6
    local outfile="$RESULTS_DIR/wl_g${ngroups}_r${rate}_i${input_tokens}_o${output_tokens}_s${seed}.yaml"

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
    prefix_length: $prefix_len
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: $input_tokens
    output_distribution:
      type: constant
      params:
        value: $output_tokens
YAMLEOF
    done
    echo "$outfile"
}

run_sim() {
    local tag=$1
    local instances=$2
    local kvblocks=$3
    local rate=$4
    local ngroups=$5
    local input_tokens=$6
    local output_tokens=$7
    local prefix_len=$8
    local scorers=$9
    local seed=${10}
    local outfile="$RD/${tag}_seed${seed}.json"

    local workload
    workload=$(gen_workload "$ngroups" "$rate" "$seed" "$input_tokens" "$output_tokens" "$prefix_len")

    local scorer_flags=""
    if [[ -n "$scorers" ]]; then
        scorer_flags="--routing-scorers $scorers"
    fi

    echo "  Running: ${tag} seed=${seed}" >&2
    # shellcheck disable=SC2086
    blis_run "$TIMEOUT_STANDARD" "$outfile" \
        --model "$MODEL" \
        --num-instances "$instances" \
        --total-kv-blocks "$kvblocks" \
        --block-size-in-tokens "$BLOCK_SIZE" \
        --routing-policy "weighted" \
        --horizon "$HORIZON" \
        --seed "$seed" \
        --workload-spec "$workload" \
        $scorer_flags || true
}

echo "========================================"
echo "Iter 21: Prefill-Decode Disaggregation"
echo "========================================"

# Parameters
NGROUPS=8
SCORERS="prefix-affinity:4,queue-depth:3"
KV=5000  # Comfortable KV to isolate the disaggregation effect

# === Sweep 1: Rate sensitivity ===
# Compare shared (8 instances, full lifecycle) vs disaggregated (4 prefill + 4 decode)
for rate in 100 200 400; do
    echo ""
    echo "=== Rate=$rate, ${NGROUPS} groups, KV=$KV ==="

    for seed in "${SEEDS[@]}"; do
        # Shared: 8 instances, full lifecycle
        run_sim "shared_r${rate}" 8 "$KV" "$rate" "$NGROUPS" 512 256 512 "$SCORERS" "$seed"

        # RR baseline: 8 instances, full lifecycle, round-robin
        run_sim "rr_r${rate}" 8 "$KV" "$rate" "$NGROUPS" 512 256 512 "" "$seed"

        # Disaggregated prefill: 4 instances, output=1 (just TTFT)
        run_sim "prefill_r${rate}" 4 "$KV" "$rate" "$NGROUPS" 512 1 512 "$SCORERS" "$seed"

        # Disaggregated decode: 4 instances, input=16 (minimal prefill), full decode
        run_sim "decode_r${rate}" 4 "$KV" "$rate" "$NGROUPS" 16 256 0 "$SCORERS" "$seed"
    done
done

# === Sweep 2: Instance split ratios ===
# Fix rate=200, try different prefill:decode ratios
RATE=200
echo ""
echo "=== Instance split ratios (rate=$RATE, total=8) ==="
for pfx_inst in 2 4 6; do
    dec_inst=$((8 - pfx_inst))
    echo ""
    echo "--- Prefill:${pfx_inst} Decode:${dec_inst} ---"
    for seed in "${SEEDS[@]}"; do
        run_sim "split_p${pfx_inst}d${dec_inst}_prefill" "$pfx_inst" "$KV" "$RATE" "$NGROUPS" 512 1 512 "$SCORERS" "$seed"
        run_sim "split_p${pfx_inst}d${dec_inst}_decode" "$dec_inst" "$KV" "$RATE" "$NGROUPS" 16 256 0 "$SCORERS" "$seed"
    done
done

# === Sweep 3: KV pressure interaction ===
# Does disaggregation help MORE under KV pressure?
echo ""
echo "=== KV pressure interaction (rate=200) ==="
for kvblocks in 5000 2000; do
    echo ""
    echo "--- KV=$kvblocks ---"
    for seed in "${SEEDS[@]}"; do
        run_sim "kv${kvblocks}_shared" 8 "$kvblocks" 200 "$NGROUPS" 512 256 512 "$SCORERS" "$seed"
        run_sim "kv${kvblocks}_prefill" 4 "$kvblocks" 200 "$NGROUPS" 512 1 512 "$SCORERS" "$seed"
        run_sim "kv${kvblocks}_decode" 4 "$kvblocks" 200 "$NGROUPS" 16 256 0 "$SCORERS" "$seed"
    done
done

echo ""
echo "=== Experiment complete ==="

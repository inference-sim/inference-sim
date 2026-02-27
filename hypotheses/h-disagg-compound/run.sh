#!/bin/bash
# Iteration 22: Combined Disaggregation + Compound Strategy + KV Migration Cost
#
# Builds on iter 21 (245x TTFT improvement from disaggregation) by:
# 1. Adding KV migration cost modeling (prefill→decode transfer latency)
# 2. Comparing routing strategies for each pool independently
# 3. Sweeping across load levels to find the crossover point where
#    disaggregation stops being beneficial
#
# KV migration cost is modeled as additional E2E latency on the decode cluster,
# applied via --admission-latency (simulates network transfer before decode starts).
set -euo pipefail

source "$(dirname "$0")/../lib/harness.sh"
setup_experiment --rebuild

RD="$(dirname "$0")/results"
mkdir -p "$RD"

SEEDS=(42 123 7777)
BLOCK_SIZE=16
HORIZON=10000000  # 10s
KV=5000
NGROUPS=8

gen_workload() {
    local ngroups=$1
    local rate=$2
    local seed=$3
    local input_tokens=$4
    local output_tokens=$5
    local prefix_len=$6
    local outfile="$RESULTS_DIR/wl_${7}_s${seed}.yaml"

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
    local rate=$3
    local input_tokens=$4
    local output_tokens=$5
    local prefix_len=$6
    local policy=$7
    local scorers=$8
    local seed=$9
    local extra=${10:-""}
    local outfile="$RD/${tag}_seed${seed}.json"

    local workload
    workload=$(gen_workload "$NGROUPS" "$rate" "$seed" "$input_tokens" "$output_tokens" "$prefix_len" "${tag}")

    local flags=""
    if [[ -n "$scorers" ]]; then
        flags="$flags --routing-scorers $scorers"
    fi
    if [[ -n "$extra" ]]; then
        flags="$flags $extra"
    fi

    echo "  ${tag} seed=${seed}" >&2
    # shellcheck disable=SC2086
    blis_run "$TIMEOUT_STANDARD" "$outfile" \
        --model "$MODEL" \
        --num-instances "$instances" \
        --total-kv-blocks "$KV" \
        --block-size-in-tokens "$BLOCK_SIZE" \
        --routing-policy "$policy" \
        --horizon "$HORIZON" \
        --seed "$seed" \
        --workload-spec "$workload" \
        $flags || true
}

echo "========================================================"
echo "Iter 22: Combined Disaggregation + Compound Strategy"
echo "  ${NGROUPS} prefix groups, KV=$KV, block=$BLOCK_SIZE"
echo "========================================================"

# === Section 1: Routing strategy comparison for prefill pool ===
# Which routing helps more for prefill-only instances?
echo ""
echo "=== Section 1: Prefill pool routing (4 inst, rate=400, output=1) ==="
for seed in "${SEEDS[@]}"; do
    run_sim "pfx_rr"      4 400 512 1 512 "round-robin" ""  "$seed"
    run_sim "pfx_pa4qd3"  4 400 512 1 512 "weighted" "prefix-affinity:4,queue-depth:3" "$seed"
    run_sim "pfx_pa1qd4"  4 400 512 1 512 "weighted" "prefix-affinity:1,queue-depth:4" "$seed"
    run_sim "pfx_ll"      4 400 512 1 512 "least-loaded" "" "$seed"
done

# === Section 2: KV migration cost sensitivity ===
# Model migration as admission latency on decode pool
# Real-world KV migration: 1ms (fast NVLink) to 50ms (network)
echo ""
echo "=== Section 2: KV migration cost (decode pool, 4 inst, rate=200) ==="
for migrate_us in 0 1000 5000 10000 50000; do
    migrate_ms=$((migrate_us / 1000))
    echo ""
    echo "--- Migration cost: ${migrate_ms}ms ---"
    for seed in "${SEEDS[@]}"; do
        # Prefill TTFT (constant — migration doesn't affect prefill)
        run_sim "migrate${migrate_ms}_prefill" 4 200 512 1 512 "weighted" "prefix-affinity:4,queue-depth:3" "$seed"
        # Decode E2E with migration latency added as admission delay
        run_sim "migrate${migrate_ms}_decode" 4 200 16 256 0 "weighted" "queue-depth:1" "$seed" "--admission-latency $migrate_us"
    done
done

# === Section 3: Load crossover — at what rate does disagg stop helping? ===
# Compare shared vs disaggregated at increasing rates
echo ""
echo "=== Section 3: Load crossover (shared-8 vs disagg P:2/D:6) ==="
for rate in 50 100 200 300 400; do
    echo ""
    echo "--- Rate=$rate ---"
    for seed in "${SEEDS[@]}"; do
        # Shared baseline
        run_sim "xover_shared_r${rate}" 8 "$rate" 512 256 512 "weighted" "prefix-affinity:4,queue-depth:3" "$seed"
        # Disaggregated
        run_sim "xover_pfx_r${rate}" 2 "$rate" 512 1 512 "weighted" "prefix-affinity:4,queue-depth:3" "$seed"
        run_sim "xover_dec_r${rate}" 6 "$rate" 16 256 0 "weighted" "queue-depth:1" "$seed"
    done
done

# === Section 4: Combined compound — disagg + SLO admission on decode pool ===
echo ""
echo "=== Section 4: Compound disagg (prefill PA + decode QD+admission, rate=400) ==="
for seed in "${SEEDS[@]}"; do
    # Full compound shared (reference)
    run_sim "compound_shared" 8 400 512 256 512 "weighted" "prefix-affinity:4,queue-depth:3" "$seed" \
        "--admission-policy slo-gated --priority-policy slo-class --scheduler priority-fcfs"
    # Disagg with compound on decode pool
    run_sim "compound_pfx" 2 400 512 1 512 "weighted" "prefix-affinity:4,queue-depth:3" "$seed"
    run_sim "compound_dec" 6 400 16 256 0 "weighted" "queue-depth:1" "$seed" \
        "--admission-policy slo-gated --priority-policy slo-class --scheduler priority-fcfs"
done

echo ""
echo "=== Experiment complete ==="

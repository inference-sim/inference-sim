#!/bin/bash
# H-Adaptive-Routing: Compare adaptive-weighted vs static routing policies
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$REPO_ROOT/simulation_worker"
MODEL="meta-llama/llama-3.1-8b-instruct"
RESULTS_DIR="$SCRIPT_DIR/results"

# Build from this worktree
echo "Building binary..." >&2
(cd "$REPO_ROOT" && go build -o simulation_worker main.go)

mkdir -p "$RESULTS_DIR"

SEEDS="42 123 7777"
NUM_INSTANCES=4
NUM_REQUESTS=500
HORIZON=200000000  # 200s in microseconds
RATE=${RATE_OVERRIDE:-200}  # default 200 (high util), override with RATE_OVERRIDE=300

echo "=== H-Adaptive-Routing Experiment ===" >&2
echo "Instances: $NUM_INSTANCES, Requests: $NUM_REQUESTS, Rate: $RATE" >&2

# Run a single sim config. Non-zero exit tolerated (|| true).
run_sim() {
    local label="$1"
    local workload="$2"
    local seed="$3"
    local policy="$4"
    local scorers="$5"
    local outfile="$RESULTS_DIR/${label}_${workload}_seed${seed}.json"

    local workload_file="$RESULTS_DIR/workload_${workload}_${seed}.yaml"

    # Generate workload YAML based on type
    case "$workload" in
        prefix)
            cat > "$workload_file" <<YAMLEOF
version: "2"
seed: ${seed}
category: language
aggregate_rate: ${RATE}
clients:
  - id: shared-prefix
    tenant_id: tenant-A
    slo_class: standard
    rate_fraction: 0.8
    prefix_group: system-prompt
    prefix_length: 256
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 64
        min: 64
        max: 512
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 32
        min: 32
        max: 256
  - id: unique-requests
    tenant_id: tenant-B
    slo_class: standard
    rate_fraction: 0.2
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 512
        std_dev: 128
        min: 64
        max: 1024
    output_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 64
        min: 64
        max: 512
YAMLEOF
            ;;
        indep)
            cat > "$workload_file" <<YAMLEOF
version: "2"
seed: ${seed}
category: language
aggregate_rate: ${RATE}
clients:
  - id: diverse-requests
    tenant_id: tenant-C
    slo_class: standard
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 512
        std_dev: 256
        min: 64
        max: 2048
    output_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 128
        min: 32
        max: 1024
YAMLEOF
            ;;
        mixed)
            cat > "$workload_file" <<YAMLEOF
version: "2"
seed: ${seed}
category: language
aggregate_rate: ${RATE}
clients:
  - id: prefix-client
    tenant_id: tenant-A
    slo_class: standard
    rate_fraction: 0.5
    prefix_group: system-prompt
    prefix_length: 256
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 64
        min: 64
        max: 512
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 32
        min: 32
        max: 256
  - id: independent-client
    tenant_id: tenant-B
    slo_class: standard
    rate_fraction: 0.5
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 512
        std_dev: 256
        min: 64
        max: 2048
    output_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 128
        min: 32
        max: 1024
YAMLEOF
            ;;
    esac

    local scorer_flags=""
    if [[ "$scorers" != "none" ]]; then
        scorer_flags="--routing-scorers $scorers"
    fi

    timeout 300 "$BINARY" run \
        --model "$MODEL" \
        --num-instances "$NUM_INSTANCES" \
        --routing-policy "$policy" \
        $scorer_flags \
        --num-requests "$NUM_REQUESTS" \
        --horizon "$HORIZON" \
        --seed "$seed" \
        --workload-spec "$workload_file" \
        > "$outfile" 2>"$RESULTS_DIR/${label}_${workload}_seed${seed}.stderr" || true
}

# Policy configurations: label|policy|scorers
CONFIGS="
adaptive|adaptive-weighted|prefix-affinity:3,queue-depth:2,kv-utilization:2
static-default|weighted|prefix-affinity:3,queue-depth:2,kv-utilization:2
static-cache-heavy|weighted|prefix-affinity:5,queue-depth:1,kv-utilization:1
static-load-heavy|weighted|prefix-affinity:1,queue-depth:3,kv-utilization:2
round-robin|round-robin|none
least-loaded|least-loaded|none
"

for seed in $SEEDS; do
    echo "--- Seed $seed ---" >&2
    for config_line in $CONFIGS; do
        label=$(echo "$config_line" | cut -d'|' -f1)
        policy=$(echo "$config_line" | cut -d'|' -f2)
        scorers=$(echo "$config_line" | cut -d'|' -f3)
        [[ -z "$label" ]] && continue
        for workload in prefix indep mixed; do
            echo "  $label/$workload/seed$seed..." >&2
            run_sim "$label" "$workload" "$seed" "$policy" "$scorers"
        done
    done
done

echo "" >&2
echo "=== All runs complete ===" >&2

# Run analysis
python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR"

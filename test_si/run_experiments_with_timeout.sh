#!/bin/bash
set -e

# Parameters
ARRIVAL_WINDOW=600    # 10 minutes
TIMEOUT_US=660000000  # 11 minutes in microseconds
HORIZON_US=1260000000 # 21 minutes in microseconds

# Load ratios to test
RATIOS=(0.01 0.015 0.02 0.025 0.03 0.04 0.05 0.1 0.15 0.2 0.3 0.5 1.0)

echo "===== Experiment Configuration ====="
echo "Arrival window: ${ARRIVAL_WINDOW}s (10 min)"
echo "Request timeout: $((TIMEOUT_US / 1000000))s (11 min)"
echo "Horizon: $((HORIZON_US / 1000000))s (21 min)"
echo "Load ratios: ${RATIOS[@]}"
echo ""

# Generate scaled workloads
echo "===== Generating Workloads ====="
for ratio in "${RATIOS[@]}"; do
    echo "Generating workload-${ratio}x.yaml..."
    python3 << PYTHON
import yaml

# Load base workload
with open('m-mid-midnight.yaml') as f:
    data = yaml.safe_load(f)

# Scale trace_rate for each cohort
scale_factor = ${ratio}
for cohort in data['cohorts']:
    if 'spike' in cohort and 'trace_rate' in cohort['spike']:
        cohort['spike']['trace_rate'] *= scale_factor

# Duration is already 600s in the base file

with open('test_si/workload-${ratio}x.yaml', 'w') as f:
    yaml.dump(data, f, default_flow_style=False)

print(f"Created workload-${ratio}x.yaml")
PYTHON
done

echo ""
echo "===== Running BLIS Experiments ====="

for ratio in "${RATIOS[@]}"; do
    echo ""
    echo ">>> Running load ${ratio}x..."
    ./blis run \
        --workload-spec test_si/workload-${ratio}x.yaml \
        --timeout 660 \
        --horizon ${HORIZON_US} \
        --model qwen/qwen3-14b \
        --log error \
        --metrics-path test_si/metrics-${ratio}x.json \
        > /dev/null 2>&1

    # Quick sanity check
    if [ -f test_si/metrics-${ratio}x.json ]; then
        completed=$(jq -r '[.requests[] | select(.e2e_ms > 0)] | length' test_si/metrics-${ratio}x.json 2>/dev/null || echo "0")
        total=$(jq -r '.requests | length' test_si/metrics-${ratio}x.json 2>/dev/null || echo "0")
        echo "    Total: ${total}, Completed: ${completed}"
    else
        echo "    ERROR: metrics file not created"
    fi
done

echo ""
echo "===== Experiments Complete ====="
echo "Results saved to test_si/metrics-*.json"

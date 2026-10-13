#!/usr/bin/env bash
#
# find-saturation.sh — Rate-sweep saturation finder.
#
# Drives `blis run` across a configurable rate sweep, records throughput and
# every post-hoc detector's final verdict per rate (via the detector bank,
# #1519), and prints a single table summarizing the saturation envelope.
# The deployment (model, hardware, parallelism, engine settings) comes from a
# blis-schemas scenario file; blis-latency-kernel prices every step.
#
# All inputs are environment variables. SCENARIO, SCENARIOS and REGISTRY are
# required; the rest have defaults.
# See scripts/README.md for the full input table and worked examples.
#
# Output:
#   - $OUT_DIR/summary.csv — one row per rate
#   - $OUT_DIR/rate-{R}.{json,stderr} — raw blis run output
#   - $OUT_DIR/rate-{R}.saturation.json — {"final":{...},"trace":[...]} report

set -euo pipefail
cd "$(dirname "$0")/.."

# Scenario file name, the directory holding it, and the blis-registry clone root. All three are
# required by blis run; there are no defaults.
SCENARIO="${SCENARIO:-}"
SCENARIOS="${SCENARIOS:-}"
REGISTRY="${REGISTRY:-}"
for v in SCENARIO SCENARIOS REGISTRY; do
  if [[ -z "${!v}" ]]; then
    echo "error: $v is not set. find-saturation.sh needs SCENARIO (a scenario file name)," >&2
    echo "SCENARIOS (the directory holding it) and REGISTRY (a blis-registry clone root)." >&2
    echo "See the find-saturation.sh section of scripts/README.md." >&2
    exit 1
  fi
done
# Catalog clone root (blis-catalog at its pinned release). Required by blis run -- there is no
# default and no search path. Falls back to BLIS_CATALOG, then ./blis-catalog.
CATALOG="${CATALOG:-${BLIS_CATALOG:-blis-catalog}}"
WORKLOAD="${WORKLOAD:-chatbot}"
NUM_REQUESTS="${NUM_REQUESTS:-6000}"
HORIZON_US="${HORIZON_US:-600000000}"   # 600s
# Trailing window for the stdout/report final-label plurality vote (#1517).
FINAL_WINDOW="${FINAL_WINDOW:-10s}"
# Which detectors to run. "all" ⇒ every detector in the roster.
DETECTORS="${DETECTORS:-all}"
RATES="${RATES:-0.5 1 2 4 6 8 10 12 14 16 20 30 40 50 60 80 100}"
SEED="${SEED:-42}"

# --catalog is always passed: blis refuses a run with neither --catalog nor BLIS_CATALOG.
# The scenario's model must be in the catalog; a model with no entry is refused (NS-6).
CFG_ARGS=(--catalog "$CATALOG" --scenarios "$SCENARIOS" --registry "$REGISTRY")

OUT_DIR="${OUT_DIR:-results/saturation-$(date +%Y%m%d-%H%M%S)-$$}"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.csv"

# Build blis once if needed
if [[ ! -x ./blis ]]; then
  echo "Building blis..."
  go build -o blis main.go
fi

echo "intended_rate,sustained_throughput,goodput_rps,goodput_vs_intended,timeout_frac,e2e_p99_ms,ttft_p99_ms,still_queued,still_running,composite_verdict,threshold_verdict,backlog_drift_verdict,peak_rate_verdict" > "$SUMMARY"
printf "Scenario:  %s (from %s)\n" "$SCENARIO" "$SCENARIOS"
printf "Workload:  %s\n" "$WORKLOAD"
printf "Detectors: %s (final window %s)\n" "$DETECTORS" "$FINAL_WINDOW"
printf "Sweeping:  %s req/s\n" "$RATES"
printf "Output:    %s\n\n" "$OUT_DIR"

# jq helper: read one detector's final label from the report, or "n/a" if the
# detector wasn't selected.
final_label() {
  local report="$1" detector="$2"
  jq -r --arg d "$detector" '.final[$d] // "n/a"' "$report"
}

for R in $RATES; do
  RAW="$OUT_DIR/rate-${R}.json"
  LOG="$OUT_DIR/rate-${R}.stderr"
  SAT_REPORT="$OUT_DIR/rate-${R}.saturation.json"

  printf "rate=%-5s ... " "$R"

  # One deterministic run per rate: the detector bank fans the same replay out
  # to every selected detector, so all verdicts come from a single pass.
  ./blis run \
    --scenario "$SCENARIO" \
    "${CFG_ARGS[@]}" \
    --workload "$WORKLOAD" \
    --rate "$R" \
    --num-requests "$NUM_REQUESTS" \
    --horizon "$HORIZON_US" \
    --seed "$SEED" \
    --detectors "$DETECTORS" \
    --saturation-final-window "$FINAL_WINDOW" \
    --saturation-report "$SAT_REPORT" \
    > "$RAW" 2> "$LOG"

  # Extract throughput stats from the run's stdout JSON
  METRICS=$(awk '/^=== Simulation Metrics ===/{flag=1; next} flag' "$RAW")
  read -r OFF GOOD TIMEOUT_FRAC E2E_P99 TTFT_P99 SQ SR <<<"$(jq -r '
    def n: . // 0;
    [
      (if (.vllm_estimated_duration_s | n) > 0 then ((.injected_requests | n) / .vllm_estimated_duration_s) else 0 end),
      (.responses_per_sec | n),
      (if (.injected_requests | n) > 0 then ((.timed_out_requests | n) / .injected_requests) else 0 end),
      (.e2e_p99_ms | n),
      (.ttft_p99_ms | n),
      (.still_queued | n),
      (.still_running | n)
    ] | @tsv' <<<"$METRICS")"

  # Extract each detector's final verdict from the saturation report's "final" map.
  COMPOSITE_VERDICT=$(final_label "$SAT_REPORT" composite)
  THRESHOLD_VERDICT=$(final_label "$SAT_REPORT" threshold)
  BACKLOG_DRIFT_VERDICT=$(final_label "$SAT_REPORT" backlog-drift)
  PEAK_RATE_VERDICT=$(final_label "$SAT_REPORT" peak-rate)

  RATIO=$(echo "scale=4; $GOOD / $R" | bc -l)
  printf "goodput=%6.2f  ratio=%5.1f%%  composite: %-11s  threshold: %-11s  backlog-drift: %-11s  peak-rate: %s\n" \
    "$GOOD" "$(echo "$RATIO * 100" | bc -l)" "$COMPOSITE_VERDICT" "$THRESHOLD_VERDICT" "$BACKLOG_DRIFT_VERDICT" "$PEAK_RATE_VERDICT"

  echo "$R,$OFF,$GOOD,$RATIO,$TIMEOUT_FRAC,$E2E_P99,$TTFT_P99,$SQ,$SR,$COMPOSITE_VERDICT,$THRESHOLD_VERDICT,$BACKLOG_DRIFT_VERDICT,$PEAK_RATE_VERDICT" >> "$SUMMARY"
done

printf "\nDone. Summary CSV: %s\n" "$SUMMARY"
printf "Per-rate raw + saturation reports in: %s/\n\n" "$OUT_DIR"
printf "Saturation knee = first rate where:\n"
printf "  - goodput_rps stops tracking intended_rate (ratio drops below 100%%), OR\n"
printf "  - a detector's final verdict flips to OVERLOADED.\n\n"
column -t -s, "$SUMMARY"

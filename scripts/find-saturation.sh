#!/usr/bin/env bash
#
# find-saturation.sh — Rate-sweep saturation finder.
#
# Drives `blis run` across a configurable rate sweep, records throughput and
# every post-hoc detector's final verdict per rate (via the detector bank,
# #1519), and prints a single table summarizing the saturation envelope.
# Reproduces the Llama-3.1-70B / TP=8 / H100 / chatbot reference validation.
#
# All inputs are environment variables; defaults match the reference run.
# See scripts/README.md for the full input table and worked examples.
#
# Output:
#   - $OUT_DIR/summary.csv — one row per rate
#   - $OUT_DIR/rate-{R}.{json,stderr} — raw blis run output
#   - $OUT_DIR/rate-{R}.saturation.json — {"final":{...},"trace":[...]} report

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
MODEL_CONFIG_FOLDER="${MODEL_CONFIG_FOLDER:-model_configs/llama-3.1-70b-instruct}"
HARDWARE="${HARDWARE:-H100}"
TP="${TP:-8}"
WORKLOAD="${WORKLOAD:-chatbot}"
LATENCY_MODEL="${LATENCY_MODEL:-trained-physics}"
NUM_REQUESTS="${NUM_REQUESTS:-6000}"
HORIZON_US="${HORIZON_US:-600000000}"   # 600s
# Trailing window for the stdout/report final-label plurality vote (#1517).
FINAL_WINDOW="${FINAL_WINDOW:-10s}"
# Which detectors to run. "all" ⇒ composite,threshold,backlog-drift.
DETECTORS="${DETECTORS:-all}"
RATES="${RATES:-0.5 1 2 4 6 8 10 12 14 16 20 30 40 50 60 80 100}"
SEED="${SEED:-42}"

# Pass --model-config-folder only if non-empty (allows MODEL_CONFIG_FOLDER="" to disable
# and force HuggingFace auto-fetch — useful for non-bundled models).
CFG_ARGS=()
[[ -n "$MODEL_CONFIG_FOLDER" ]] && CFG_ARGS=(--model-config-folder "$MODEL_CONFIG_FOLDER")

OUT_DIR="${OUT_DIR:-results/saturation-$(date +%Y%m%d-%H%M%S)-$$}"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.csv"

# Build blis once if needed
if [[ ! -x ./blis ]]; then
  echo "Building blis..."
  go build -o blis main.go
fi

echo "intended_rate,sustained_throughput,goodput_rps,goodput_vs_intended,timeout_frac,e2e_p99_ms,ttft_p99_ms,still_queued,still_running,composite_verdict,threshold_verdict,backlog_drift_verdict" > "$SUMMARY"
printf "Model:     %s (TP=%d, %s)\n" "$MODEL" "$TP" "$HARDWARE"
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
    --model "$MODEL" \
    "${CFG_ARGS[@]}" \
    --hardware "$HARDWARE" \
    --tp "$TP" \
    --latency-model "$LATENCY_MODEL" \
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

  RATIO=$(echo "scale=4; $GOOD / $R" | bc -l)
  printf "goodput=%6.2f  ratio=%5.1f%%  composite: %-11s  threshold: %-11s  backlog-drift: %s\n" \
    "$GOOD" "$(echo "$RATIO * 100" | bc -l)" "$COMPOSITE_VERDICT" "$THRESHOLD_VERDICT" "$BACKLOG_DRIFT_VERDICT"

  echo "$R,$OFF,$GOOD,$RATIO,$TIMEOUT_FRAC,$E2E_P99,$TTFT_P99,$SQ,$SR,$COMPOSITE_VERDICT,$THRESHOLD_VERDICT,$BACKLOG_DRIFT_VERDICT" >> "$SUMMARY"
done

printf "\nDone. Summary CSV: %s\n" "$SUMMARY"
printf "Per-rate raw + saturation reports in: %s/\n\n" "$OUT_DIR"
printf "Saturation knee = first rate where:\n"
printf "  - goodput_rps stops tracking intended_rate (ratio drops below 100%%), OR\n"
printf "  - a detector's final verdict flips to OVERLOADED.\n\n"
column -t -s, "$SUMMARY"

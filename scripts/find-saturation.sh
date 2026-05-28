#!/usr/bin/env bash
#
# find-saturation.sh — Rate-sweep saturation finder.
#
# Drives `blis run` across a configurable rate sweep, records throughput and
# both classifier verdicts (drain-ratio + slope-based) per rate, and prints a
# single table summarizing the saturation envelope. Reproduces the Llama-3.1-70B
# / TP=8 / H100 / chatbot reference validation from PR #1395.
#
# All inputs are environment variables; defaults match the PR's reference run.
# See scripts/README.md for the full input table and worked examples.
#
# Output:
#   - $OUT_DIR/summary.csv — one row per rate
#   - $OUT_DIR/rate-{R}.{json,stderr} — raw blis run output
#   - $OUT_DIR/rate-{R}.{drain-ratio,slope-based}.json — classifier reports

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
SATURATION_WINDOW_S="${SATURATION_WINDOW_S:-10}"
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

echo "intended_rate,sustained_throughput,goodput_rps,goodput_vs_intended,timeout_frac,e2e_p99_ms,ttft_p99_ms,still_queued,still_running,drain_ratio_verdict,drain_ratio_rho,slope_based_verdict" > "$SUMMARY"
printf "Model:    %s (TP=%d, %s)\n" "$MODEL" "$TP" "$HARDWARE"
printf "Workload: %s\n" "$WORKLOAD"
printf "Sweeping: %s req/s\n" "$RATES"
printf "Output:   %s\n\n" "$OUT_DIR"

run_blis() {
  local rate="$1" classifier="$2" report_path="$3" raw_path="$4" log_path="$5"
  ./blis run \
    --model "$MODEL" \
    "${CFG_ARGS[@]}" \
    --hardware "$HARDWARE" \
    --tp "$TP" \
    --latency-model "$LATENCY_MODEL" \
    --workload "$WORKLOAD" \
    --rate "$rate" \
    --num-requests "$NUM_REQUESTS" \
    --horizon "$HORIZON_US" \
    --seed "$SEED" \
    --saturation-window "$SATURATION_WINDOW_S" \
    --saturation-classifier "$classifier" \
    --saturation-report "$report_path" \
    > "$raw_path" 2> "$log_path"
}

for R in $RATES; do
  RAW="$OUT_DIR/rate-${R}.json"
  LOG="$OUT_DIR/rate-${R}.stderr"
  DR_REPORT="$OUT_DIR/rate-${R}.drain-ratio.json"
  SB_REPORT="$OUT_DIR/rate-${R}.slope-based.json"

  printf "rate=%-5s ... " "$R"

  # Run with drain-ratio (default classifier; throughput numbers come from this run)
  run_blis "$R" "drain-ratio" "$DR_REPORT" "$RAW" "$LOG"

  # Run again with slope-based classifier — same workload + seed, only verdict differs
  run_blis "$R" "slope-based" "$SB_REPORT" "$OUT_DIR/.tmp.raw" "$OUT_DIR/.tmp.log"
  rm -f "$OUT_DIR/.tmp.raw" "$OUT_DIR/.tmp.log"

  # Extract throughput stats from the drain-ratio run's stdout JSON
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

  # Extract verdicts from the saturation reports
  DR_VERDICT=$(jq -r .classification "$DR_REPORT")
  DR_RHO=$(jq -r '.note | capture("ρ ≈ (?<r>[0-9.]+)").r // "n/a"' "$DR_REPORT")
  SB_VERDICT=$(jq -r .classification "$SB_REPORT")

  RATIO=$(echo "scale=4; $GOOD / $R" | bc -l)
  printf "goodput=%6.2f  ratio=%5.1f%%  ρ=%-5s  drain-ratio: %-23s  slope-based: %s\n" \
    "$GOOD" "$(echo "$RATIO * 100" | bc -l)" "$DR_RHO" "$DR_VERDICT" "$SB_VERDICT"

  echo "$R,$OFF,$GOOD,$RATIO,$TIMEOUT_FRAC,$E2E_P99,$TTFT_P99,$SQ,$SR,$DR_VERDICT,$DR_RHO,$SB_VERDICT" >> "$SUMMARY"
done

printf "\nDone. Summary CSV: %s\n" "$SUMMARY"
printf "Per-rate raw + classifier reports in: %s/\n\n" "$OUT_DIR"
printf "Saturation knee = first rate where:\n"
printf "  - goodput_rps stops tracking intended_rate (ratio drops below 100%%), OR\n"
printf "  - drain_ratio_verdict flips to PERSISTENTLY_SATURATED, OR\n"
printf "  - slope_based_verdict flips to PERSISTENTLY_SATURATED.\n\n"
column -t -s, "$SUMMARY"

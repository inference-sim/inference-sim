#!/usr/bin/env bash
# Original to blis-sglang.
# What ci-light.yml runs for a stacked layer, chosen from the files the layer changes (one path per line on stdin).
#   ci-light-select.sh packages   the Go packages to test: the root package (the architecture, import, field and
#                                 provenance tests live there) and each package that holds a changed file, found by
#                                 walking up from the file to the nearest directory with a Go file. A change to go.mod
#                                 or go.sum selects every package (./...).
#   ci-light-select.sh checks     two lines, fields=true|false and oracle-checks=true|false: whether the generated-file
#                                 checks of the field list, or the oracle's guards over this repository's files, are
#                                 reached by a change to what they read.
#   ci-light-select.sh groups     the oracle groups the layer reaches, as a JSON array for a matrix: resolve, lower,
#                                 sched, cache, hicache-metrics. A group is reached when a file its runners, scenarios,
#                                 goldens or answers are made of changes, or when a file every group shares changes.
# The metrics goldens name oracle/run_sched.py among the files their run executed, so a change to it reaches the
# hicache-metrics group as well as sched.
# A change to Go code alone reaches no group: go test of its package judges it against the committed goldens, and the
# goldens are compared with SGLang's output by the layer that changes them or the oracle that produces them.
set -euo pipefail
mode="${1:?usage: ci-light-select.sh packages|groups|checks < changed-files}"
files="$(cat)"

case "$mode" in
packages)
  if printf '%s\n' "$files" | grep -qE '^go\.(mod|sum)$'; then echo "./..."; exit 0; fi
  pkgs="."
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    d="$(dirname "$f")"
    while [ "$d" != "." ] && ! ls "$d"/*.go >/dev/null 2>&1; do d="$(dirname "$d")"; done
    [ "$d" = "." ] || pkgs="$pkgs ./$d"
  done <<< "$files"
  printf '%s\n' $pkgs | LC_ALL=C sort -u | paste -sd' ' -
  ;;
groups)
  all='^(oracle-env\.lock|provenance\.go|\.github/workflows/ci-light\.yml|\.github/actions/oracle-env/|oracle/(harness|traces|mocks|requirements|check_(mocks|lock|peers|modules))[^/]*$|oracle/testdata/)'
  pattern() {
    case "$1" in
    resolve) echo '^(config/testdata/serverargs\.json|oracle/(run_serverargs|compare_serverargs|run_import)\.py|oracle/serverargs/)' ;;
    lower) echo '^(lower/testdata/|oracle/lower/|oracle/serverargs/|oracle/run_scheduler\.py|oracle/scenarios/[^/]*\.json$|oracle/testdata/golden/)' ;;
    sched) echo '^(core/sched/testdata/|oracle/(run_sched|gen_fuzz)\.py|oracle/scenarios-sched/)' ;;
    cache) echo '^(core/cache/testdata/|oracle/(run_cache_iface|run_dfs_order|run_pools|cache_interfaces)\.py|oracle/scenarios/(cache/|[^/]*\.json$)|oracle/scenarios-(cache-iface|dfs-order|pools)/)' ;;
    hicache-metrics) echo '^(core/(hicache|metrics)/testdata/|oracle/(run_hicache[^/]*|run_host_pools|run_metrics|run_sched|step_runner)\.py|oracle/scenarios/hicache|oracle/scenarios-(metrics|host-pools)/)' ;;
    esac
  }
  everything=0
  if printf '%s\n' "$files" | grep -qE "$all"; then everything=1; fi
  out=""
  for g in resolve lower sched cache hicache-metrics; do
    if [ "$everything" = 1 ] || printf '%s\n' "$files" | grep -qE "$(pattern "$g")"; then out="$out\"$g\","; fi
  done
  echo "[${out%,}]"
  ;;
checks)
  self='^(provenance\.go|\.github/workflows/ci-light\.yml)$'
  f='^(fields/|fields\.json$|config/(gen_serverargs\.py|serverargs_gen\.go)$)'
  c='^(oracle/|tools/surface/|surface\.json$|docs/metrics-|cmd/metricsmanifest/|core/metrics/[^/]*\.go$)'
  for pair in "fields:$f" "oracle-checks:$c"; do
    n="${pair%%:*}"; r="${pair#*:}"
    if printf '%s\n' "$files" | grep -qE "$self|$r"; then echo "$n=true"; else echo "$n=false"; fi
  done
  ;;
*) echo "unknown mode $mode" >&2; exit 2 ;;
esac

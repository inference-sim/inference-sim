#!/usr/bin/env bash
#
# archon-review.sh — run archon's PR review and compose the comment body.
#
# Environment (all required):
#   ARCHON_BIN   path to the archon-go binary
#   BASE_SHA     the PR's base branch tip
#   HEAD_SHA     the PR's head commit (already fetched into the object store)
#   DECL_FILE    file holding plan-declaration candidate text
#   OUTPUT_FILE  where to write the comment body
#   RUN_URL      workflow run URL, used in the truncation notice
# Optional:
#   GITHUB_STEP_SUMMARY  appended to when set
#
# Exits 0 for every reviewed outcome, including archon failure, so the caller always has a
# body to post. Exit 2 only on a usage or environment error.
#
# This script is executed from the DEFAULT BRANCH checkout: the workflow fetches the PR head
# into the object store but never checks it out. A pull request therefore cannot get its own
# copy of this script, the resolver, or .archon-version executed. Adding a `ref:` to the
# checkout step would turn that property into arbitrary code execution on the runner.
#
# `set -e` is deliberately off: archon's non-zero exit is an outcome to report, not a crash.
# Every state-changing command is therefore checked explicitly.

set -uo pipefail

readonly MAX_COMMENT_CHARS=60000   # GitHub's limit is 65536; leave room for the notice

usage() { echo "usage: $0 (required environment variable $1 is unset)" >&2; exit 2; }

# Absolutise the caller's file arguments before anchoring to the repository root.
abspath() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$PWD" "$1" ;; esac; }

# OUTPUT_FILE is handled first so it can be truncated before anything else can fail. The
# runner is self-hosted and the path is fixed, so a body left by an earlier run must not be
# posted if this one exits before composing its own.
[[ -n "${OUTPUT_FILE:-}" ]] || usage OUTPUT_FILE
OUTPUT_FILE=$(abspath "$OUTPUT_FILE")
: > "$OUTPUT_FILE" || { echo "could not write to $OUTPUT_FILE" >&2; exit 2; }

for var in ARCHON_BIN BASE_SHA HEAD_SHA DECL_FILE RUN_URL; do
  [[ -n "${!var:-}" ]] || usage "$var"
done

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) \
  || { echo "could not locate the script directory" >&2; exit 2; }

DECL_FILE=$(abspath "$DECL_FILE")

repo_root=$(git rev-parse --show-toplevel) \
  || { echo "not inside a git repository" >&2; exit 2; }
cd "$repo_root" || { echo "could not enter the repository root" >&2; exit 2; }

# emit <body> — write the comment body and the job summary, then stop. The single exit point
# for a composed body, so neither the summary append nor the truncation can be skipped.
emit() {
  local body="$1"
  local full_len=${#1}
  [[ -z "${GITHUB_STEP_SUMMARY:-}" ]] || printf '%s\n' "$body" >> "$GITHUB_STEP_SUMMARY"
  if (( full_len > MAX_COMMENT_CHARS )); then
    body="${body:0:MAX_COMMENT_CHARS}

_Output truncated (${full_len} chars). See the [workflow run](${RUN_URL}) for the full report._"
  fi
  printf '%s\n' "$body" > "$OUTPUT_FILE" \
    || { echo "could not write the comment body to $OUTPUT_FILE" >&2; exit 2; }
  exit 0
}

MERGE_BASE=$(git merge-base "$BASE_SHA" "$HEAD_SHA" 2>/dev/null) || emit "$(printf '## Archon Error\n\nCould not compute merge-base between base (%s) and head (%s). The PR head may not be reachable — check that the fetch step succeeded.' "$BASE_SHA" "$HEAD_SHA")"

WORK_DIR=$(mktemp -d "${RUNNER_TEMP:-/tmp}/archon-review.XXXXXX") \
  || { echo "could not create a work directory" >&2; exit 2; }
trap 'rm -rf "$WORK_DIR"' EXIT INT TERM
PLAN_FILE="$WORK_DIR/plan.json"
STATUS_FILE="$WORK_DIR/plan-status.txt"

# Guarded: a missing, non-executable, or crashing resolver must degrade to a delta review
# with a warning, never abort the step and leave the trigger with no comment. Resolver
# stderr is left on the job log, since the synthesised message points a reader there.
"$script_dir/archon-plan-resolve.sh" "$BASE_SHA" "$HEAD_SHA" "$DECL_FILE" "$PLAN_FILE" > "$STATUS_FILE" \
  || printf 'status=error\nplan_path=\nmessage=plan resolution failed to run; see the workflow logs\n' > "$STATUS_FILE"

plan_field() { sed -n "s/^$1=//p" "$STATUS_FILE"; }
PLAN_STATUS=$(plan_field status)
PLAN_NOTE=""

# A fresh output directory per invocation: a partial bundle from a failed plan-aware run
# must never be posted as though it were the delta review.
run_review() {
  rm -rf .archon || return 1
  "$ARCHON_BIN" pr-review . "$MERGE_BASE" "$HEAD_SHA" --out .archon "$@"
}

case "$PLAN_STATUS" in
  resolved)
    if [[ "$(plan_field plan_source)" == "head" ]]; then
      PLAN_NOTE=$(printf '> [!NOTE]\n> Plan check used `%s` taken from this PR'"'"'s own head (`%s`). The base branch carries no copy, so the plan itself is not independently verified.' \
        "$(plan_field plan_path)" "$(plan_field plan_commit)")
    else
      PLAN_NOTE=$(printf '_Plan check: `%s` from the base branch tip (`%s`)._' \
        "$(plan_field plan_path)" "$(plan_field plan_commit)")
    fi
    if ! run_review --plan "$PLAN_FILE"; then
      PLAN_NOTE=$(printf '> [!WARNING]\n> Plan-aware review failed for `%s`; reporting the delta review only. No dist ratchet or plan verdict in this comment.' \
        "$(plan_field plan_path)")
      PLAN_STATUS=fallback
    fi
    ;;
  none) ;;
  error)
    PLAN_NOTE=$(printf '> [!WARNING]\n> Archon plan check skipped: %s. Reviewed without `--plan`, so this comment carries no dist ratchet or plan verdict.' \
      "$(plan_field message)")
    ;;
  *)
    echo "::error::archon-plan-resolve.sh returned an unknown status: ${PLAN_STATUS:-<empty>}" >&2
    PLAN_NOTE=$(printf '> [!WARNING]\n> Archon plan detection returned an unrecognised result and was skipped. Reviewed without `--plan`, so this comment carries no dist ratchet or plan verdict.')
    PLAN_STATUS=error
    ;;
esac

# with_note <body> — prepend the plan note when there is one. Applied to the failure bodies
# too, so the reason a declared plan was not checked is never dropped.
with_note() {
  if [[ -z "$PLAN_NOTE" ]]; then printf '%s' "$1"; else printf '%s\n\n%s' "$PLAN_NOTE" "$1"; fi
}

if [[ "$PLAN_STATUS" != "resolved" ]]; then
  run_review || emit "$(with_note "$(printf '## Archon Error\n\narchon-go pr-review failed. Check the workflow logs.')")"
fi

[[ -f .archon/review.md ]] \
  || emit "$(with_note "$(printf '## Archon Error\n\narchon-go exited 0 but did not produce .archon/review.md.')")"

emit "$(with_note "$(cat .archon/review.md)")"

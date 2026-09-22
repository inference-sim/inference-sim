#!/usr/bin/env bash
#
# deliver-author-gate.sh — decide whether an AI flow may run on content authored by <login>.
#
# Usage:
#   scripts/deliver-author-gate.sh <login>                    # network form (bounded gh probe)
#   scripts/deliver-author-gate.sh --permission <perm> <login>  # test seam, makes no network call
#
# WHY THIS EXISTS (#1813). Every AI flow — the bot delivery loop (implement/verify/correct), the
# interactive @claude, /blis-pr-review — can be pointed at a PR or issue authored by someone WITHOUT
# write access. On such an item the untrusted content is not just the comments (#1806 filters those
# for the delivery loop) — it is the PR/issue BODY and, on a PR, the DIFF itself. Reading that
# outside-authored body/diff IS the task, and it flows straight into an agent running on a persistent
# self-hosted runner with credentials (and, in the correction phase, contents: write). A maintainer
# TRIGGERING the flow is not a safeguard: triggering does not sanitize the body/diff.
#
# So the trust signal is the AUTHOR of the PR/issue, not fork-ness: a same-repo branch opened by an
# outsider is as untrusted as a fork, and a fork opened by a maintainer is fine. This gate is the
# container-level complement to #1806's comment-text filter, and to the TRIGGERER gate
# (check-permissions / authorize) that every flow already runs — it is ANDed with that, never a
# replacement.
#
# ── The decision ─────────────────────────────────────────────────────────────────────────────────
#
# It is the same law sibling deliver-issue-refinements.sh applies to comment authors — a login
# counts iff it holds admin/write/maintain — with two differences suited to a single-author gate:
#   * a login on the BOT ALLOWLIST is trusted WITHOUT a probe, because the delivery loop's own PRs
#     and issues are bot-authored and the permission endpoint 404s for a bot login (probing would
#     deny exactly the automation this must let through);
#   * a clean 404 is a DEFINITIVE denial (not a collaborator), distinct from a 5xx/403/network/
#     deadline failure which means the probe could not answer. The first blocks; the second fails
#     closed AND loud (R1) so a probe outage is never mistaken for either an answer or a silent drop.
#
# Outcomes (stdout is machine-readable; detail on stderr):
#   AUTHOR-GATE: allowed …   exit 0  → the caller runs the agent
#   AUTHOR-GATE: blocked …   exit 1  → definitive denial; the caller refuses and says why
#   AUTHOR-GATE-READ-FAILED  exit 3  → probe could not answer; the caller refuses AND reports a
#                                       probe failure (retry), never a denial
#   usage                    exit 2
#
# The bot allowlist is a LOGIN allowlist, deliberately not "any login ending in [bot]": a
# third-party App (e.g. dependabot[bot]) authoring a fork PR must be probed and denied, not waved
# through on its suffix. Matches the two identities the delivery loop actually authors with —
# github.token pushes/opens as github-actions[bot]; claude-code-action posts as claude[bot].

set -uo pipefail

SELF="$0"

# github-actions[bot] and claude[bot] are trusted automation. See the header for why this is an
# explicit login list rather than a `[bot]`-suffix test.
is_trusted_bot() {
  case "$1" in
    'github-actions[bot]' | 'claude[bot]') return 0 ;;
    *) return 1 ;;
  esac
}

# A collaborator login is 1+ chars of [A-Za-z0-9-]; only such a login is ever put in a probe URL.
probeable_login() { [[ "$1" =~ ^[A-Za-z0-9-]+$ ]]; }

# A bot-shaped login (`<name>[bot]`) that is not on the trusted allowlist — a third-party App such
# as dependabot[bot] authoring a fork PR. It is a definitive denial WITHOUT a probe: an App is not a
# collaborator, and this avoids putting the `[ ]` of a bot login into a probe URL at all.
bot_shaped_login() { [[ "$1" =~ ^[A-Za-z0-9-]+\[bot\]$ ]]; }

allowed() { printf 'AUTHOR-GATE: allowed — %s\n' "$1"; exit 0; }
blocked() {
  printf 'AUTHOR-GATE: blocked — %s\n' "$1"
  echo "$SELF: refusing to run an AI flow on content authored by an account without write access: $1" >&2
  exit 1
}
read_failed() {
  printf 'AUTHOR-GATE-READ-FAILED\n\n'
  printf 'Could not verify the author'"'"'s repository permission: %s\n' "$1"
  printf 'Failing closed — this is a probe failure, not a permission decision; retrying should work.\n'
  echo "$SELF: $1" >&2
  exit 3
}
usage() {
  echo "usage: $SELF <login> | $SELF --permission <perm> <login>" >&2
  exit 2
}

# Parse args: either "<login>" or "--permission <perm> <login>". The seam lets the tests exercise
# the decision without a network call, and lets a caller that has ALREADY resolved a permission
# (rare) reuse this decision.
INJECTED_PERM=""
HAVE_INJECTED=false
if [[ "${1:-}" == "--permission" ]]; then
  [[ $# -eq 3 ]] || usage
  INJECTED_PERM="$2"
  HAVE_INJECTED=true
  LOGIN="$3"
else
  [[ $# -eq 1 ]] || usage
  LOGIN="$1"
fi

# Empty login is a read failure, never a fall-through to a probe of an empty path.
if [[ -z "$LOGIN" ]]; then
  read_failed "no author login was supplied (empty) — cannot decide, so refusing"
fi

# Bots are classified BEFORE anything else (see header): the two trusted ones pass without a probe;
# any other bot-shaped login is a definitive denial without a probe.
if is_trusted_bot "$LOGIN"; then
  allowed "trusted automation ($LOGIN)"
fi
if bot_shaped_login "$LOGIN"; then
  blocked "@$LOGIN is a bot that is not trusted automation for this repo"
fi

# Only a plain collaborator-shaped login may proceed to a probe. Anything else — spaces, shell
# metacharacters, path separators — is refused before it can reach a URL or the shell.
if ! probeable_login "$LOGIN"; then
  read_failed "author login is not a valid GitHub login: '$LOGIN'"
fi

# The offline seam: decide directly from the injected permission, no network.
decide_from_permission() {
  case "$1" in
    admin | write | maintain) allowed "@$LOGIN has '$1' access" ;;
    *) blocked "@$LOGIN has '$1' access" ;;
  esac
}

if [[ "$HAVE_INJECTED" == true ]]; then
  decide_from_permission "$INJECTED_PERM"
fi

# ── Network form ───────────────────────────────────────────────────────────────────────────────
command -v gh >/dev/null 2>&1 || read_failed "gh is not on PATH"

# Bound every gh call so a stalled probe cannot hang the delivery job until its 60/120-minute
# timeout. Same idiom as deliver-issue-refinements.sh: `timeout`/`gtimeout` where present, else a
# portable watchdog; SIGTERM escalated to SIGKILL after a grace so a TERM-resistant call is still
# bounded. Overridable so the tests can force short values.
GH_DEADLINE_SECONDS="${GH_DEADLINE_SECONDS:-30}"
GH_KILL_GRACE_SECONDS="${GH_KILL_GRACE_SECONDS:-5}"
_GH_BIN="$(command -v gh)"
if command -v timeout >/dev/null 2>&1; then
  _GH_TIMEOUT="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
  _GH_TIMEOUT="gtimeout"
else
  _GH_TIMEOUT=""
fi

run_bounded() {
  local secs="$1"; shift
  "$@" &
  local pid=$!
  (
    sleep "$secs"
    kill -TERM "$pid" 2>/dev/null
    sleep "$GH_KILL_GRACE_SECONDS"
    kill -KILL "$pid" 2>/dev/null
  ) >/dev/null 2>&1 &
  local watcher=$!
  wait "$pid" 2>/dev/null
  local rc=$?
  kill -KILL "$watcher" 2>/dev/null
  wait "$watcher" 2>/dev/null
  return "$rc"
}

gh() {
  if [[ -n "$_GH_TIMEOUT" ]]; then
    "$_GH_TIMEOUT" --kill-after="$GH_KILL_GRACE_SECONDS" "$GH_DEADLINE_SECONDS" "$_GH_BIN" "$@"
  else
    run_bounded "$GH_DEADLINE_SECONDS" "$_GH_BIN" "$@"
  fi
}

REPO="${GH_REPO:-${GITHUB_REPOSITORY:-}}"
if [[ -z "$REPO" ]]; then
  REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner) \
    || read_failed "could not determine the repository (set GH_REPO or GITHUB_REPOSITORY)"
fi

ERR=$(mktemp) || read_failed "could not create a temporary file"
trap 'rm -f "$ERR"' EXIT

# A 404 is a real answer — GitHub says this login is not a collaborator (and a non-user login such
# as a deleted account returns it too) — whereas a 401/403/5xx/network failure or a deadline expiry
# from the bounded wrapper means the probe could not ask. The split is the difference between a
# denial and a fail-closed-loud, so it must not be collapsed.
if perm=$(gh api "repos/$REPO/collaborators/$LOGIN/permission" --jq '.permission' 2>"$ERR"); then
  decide_from_permission "$perm"
fi
if grep -qi 'HTTP 404' "$ERR"; then
  blocked "@$LOGIN is not a collaborator (HTTP 404)"
fi
read_failed "the permission probe for @$LOGIN failed: $(tr '\n' ' ' < "$ERR")"

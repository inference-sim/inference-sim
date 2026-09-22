# shellcheck shell=bash
#
# lib-gh-write-access.sh — the repository's ONE implementation of "does this comment author hold
# write access?", plus the bounded `gh` wrapper every such lookup runs under.
#
# SOURCED, never executed. It defines functions and no top-level behaviour.
#
# WHY IT IS A LIBRARY (#1806). Two scripts need this exact answer, for two different reasons:
#
#   - deliver-issue-refinements.sh (#1782) — WHICH COMMENTS carry design authority over an issue
#     body, so an implement phase does not build a superseded spec.
#   - deliver-trusted-comments.sh (#1806) — WHICH COMMENT TEXT an AI flow is allowed to read at
#     all, so a public repository's drive-by comment cannot steer an agent that holds credentials
#     on a self-hosted runner.
#
# They are the same trust boundary, resolved against the same endpoint, with the same three
# failure directions (definitive "no", could-not-ask, and hang). A second copy would be a second
# place for the 404-vs-403 split to drift, and that split is the one thing in here that a reading
# of the docs got wrong once already.
#
# ── The trust rule ────────────────────────────────────────────────────────────────────────────
#
# An author counts iff they hold ADMIN, WRITE or MAINTAIN permission on this repository — the same
# boundary `/approve-issue-for-pr-delivery`, `check-permissions` and the review triggers already
# use. Resolved per distinct author against
# `repos/{owner}/{repo}/collaborators/{login}/permission`.
#
# `authorAssociation` is deliberately NOT the signal, and the reason is a measurement rather than a
# preference: this repository's maintainer reports `authorAssociation: CONTRIBUTOR`, because GitHub
# reports "has had a PR merged" in preference to collaborator status. Trusting
# OWNER/MEMBER/COLLABORATOR would drop exactly the comments that matter while still admitting
# anyone whose PR has ever been merged.
#
# ── What the caller must provide before sourcing ──────────────────────────────────────────────
#
#   degrade <reason>   a function that reports an unreadable channel and exits. Each caller has its
#                      own marker (`REFINEMENT-READ-FAILED`, `COMMENT-READ-FAILED`), so the
#                      library calls the caller's rather than owning one.
#   SELF               the calling script's path, for stderr attribution.
#   REPO               owner/name.
#   TMP                a writable scratch directory.
#
# `REPO` and `TMP` may be set after sourcing, as long as they are set before
# `resolve_write_access_map` runs; `gh_bounded_setup` needs neither.

# ── Bounded `gh` ──────────────────────────────────────────────────────────────────────────────
#
# Bound every GitHub call so a stalled request cannot hang the whole job until its 60/120-minute
# timeout. The permission lookup is the likeliest culprit, but the same risk attaches to any `gh`
# call, so the deadline wraps all of them.
#
# `timeout` (coreutils — present on the Linux CI and self-hosted runners that actually run
# deliveries) or `gtimeout` (macOS with coreutils) enforces it where present; where neither is
# installed the portable `run_bounded` watchdog enforces the same deadline, so NO execution path
# runs `gh` unbounded. The deadline first sends SIGTERM and then, after a short grace, escalates to
# SIGKILL — which cannot be caught — so even a call that ignores or blocks SIGTERM is bounded at
# deadline + grace, never indefinitely. Any of these exits non-zero, so every caller treats it
# exactly like any other "could not ask" failure — fail-closed, never mistaken for a definitive
# answer. GH_DEADLINE_SECONDS / GH_KILL_GRACE_SECONDS are overridable so tests can force short
# values.
gh_bounded_setup() {
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
}

# Portable fallback deadline, used when neither `timeout` nor `gtimeout` is installed, so there is
# NO execution path on which a stalled `gh` call runs unbounded — not even on a host without
# coreutils. A watchdog subshell kills the call after the deadline, escalating SIGTERM to SIGKILL
# after the grace so a TERM-resistant call cannot evade it. Its stdout/stderr go to /dev/null so it
# can never hold the command-substitution pipe open: were it to, `out=$(gh …)` would block on the
# watchdog's own sleep instead of returning when the call does, defeating the bound.
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

# ── Permission resolution ─────────────────────────────────────────────────────────────────────

# Prints the author's permission and returns 0 when the answer is DEFINITIVE; returns 1 when the
# lookup could not be made at all.
#
# The distinction is the difference between two outcomes that must not be conflated. A 404 is a real
# answer — GitHub says this login is not a collaborator (it is also what a non-user login such as
# `github-actions` returns) — whereas a 401/403/429/5xx/network failure, or a deadline expiry from
# the bounded `gh` wrapper above, means the caller could not ask.
# Verified against this repository: a genuine non-collaborator returns 200 with `read`, so "no write
# access" normally arrives as a successful lookup and a failure really is a failure.
#
# Why it matters: `GET /repos/{owner}/{repo}/collaborators/{login}/permission` needs PUSH access, so
# a contributor with read-only access running this gets a failure for EVERY author. Without this
# split that reads as "nobody has write access", which for a refinements digest means "no
# refinements" and for a comment digest means "no comments" — a silent empty channel in both cases.
resolve_permission() {
  local login="$1" out
  if out=$(gh api "repos/$REPO/collaborators/$login/permission" --jq '.permission' 2>"$TMP/err"); then
    printf '%s' "$out"
    return 0
  fi
  if grep -qi 'HTTP 404' "$TMP/err"; then
    printf 'none'
    return 0
  fi
  printf '%s' "$(tr '\n' ' ' < "$TMP/err")"
  return 1
}

# Reads newline-delimited logins on stdin and WRITES a compact JSON object mapping each to
# true/false into the file named by $1. Sets WA_ATTEMPTED and WA_RESOLVED so the caller can decide
# what a wholly-unresolvable thread means for its own channel.
#
# The map goes to a FILE rather than to stdout precisely so the counters survive: a caller writing
# `map=$(resolve_write_access_map)` would run the whole loop in a command-substitution subshell,
# where WA_ATTEMPTED/WA_RESOLVED are set and then discarded — and the wholly-unresolvable check both
# callers make would silently read 0/0 and never fire. That is the same silent-empty class the
# callers' degrade markers exist to end, so the signature forecloses it.
#
# PARTIAL resolution deliberately does NOT degrade. If some authors resolve and some do not, the
# unresolved ones map to false — each named on stderr — and the caller continues. Degrading on any
# single lookup failure would let one deleted account or one renamed login block a whole delivery,
# which is worse than dropping a comment. Under-trusting costs a missed comment; over-trusting hands
# an agent's instructions to an unverified author.
resolve_write_access_map() {
  local out="$1" access='{}' login ok perm
  WA_ATTEMPTED=0
  WA_RESOLVED=0
  while IFS= read -r login; do
    [[ -n "$login" ]] || continue
    WA_ATTEMPTED=$((WA_ATTEMPTED + 1))
    ok=false
    if perm=$(resolve_permission "$login"); then
      WA_RESOLVED=$((WA_RESOLVED + 1))
      case "$perm" in
        admin | write | maintain) ok=true ;;
        *) echo "$SELF: @$login has no write access (permission='$perm') — their comments carry no authority" >&2 ;;
      esac
    else
      echo "$SELF: could not establish @$login's repository permission ($perm) — their comments are being dropped" >&2
    fi
    access=$(jq -c --arg l "$login" --argjson v "$ok" '. + {($l): $v}' <<< "$access") \
      || degrade "could not record write access for @$login"
  done
  printf '%s' "$access" > "$out" || degrade "could not write the write-access map to $out"
}

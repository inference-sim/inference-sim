#!/usr/bin/env bash
#
# deliver-seed-refs.sh — read the two references a delivery seed needs out of a sub-issue body.
#
# Usage: scripts/deliver-seed-refs.sh <body-file>
#
# deliver-implement.yml pushes the delivery branch BEFORE the agent runs (#1722); the agent then
# opens the draft PR as its own first action. Either way the WORKFLOW, not the agent, has to answer
# two questions from the issue body alone before anything is created:
#
#   1. Which branch does this PR target? An archon sub-issue targets its feature branch, and
#      basing the delivery on the default branch instead would put every unrelated commit
#      between the two into the PR's diff.
#   2. Does the issue declare an `archon-plan:`? The seeded PR body has to carry it verbatim,
#      because scripts/archon-plan-resolve.sh reads the plan FROM THE PR BODY and a PR to a
#      feature branch links no closing issue. Seeding it means the declaration survives an
#      agent that never finishes writing the real body.
#
# It lives in a file rather than inline in the workflow so that it can be tested
# (scripts/deliver_seed_refs_test.go). Getting (1) wrong produces a PR whose diff is mostly
# other people's commits; getting (2) wrong silently skips the dist ratchet. Neither is
# visible without reading a real delivery, which is not a property that survives the next edit.
#
# Prints exactly five lines on stdout, the value after `=` possibly empty:
#   target_branch=<ref or empty>
#   archon_plan=<the declaration line verbatim, or empty>
#   heading_seen=true|false   — a `Target branch` heading was present in the body
#   plan_seen=true|false      — an `archon-plan:` declaration was present in the body
#   unclosed_fence=true|false — the body ended inside a fence, so content was discarded
#
# The two `*_seen` flags exist so the caller can tell "nothing was declared" from "something was
# declared that I could not read", and warn on the second (R1: never silent). Without them a heading
# the pattern does not quite match — `## Target branch (base)`, say — or a declaration hidden by an
# unclosed fence reads identically to a standalone issue, and the delivery quietly goes to the
# default branch with the dist ratchet quietly off.
#
# Both are ADVISORY. This script does not decide whether the branch exists — the caller
# checks that against the remote and falls back to the default branch — because a ref that
# resolves here but not on the remote must not fail the delivery.
#
# Exits 0 whenever the body could be read; exit 2 only on a usage error. An absent section or
# declaration is an empty value, never an error: most issues this loop delivers are standalone
# and legitimately have neither.
#
# `set -e` is deliberately off, matching archon-plan-resolve.sh: a missing section is expected
# control flow and `grep` exits 1 on no match.

set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <body-file>" >&2
  exit 2
fi

BODY_FILE="$1"

if [[ ! -r "$BODY_FILE" ]]; then
  echo "$0: cannot read $BODY_FILE" >&2
  exit 2
fi

# CRLF is stripped once, up front: a body edited through the GitHub web UI can carry it, and a
# trailing \r would otherwise ride along inside the captured ref and make every remote lookup
# miss.
# EVERY text tool this script depends on is proved usable up front.
#
# This closes a class rather than an instance. `set -e` is deliberately off (a missing section is
# expected control flow and `grep` exits 1 on no match), so a tool that fails for a REAL reason —
# missing, broken, OOM-killed, or unwritable temp space — otherwise turns into an empty result at
# exit 0. Two instances of that were reported on #1723: a failing `awk` reported a valid body as
# empty, and a failing `sed` reported a body that DID declare `archon-plan:` as `plan_seen=false`,
# silently disabling the dist ratchet. Guarding each pipeline individually cannot work, because a
# legitimate no-match is also a non-zero exit; proving the tools work once can.
for tool in sed grep awk tr head tail; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "$0: $tool is not on PATH; refusing to parse an issue body with a missing tool" >&2
    exit 2
  fi
done
if ! printf 'x\n' | sed -n 'p' >/dev/null 2>&1 \
   || ! printf 'x\n' | grep -q 'x' \
   || ! printf 'x\n' | awk '{ exit 0 }' \
   || ! printf 'x\n' | tr -d 'x' >/dev/null 2>&1 \
   || ! printf 'x\n' | head -n1 >/dev/null 2>&1 \
   || ! printf 'x\n' | tail -n +1 >/dev/null 2>&1; then
  echo "$0: one of sed/grep/awk/tr/head/tail is present but not usable; refusing to report an empty body as a valid one" >&2
  exit 2
fi

BODY=$(tr -d '\r' < "$BODY_FILE")

# The fence state is returned on STDOUT as a first line, not through a temp file or awk's stderr.
# The previous plumbing did both and was unsafe in a way that mattered: `FENCE_STATE=$(mktemp)` was
# unchecked and there is no `set -e`, so on a full or unwritable TMPDIR the redirect `2>""` failed,
# `BODY` came back EMPTY, and a perfectly valid issue body produced
# `target_branch= archon_plan= heading_seen=false plan_seen=false unclosed_fence=false` with exit 0 —
# every guard silent, the delivery based on the default branch with no plan line. The trigger is the
# runner's known failure mode (storage exhaustion), so this was not theoretical. Reported on #1723.
#
# Using stderr as a data channel had a second symptom from the same root: a REAL awk diagnostic would
# be captured, fail to match the sentinel, and be discarded silently.
#
# So: awk buffers the surviving lines, prints `closed`/`unclosed` first, then the body. No temp file
# to fail, leak, or need a trap; awk's stderr stays a diagnostic channel.
FENCE_OUT=$(printf '%s\n' "$BODY" | awk '
  function runlen(s, ch,   n) { n = 0; while (substr(s, n + 1, 1) == ch) n++; return n }
  {
    match($0, /^ */); ind = RLENGTH
    rest = substr($0, ind + 1)
    ch = substr(rest, 1, 1)
    if (infence) {
      if (ind <= 3 && ch == fencechar) {
        n = runlen(rest, fencechar)
        if (n >= fencelen) {
          tail = substr(rest, n + 1)
          gsub(/[ \t]/, "", tail)
          if (tail == "") { infence = 0 }
        }
      }
      next
    }
    if (ind <= 3 && (ch == "`" || ch == "~")) {
      n = runlen(rest, ch)
      if (n >= 3) { fencechar = ch; fencelen = n; infence = 1; next }
    }
    kept[++k] = $0
  }
  END {
    print (infence ? "unclosed" : "closed")
    for (i = 1; i <= k; i++) print kept[i]
  }
')

# The marker is validated rather than merely read. Without this, ANY failure of the awk stage —
# a missing or broken awk, an OOM kill, a syntax error introduced by a future edit — yields an empty
# FENCE_OUT, which reads as "closed" with an empty body: all five values come back empty at exit 0
# and every downstream guard stays silent. That is the same silent-empty class as the temp file this
# replaced (#1723 review), reached by a different route, so it is closed explicitly here.
FENCE_MARKER="${FENCE_OUT%%$'\n'*}"
if [[ "$FENCE_MARKER" != "closed" && "$FENCE_MARKER" != "unclosed" ]]; then
  echo "$0: fence scan did not run (awk produced no marker); refusing to report an empty body as a valid one" >&2
  exit 2
fi
UNCLOSED_FENCE=false
if [[ "$FENCE_MARKER" == "unclosed" ]]; then
  UNCLOSED_FENCE=true
fi
# Everything after the first line is the stripped body. A body that reduces to nothing leaves
# FENCE_OUT as the marker alone, with no newline, so the strip must be conditional.
if [[ "$FENCE_OUT" == *$'\n'* ]]; then
  BODY="${FENCE_OUT#*$'\n'}"
else
  BODY=""
fi

# The `## Target branch` section's prose is not fixed. Both of these are documented in
# docs/contributing/templates/archon-issue-examples.md:
#
#     `feature/<name>` (PR against the feature branch, NOT main)
#     `feature/<name>` → `main`
#
# so the section can hold one ref or two, and the SECOND one in the arrow form is the base of
# the eventual feature→main PR, not of this delivery. Taking the FIRST backticked ref is
# therefore correct for both shapes, and is why this reads a ref rather than trying to parse
# the sentence.
#
# The sed range ends at the next `##` heading so a backticked ref in a later section cannot be
# picked up. In sed a range's end pattern is only looked for on lines AFTER the start, so the
# `## Target branch` line itself does not close the range; a section that runs to end-of-body
# is read to EOF, which is what the templates' final-section placement needs.
#
# `head -n1` after `grep -o`, NOT `grep -m1 -o`: `-m1` stops after the first matching LINE, and
# `-o` still prints EVERY match on that line — so the arrow form emitted both `feature/<name>`
# and `main`, and the two-line result was then rejected by the whitespace guard below, silently
# falling back to the default branch. Pinned by the arrow-form case in the test.
#
# The heading is matched case-INSENSITIVELY, written as bracket classes rather than with sed's `I`
# flag because that flag is a GNU extension and this script is exercised on macOS (BSD sed) too. It
# has to agree with the case-insensitive `grep -i` that computes heading_seen below: when detection
# accepted a spelling extraction rejected, `## target branch` warned and fell back to the default
# branch even though the author had written a perfectly usable section.
# ONLY THE FIRST NON-BLANK LINE of the section is searched for a ref, which is both simpler than
# scanning the section and the fix for a real silent-wrong-base bug (#1723 review, F4): fenced lines
# are removed, so a fence spanning a `## ` boundary deletes that boundary, the section range then
# runs on into what the author sees as a LATER section, and a ref from there became the delivery's
# base with no warning. Reproduced: a `## Notes` heading inside a fence made `feature/WRONG` win.
#
# Every documented shape puts the ref on the first line of the section
# (docs/contributing/templates/archon-issue-examples.md):
#
#     `feature/<name>` (PR against the feature branch, NOT main)
#     `feature/<name>` → `main`
#
# so nothing legitimate is lost, and a section whose first line is prose ("Not stated.") now yields
# no ref and warns instead of silently guessing.
#
# `head -n1` after `grep -o`, NOT `grep -m1 -o`: `-m1` stops after the first matching LINE while `-o`
# still prints EVERY match on it, so the arrow form emitted two refs and the pair was then rejected
# by the whitespace guard — silently falling back to the default branch.
#
# The heading is matched case-INSENSITIVELY via bracket classes rather than sed's `I` flag, which is
# a GNU extension; this script is exercised on macOS (BSD sed) too. It must agree with the
# case-insensitive `grep -i` computing heading_seen below.
if ! SECTION=$(printf '%s\n' "$BODY" \
  | sed -n '/^ \{0,3\}#\{1,6\}[[:space:]]*[Tt][Aa][Rr][Gg][Ee][Tt][[:space:]][Bb][Rr][Aa][Nn][Cc][Hh][[:space:]]*$/,/^ \{0,3\}#\{1,6\}[[:space:]]/p'); then
  echo "$0: sed failed while locating the Target branch section" >&2
  exit 2
fi
# `|| true` from here on covers a legitimate NO MATCH only — grep exits 1 when the section has no
# backticked ref, which is an answer rather than a failure. The tools themselves were proved above.
TARGET_BRANCH=$(printf '%s\n' "$SECTION" \
  | tail -n +2 \
  | grep -m1 '[^[:space:]]' \
  | grep -oE '`[^`]+`' \
  | head -n1 \
  | tr -d '`') || true

TARGET_BRANCH="${TARGET_BRANCH#"${TARGET_BRANCH%%[![:space:]]*}"}"
TARGET_BRANCH="${TARGET_BRANCH%"${TARGET_BRANCH##*[![:space:]]}"}"

# The captured ref comes from an ISSUE BODY, which anyone can write, and the caller feeds it to
# `git ls-remote` and `gh pr create --base`. deliver-verify.yml already records the general rule
# for this repository — "a ref name is attacker-influenceable and inlining it is a script-injection
# vector" — so it is validated by SHAPE here rather than trusted downstream.
#
# Allowed: letters, digits, dot, underscore, slash, dash. That covers every branch name this
# repository uses (`main`, `feature/<name>`) and excludes the characters that give a ref meaning to
# a shell or to git's revision parser. Anything else is discarded, and the caller falls back to the
# default branch — a wrong-looking ref must never fail a delivery, and must never be executed.
#
# A LEADING DASH is refused specifically: `git ls-remote --heads origin -foo` and
# `gh pr create --base -foo` can read it as an option rather than a ref.
if [[ ! "$TARGET_BRANCH" =~ ^[A-Za-z0-9._/-]+$ ]] || [[ "$TARGET_BRANCH" == -* ]]; then
  TARGET_BRANCH=""
fi

# `..` and a trailing `.lock` are refused by `git check-ref-format` too, and `..` additionally has
# revision-range meaning. Refusing them keeps the caller from resolving something other than a
# branch.
if [[ "$TARGET_BRANCH" == *..* || "$TARGET_BRANCH" == *.lock || "$TARGET_BRANCH" == /* || "$TARGET_BRANCH" == */ ]]; then
  TARGET_BRANCH=""
fi

# The SAME pattern archon-plan-resolve.sh matches, deliberately: if the two disagree, a plan
# seeded here would not be the plan resolved there. Anchored so a sentence mentioning
# `archon-plan:` in passing is not a declaration, and requiring `\S` after the colon so a bare
# `archon-plan:` with no path is not treated as one.
if ! BODY_NO_INDENT=$(printf '%s\n' "$BODY" | sed -E '/^ {4,}/d'); then
  echo "$0: sed failed while removing indented lines" >&2
  exit 2
fi
ARCHON_PLAN=$(printf '%s\n' "$BODY_NO_INDENT" \
  | grep -m1 -E '^[^A-Za-z0-9]*archon-plan:[[:space:]]*\S') || true
ARCHON_PLAN=${ARCHON_PLAN%%$'\n'*}

# Deliberately LOOSER than the section pattern above: it answers "did the author try to declare a
# target branch", so it must still match the headings the strict pattern rejects — that mismatch is
# exactly what the caller needs to warn about.
HEADING_SEEN=false
if printf '%s\n' "$BODY" | grep -qiE '^ {0,3}#{1,6}[[:space:]]*Target branch'; then
  HEADING_SEEN=true
fi

# The plan half of the same idea. Without it, an extractor that misses a declaration leaves the PR
# body with no `archon-plan:` line, deliver-verify.yml's grep finds nothing, and the gate reads
# `absent` — which PASSES — rather than `unverified`, which blocks. So a silent miss here would
# silently switch the dist ratchet off, the exact failure deliver-verify.yml warns about.
PLAN_SEEN=false
if printf '%s\n' "$BODY_NO_INDENT" | grep -qE '^[^A-Za-z0-9]*archon-plan:[[:space:]]*\S'; then
  PLAN_SEEN=true
fi

echo "target_branch=$TARGET_BRANCH"
echo "archon_plan=$ARCHON_PLAN"
echo "heading_seen=$HEADING_SEEN"
echo "plan_seen=$PLAN_SEEN"
echo "unclosed_fence=$UNCLOSED_FENCE"

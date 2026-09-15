#!/usr/bin/env bash
#
# deliver-seed-refs.sh — read the two references a delivery seed needs out of a sub-issue body.
#
# Usage: scripts/deliver-seed-refs.sh <body-file>
#
# deliver-implement.yml opens the delivery branch and its draft PR BEFORE the agent runs
# (#1722), which means the workflow — not the agent — has to answer two questions from the
# issue body alone:
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
# Prints exactly two lines on stdout, either side of `=` possibly empty:
#   target_branch=<ref or empty>
#   archon_plan=<the declaration line verbatim, or empty>
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
BODY=$(tr -d '\r' < "$BODY_FILE")

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
TARGET_BRANCH=$(printf '%s\n' "$BODY" \
  | sed -n '/^[[:space:]]*#\{1,6\}[[:space:]]*Target branch[[:space:]]*$/,/^[[:space:]]*#\{1,6\}[[:space:]]/p' \
  | grep -oE '`[^`]+`' \
  | head -n1 \
  | tr -d '`') || true

# Trailing/leading whitespace inside the backticks would survive `tr -d`, and a ref is never
# whitespace, so trim rather than carry it into a remote lookup.
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
ARCHON_PLAN=$(printf '%s\n' "$BODY" \
  | grep -m1 -E '^[^A-Za-z0-9]*archon-plan:[[:space:]]*\S') || true

# Only ever a single line, so a body carrying an embedded newline cannot inject a second
# key=value pair into a caller reading this output line-by-line.
ARCHON_PLAN=${ARCHON_PLAN%%$'\n'*}

echo "target_branch=$TARGET_BRANCH"
echo "archon_plan=$ARCHON_PLAN"

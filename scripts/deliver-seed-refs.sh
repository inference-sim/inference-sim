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
# Prints exactly four lines on stdout, the value after `=` possibly empty:
#   target_branch=<ref or empty>
#   archon_plan=<the declaration line verbatim, or empty>
#   heading_seen=true|false   — a `Target branch` heading was present in the body
#   plan_seen=true|false      — an `archon-plan:` declaration was present in the body
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
BODY=$(tr -d '\r' < "$BODY_FILE")

FENCE_STATE=$(mktemp)

# FENCED CODE BLOCKS ARE STRIPPED FIRST, and this is not hygiene — it is a correctness fix.
# docs/contributing/templates/archon-issue-examples.md shows the whole sub-issue template inside a
# fence, `## Target branch` and a `feature/...` ref included. A contributor who pastes that example
# into an issue body would otherwise have the FENCED ref win over their real one. A fictional
# placeholder happens to fail the remote check and fall back, but a fenced REAL branch name would
# silently become the delivery's base.
#
# The first version of this toggled on ANY line starting with ``` or ~~~, which is a line-PARITY
# rule, and an ODD number of marker lines therefore discarded the whole rest of the body. A single
# indented ``` — which GitHub renders as literal text inside an indented code block, so the author
# sees nothing wrong — was enough to make a real `## Target branch` section and a real
# `archon-plan:` line both vanish. Reported on #1723.
#
# So the delimiters are tracked properly, per CommonMark: remember the opening character and its
# run length, close only on a run of the SAME character at least that long with nothing but spaces
# after it, and ignore any marker indented 4+ spaces (that is an indented code block, not a fence).
# Run lengths are counted in a loop rather than with an interval regex (`{3,}`), because interval
# expressions are not portable to the BSD awk this suite also runs under.
#
# NOTE the residual: a genuinely UNCLOSED fence still swallows everything after it, because that is
# what CommonMark says it means, and no parser can fix that. It is reported instead — see
# `unclosed_fence` below, which is why the caller can refuse rather than seed a guess.
#
# The two `*_seen` signals are computed on the STRIPPED body, i.e. on what the author actually
# declared in visible content. Computing them on the raw body conflated "declared" with "quoted an
# example", and the caller turned that into a hard error refusing legitimate deliveries (#1723
# review, F2).
BODY=$(printf '%s\n' "$BODY" | awk '
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
    print
  }
  END { if (infence) print "@@UNCLOSED_FENCE@@" > "/dev/stderr" }
' 2>"$FENCE_STATE")

# An UNCLOSED fence swallows everything after it — correct per CommonMark, and the one case a
# parser cannot rescue. Recorded here so the caller can distinguish it from a deliberately-fenced
# example: content lost to an unclosed fence deserves a loud signal, content the author chose to
# put inside a closed fence deserves silence.
UNCLOSED_FENCE=false
if [[ -s "$FENCE_STATE" ]] && grep -q '@@UNCLOSED_FENCE@@' "$FENCE_STATE"; then
  UNCLOSED_FENCE=true
fi
rm -f "$FENCE_STATE"

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
TARGET_BRANCH=$(printf '%s\n' "$BODY" \
  | sed -n '/^[[:space:]]*#\{1,6\}[[:space:]]*[Tt][Aa][Rr][Gg][Ee][Tt][[:space:]][Bb][Rr][Aa][Nn][Cc][Hh][[:space:]]*$/,/^[[:space:]]*#\{1,6\}[[:space:]]/p' \
  | sed '1d' \
  | grep -m1 '[^[:space:]]' \
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

# Deliberately LOOSER than the section pattern above: it answers "did the author try to declare a
# target branch", so it must still match the headings the strict pattern rejects — that mismatch is
# exactly what the caller needs to warn about.
HEADING_SEEN=false
if printf '%s\n' "$BODY" | grep -qiE '^[[:space:]]*#{1,6}[[:space:]]*Target branch'; then
  HEADING_SEEN=true
fi

# The plan half of the same idea. Without it, an extractor that misses a declaration leaves the PR
# body with no `archon-plan:` line, deliver-verify.yml's grep finds nothing, and the gate reads
# `absent` — which PASSES — rather than `unverified`, which blocks. So a silent miss here would
# silently switch the dist ratchet off, the exact failure deliver-verify.yml warns about.
PLAN_SEEN=false
if printf '%s\n' "$BODY" | grep -qE '^[^A-Za-z0-9]*archon-plan:[[:space:]]*\S'; then
  PLAN_SEEN=true
fi

echo "target_branch=$TARGET_BRANCH"
echo "archon_plan=$ARCHON_PLAN"
echo "heading_seen=$HEADING_SEEN"
echo "plan_seen=$PLAN_SEEN"
echo "unclosed_fence=$UNCLOSED_FENCE"

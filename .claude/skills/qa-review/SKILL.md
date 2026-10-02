---
name: qa-review
description: Cross-vendor two-agent review (Gemini questioner + OpenAI Sol answerer) of a PR or a local branch, run on demand. Drives the committed scripts/qa-review/*.py directly — the same mechanic the delivery gate (deliver-verify.yml) runs — so it never drifts from the gate. Read-only on code; prints a PASS/BLOCK verdict and, with --post, comments on the PR. Invoke as /qa-review <PR> or /qa-review --branch.
---

# qa-review — cross-vendor two-agent review, on demand

Runs the RFC #1603 cross-vendor review on demand. A **questioner** on one model
family (Gemini) generates probing questions; an isolated **answerer** on a
different family (OpenAI Sol) investigates the real code through a read-only tool
loop and answers each with `file:line` evidence; a **renderer** turns the pair
into a Markdown report with a `PASS`/`BLOCK` verdict. The decorrelated second
opinion — a different vendor than the Opus that writes code and the Sonnet that
reviews it — is the point: it catches what a single-vendor review misses.

**This skill is a front-door to the gate's own tooling, not a copy of it.** It
shells out to the committed `scripts/qa-review/*.py` (resolved from the repo
root), which are the exact scripts `.github/workflows/deliver-verify.yml` runs.
It therefore inherits every hardening those scripts carry — the retrying HTTP
client (`_http.py`, #1833), the answerer's `gh_issue`/`pr_diff` tools (#1792),
write-access comment filtering (#1806), the adjudicator's `same_login`/structural
report selection (#1716/#1834) — and cannot fall behind them. The canonical
description of the pieces, flags, and environment is
[`scripts/qa-review/README.md`](../../../scripts/qa-review/README.md).

**Two things this skill is NOT:**

- It is **not the gate.** It prints a verdict for a human to read; it does not
  emit the `QA-VERDICT: PASS|BLOCK|MISSING` marker or set delivery state —
  deriving that marker from the answerer output is `deliver-verify.yml`'s job
  (#1715/#1716). Running this skill does not change any gate result.
- It is **not `blis-pr-review`.** That skill is a single-vendor (Anthropic)
  self-review across BLIS-specific perspectives; this one is the decorrelated
  cross-vendor second opinion. They are complementary.

It is **read-only on code**: it never edits, commits, or pushes. By default it
also does not post anything — pass `--post` to comment on the PR.

## Arguments

- `/qa-review <PR-number-or-URL> [--post] [--adjudicate]` — **PR mode.** Review an
  open or merged PR (mirrors `deliver-verify.yml` round 0). `--post` posts the
  report as a PR comment; omit it to preview locally. `--adjudicate` runs the
  **adjudication** path (re-check the prior report's blocking findings against the
  author's responses) instead of a fresh review — the re-verify behavior.
- `/qa-review --branch [--base <ref>] [--issue <N>] [--post]` — **local-branch
  mode.** Review the current branch's `HEAD` against `<ref>` (default `main`) with
  no PR. `--issue <N>` feeds that issue's acceptance criteria so the
  implementation-completeness question is answerable. See the limitations below.

If neither a PR nor `--branch` is given, ask which to review before proceeding.

## Config (environment)

Read by the scripts; the skill only passes them through.

- `OPENAI_BASE_URL` + `OPENAI_API_KEY` (or `LITELLM_KEY`) — an OpenAI-compatible
  LLM endpoint (e.g. a LiteLLM proxy). **Required** by every script.
- `QA_QUESTIONER_MODEL` (default `gcp/gemini-3.6-flash`) — a cheap, non-Anthropic
  model, named **as your endpoint routes it**.
- `QA_ANSWERER_MODEL` / `QA_ADJUDICATOR_MODEL` (default `azure/gpt-5.6-sol`) —
  must support function/tool calling; deliberately a different vendor from the
  Anthropic models used for code-writing and `blis-pr-review`.
- `QA_REPO` — `owner/repo` for the `gh` calls. Defaults below to the repo the
  current clone points at, so forks work with no configuration.

> The default model ids are examples from one proxy — set the `*_MODEL` vars to
> ids your endpoint actually serves. If a model is not routable the script errors;
> do **not** fall back to an Anthropic model, which would defeat the cross-vendor
> design — report the error and stop.

## Setup (both modes)

Resolve the repo root and scripts from git, so the skill is portable across
clones, forks, and machines (no absolute paths):

```bash
ROOT="$(git rev-parse --show-toplevel)"
QA="$ROOT/scripts/qa-review"
REPO="${QA_REPO:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
export QA_REPO="$REPO"   # the answerer/adjudicator gh_issue/pr_diff tools read this
AMODEL="${QA_ANSWERER_MODEL:-azure/gpt-5.6-sol}"
```

---

## Mode A — PR review (mirrors `deliver-verify.yml` round 0)

### 1. Resolve the PR + gather intent

```bash
PR=<arg>
gh pr view "$PR" --repo "$REPO" \
  --json number,title,body,headRefOid,baseRefName,state,url > /tmp/qa-$PR-pr.json
title="$(jq -r '.title // ""' /tmp/qa-$PR-pr.json)"
jq -r '.body // ""' /tmp/qa-$PR-pr.json > /tmp/qa-$PR-body.txt
gh pr diff "$PR" --repo "$REPO" > /tmp/qa-$PR-diff.txt
```

Parse the closing issue from the PR body (`Closes/Fixes/Resolves #N`); set
`ISSUE` to the first such number (the answerer takes a single `--issue`). If the
PR closes no issue, leave `ISSUE` empty.

### 2. Materialize the PR head into a throwaway worktree

So the answerer investigates the real PR-head code without disturbing your
checkout. Fetch by URL (remote-name-agnostic):

```bash
git -C "$ROOT" fetch "https://github.com/$REPO.git" "pull/$PR/head"
WT="$(mktemp -d /tmp/qa-review-$PR-XXXX)"
git -C "$ROOT" worktree add --detach "$WT" FETCH_HEAD
```

### 3. Questioner (cross-vendor)

```bash
python3 "$QA/questioner.py" \
  --title "$title" \
  --body-file /tmp/qa-$PR-body.txt \
  --diff-file /tmp/qa-$PR-diff.txt \
  > /tmp/qa-$PR-questions.json
```

Emits `{"model":…, "questions":[{id,topic,question}…]}` — fixed policy questions
(`F1..`) plus generated ones (`G1..`). If it errors (model not routable, endpoint
unreachable), STOP and report the error verbatim.

### 4. Answerer (OpenAI Sol via a read-only tool loop)

```bash
python3 "$QA/answerer.py" \
  --worktree "$WT" --no-exec \
  --questions-file /tmp/qa-$PR-questions.json \
  --pr "$PR" --issue "$ISSUE" \
  > /tmp/qa-$PR-answers.json
```

`--no-exec` drops the code-executing `go` tool, leaving read-only
`read_file`/`grep`/`list_dir` and the `gh_issue`/`pr_diff` tools — the same stance
the gate takes on PR code. (Reviewing a branch you trust? Drop `--no-exec` to let
the answerer run `go build`/`go test`.) The `--pr`/`--issue` numbers let its
`gh_issue`/`pr_diff` tools verify issue-completeness for `F1` (#1792). Per answer:
`status ∈ {CONFIDENT, CANNOT_ANSWER, FLAW_FOUND}` with `answer`, `evidence`, and an
optional non-blocking `note`. If it errors, STOP and report it verbatim.

### 5. Render the report + verdict

Do NOT hand-summarize — run the renderer so every question, result, and answer is
shown. `--qmodel` is deliberately omitted: the renderer reads the questioner model
from the questions document, so the banner cannot drift. Add the `--post-to-pr`
line **only if the user passed `--post`**:

```bash
python3 "$QA/render_report.py" \
  --questions /tmp/qa-$PR-questions.json \
  --answers /tmp/qa-$PR-answers.json \
  --pr "$PR" --title "$title" --amodel "$AMODEL" \
  --out /tmp/qa-review-$PR-report.md \
  --post-to-pr --repo "$REPO"      # include ONLY with --post
```

Verdict rule: **`BLOCK` iff any answer is `FLAW_FOUND` or `CANNOT_ANSWER`**, else
`PASS`. The output is the one-line verdict header, the full untruncated
`ID | Topic | Result | Question | Answer` table, an **Items to fix** section
(blocking findings + evidence), and an **Important to consider** section (notes).

**CRITICAL — the renderer's stdout IS your entire final reply.** Reproduce it
verbatim (the whole table, plus the `[posted to PR …]` line if `--post`) and add
NOTHING around it: no preamble, no hand-written summary, no condensed bullets, no
"calibration" notes. The renderer posts the comment itself via `--post-to-pr` — do
not post separately or by hand.

### 6. Cleanup

```bash
git -C "$ROOT" worktree remove --force "$WT"
rm -f /tmp/qa-$PR-pr.json /tmp/qa-$PR-body.txt /tmp/qa-$PR-diff.txt \
      /tmp/qa-$PR-questions.json /tmp/qa-$PR-answers.json
```

Keep `/tmp/qa-review-$PR-report.md` (re-viewable; stable per PR).

## Adjudication (`--adjudicate`)

Use AFTER a PR already has a qa-review **BLOCK** report comment and the author has
replied (a fix, an out-of-scope justification, or a filed follow-up issue). This
re-checks the prior blocking findings against those responses — skeptically,
defaulting to STILL-OPEN. Mirrors the gate's re-verify round.

```bash
PR=<arg>
git -C "$ROOT" fetch "https://github.com/$REPO.git" "pull/$PR/head"
WT="$(mktemp -d /tmp/qa-review-$PR-XXXX)"
git -C "$ROOT" worktree add --detach "$WT" FETCH_HEAD
python3 "$QA/adjudicator.py" \
  --worktree "$WT" --pr "$PR" --no-exec \
  --post-to-pr                       # include ONLY with --post
git -C "$ROOT" worktree remove --force "$WT"
```

Per prior finding it returns `RESOLVED` / `WAIVED_JUSTIFICATION` /
`WAIVED_DEFERRED` / `STILL_OPEN`; the aggregate is **`BLOCK` iff any finding is
`STILL_OPEN` or un-adjudicated**, emitted on stderr as
`[adjudication verdict: PASS|BLOCK]`. Print its rendered report verbatim (same rule
as Step 5).

> **Public-repo safety.** The adjudicator reads prior findings from PR comments.
> On a public repo, set `--report-author <login>` (or `QA_REPORT_AUTHOR`) to the
> login that posted the report, so a stranger cannot supply a report-shaped comment
> that clears every finding. The gate pins this to `github-actions[bot]`; for an
> interactive run it is whoever posted the report (e.g. you, via `--post`).

---

## Mode B — local branch, no PR (`--branch`)

Pre-PR review of the current branch. Drives the **same scripts** with git-computed
inputs. Commit your work first — this reviews `HEAD` vs `<base>`, not uncommitted
changes.

```bash
BASE="${BASE:-main}"            # --base <ref>
BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
MB="$(git -C "$ROOT" merge-base "$BASE" HEAD)"
git -C "$ROOT" diff "$MB"..HEAD > /tmp/qa-branch-diff.txt

# Optional issue context (recommended — without it F1 is unanswerable, so it BLOCKs):
if [ -n "$ISSUE" ]; then
  gh issue view "$ISSUE" --repo "$REPO" --json title,body \
    -q '"# \(.title)\n\n\(.body)"' > /tmp/qa-branch-body.txt
  TITLE="$(gh issue view "$ISSUE" --repo "$REPO" --json title -q .title)"
else
  : > /tmp/qa-branch-body.txt
  TITLE="$BRANCH"
fi

# A clean detached worktree at HEAD, so the diff and the code the answerer sees agree:
WT="$(mktemp -d /tmp/qa-branch-XXXX)"
git -C "$ROOT" worktree add --detach "$WT" HEAD

python3 "$QA/questioner.py" \
  --title "$TITLE" --body-file /tmp/qa-branch-body.txt \
  --diff-file /tmp/qa-branch-diff.txt > /tmp/qa-branch-questions.json

python3 "$QA/answerer.py" \
  --worktree "$WT" --no-exec \
  --questions-file /tmp/qa-branch-questions.json \
  --issue "$ISSUE" \
  > /tmp/qa-branch-answers.json      # no --pr: there is no PR to diff

python3 "$QA/render_report.py" \
  --questions /tmp/qa-branch-questions.json \
  --answers /tmp/qa-branch-answers.json \
  --pr "$BRANCH" --title "$TITLE" --amodel "$AMODEL" \
  --out /tmp/qa-review-branch-report.md

git -C "$ROOT" worktree remove --force "$WT"
rm -f /tmp/qa-branch-diff.txt /tmp/qa-branch-body.txt \
      /tmp/qa-branch-questions.json /tmp/qa-branch-answers.json
```

Print the rendered report verbatim as the final reply (same rule as Step 5).

**Local-branch limitations (by design — the scripts are PR-shaped):**

- No `pr_diff` tool for the answerer (there is no PR). The questioner still sees
  the full diff, and the answerer investigates the checked-out `HEAD`.
- No adjudication (`adjudicator.py` requires a PR and its comment thread).
- The report header shows the branch name where a PR number would be — cosmetic.
- Without `--issue`, the fixed completeness question `F1` is a blocking
  `CANNOT_ANSWER`; pass `--issue <N>` for a meaningful verdict.
- `--post` is accepted but has no PR to post to; it is a no-op in this mode.

## Notes

- **Porting / BLIS-specific tuning.** The fixed questions and topics
  (`questioner.py`) and the answerer/adjudicator system prompts are tuned for BLIS
  (its `INV-*` invariants, `R*` rules, run/replay/observe parity). This is why the
  skill lives in this repo rather than being generic.
- Re-running posts another comment (no dedup) — preview without `--post` first.
- Keep the whole report; the model-free surface of the scripts is unit-tested in
  `scripts/qa_review_test.go`, so this skill stays in step with them under
  `go test ./scripts/...`.

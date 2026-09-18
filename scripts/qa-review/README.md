# scripts/qa-review/

Cross-vendor two-agent PR review, vendored from the `/qa-review` prototype
(RFC #1603). A **questioner** on one model family generates probing questions;
an isolated **answerer** on a different family investigates the actual PR-head
code to answer them; a **renderer** turns the pair into a Markdown report and a
PASS/BLOCK verdict; an **adjudicator** re-checks prior blocking findings against
the author's defence on a re-verify round.

The decorrelated second opinion (a different vendor than the Opus code-writer
and the Sonnet reviewer) is the point — it catches what a single-vendor review
misses.

These scripts are **stdlib-only Python 3** (`urllib`, `json`, `re`, `argparse`,
`subprocess`) — no third-party dependency, no `requirements.txt`. They talk to
the LiteLLM proxy over its OpenAI-compatible `/chat/completions` surface.

> **Status.** #1714 vendored the tooling and tested its deterministic surface.
> #1715 made the verdict a **blocking gate signal**: `scripts/deliver-gate.sh`
> now requires `QA_VERDICT ∈ {PASS, BLOCK, MISSING}` as a seventh fail-closed
> input, and `deliver-verify.yml` runs the questioner + answerer against an
> ephemeral read-only PR-head worktree with `--no-exec`.
>
> Both halves are now live in this PR: the gate change and the `deliver-verify.yml`
> wiring — the `Run qa-review` and `Read the QA verdict marker` steps that produce and
> read `QA_VERDICT`. Because the delivery agent's GitHub App token has no `workflows`
> permission, the workflow half was applied by a `workflows`-scoped push rather than by
> the agent; `scripts/deliver_qa_verdict_test.go` holds the contract over the live
> workflow.
>
> #1716 completes the epic (#1717) by making the re-verify **adjudicate-only**: round 0
> keeps the full questioner+answerer pass, every later round runs only `adjudicator.py`
> over the findings that pass already raised. Both halves are live — the report-comment
> selection (below) and the workflow wiring (the `Run qa-review adjudication` step gated on
> the round counter). As with the #1715 half, the workflow change was applied by a
> `workflows`-scoped push (the delivery agent's token lacks the permission);
> `scripts/deliver_qa_adjudicate_test.go` holds the contract over the live workflow.
>
> #1792 closes the round-0 completeness gap that the epic surfaced: the answerer had none of the
> tools needed to see the closing issue or the diff, so fixed policy question F1 ("does this fully
> implement the issue?") was always a blocking `CANNOT_ANSWER` and the loop's qa-review could
> never `PASS` from round 0. The answerer now carries the **same read-only `gh_issue`/`pr_diff`
> tools the adjudicator already has** (kept under `--no-exec`), and `deliver-verify.yml`'s
> `Run qa-review` step hands it the PR and closing-issue **numbers** so those tools have something
> to query. `scripts/qa_review_test.go` pins the tool inventory (`TestNoExecSeam`) and that the
> numbers reach the prompt; `scripts/deliver_qa_verdict_test.go` pins the workflow wiring.

## The pieces

| Script | Role | stdout |
|---|---|---|
| `questioner.py` | probing-question generator (cross-vendor) | `{"model", "questions":[{id,topic,question}]}` |
| `answerer.py` | agentic, read-only answerer over the worktree | `[{id,status,answer,evidence,note}]` |
| `render_report.py` | report renderer (+ optional PR posting) | the Markdown report |
| `adjudicator.py` | author-defence re-check (used by #1716) | the adjudication report |

### questioner.py

Emits three **fixed** policy questions (`F1..F3`) verbatim — issue-implementation
completeness, documentation currency, stale comments — plus model-generated,
**topic-seeded** questions (`G1..`) across eight topics: correctness,
invariants, rules, parity, tests, rationale, scope, errors. One
`POST /chat/completions`.

Cross-vendor models routinely emit regex/shell fragments like `\s` inside JSON
string values, which are invalid JSON escapes. `repair_json()` escapes any
backslash that is not part of a valid JSON escape, leaving well-formed `\\`
pairs untouched, so the payload parses.

### answerer.py

Runs an OpenAI-compatible function-calling loop with **read-only tools
sandboxed to `--worktree`**: `read_file`, `grep`, `list_dir`, and — unless
`--no-exec` — a guarded `go build`/`go test`/`go vet`. It investigates the
PR-head code and returns, per question, a `status ∈ {CONFIDENT, CANNOT_ANSWER,
FLAW_FOUND}` with an `answer`, `evidence` (file:line or a repro), and an
optional non-blocking `note`.

**Issue-completeness tools (`gh_issue` / `pr_diff`, #1792).** The worktree tools
can read the PR-head code but cannot see *the issue the PR is meant to satisfy*
or *what actually changed*, so the fixed policy question F1 ("does this PR fully
implement the issue it closes?") was unanswerable from the worktree alone — it
fell to `CANNOT_ANSWER`, which is blocking, so the delivery loop's own qa-review
could never reach `PASS` from round 0. The answerer now carries the **same
read-only `gh_issue`/`pr_diff` tools the adjudicator already has** — `gh_issue`
reads the closing issue's acceptance criteria + comments, `pr_diff` reads the
base→head diff. They are ported verbatim from `adjudicator.py`, and because they
are read-only `gh` network calls (not PR-code execution) they stay under
`--no-exec`, which continues to drop only the code-executing `go` tool. The
answerer is told the PR number and the closing-issue number (`--pr` / `--issue`)
so it knows which to query; the system prompt directs it to consult them for
F1/F2/F3 but **not** to answer from the diff alone when the surrounding code
decides the answer — verify against the real code. `QA_REPO` targets the calls.

The tool loop has a finite budget (`MAX_TOOL_TURNS = 24`). Exhausting it is a
normal outcome, not an error, so the loop **degrades instead of crashing**: it
honors a final answer array the assistant already produced, and otherwise
reports every question `CANNOT_ANSWER`. That status is blocking, so a run that
ran out of turns can never silently `PASS`. Exhaustion is always logged to
stderr.

### render_report.py

Verdict is **`BLOCK` iff any answer status ∈ {FLAW_FOUND, CANNOT_ANSWER}**, else
`PASS`. The output is a one-line verdict **header**
(`## qa-review — PR #N: ✅ PASS` / `⛔ BLOCK`), a subtitle, the full untruncated
`ID | Topic | Result | Question | Answer` table, an **Items to fix** section
(blocking findings + evidence), and an **Important to consider** section
(non-blocking notes). `default_banner(qmodel, amodel)` is built from the actual
models so it can't drift.

The verdict lives in the header emoji — it is **not** a trailing machine marker.
Deriving the `QA-VERDICT: PASS|BLOCK` gate marker from this output is #1715's
job, not this tooling's.

### adjudicator.py

Reads the most recent qa-review **report** comment's **Items to fix**
(`select_report_comment` + `parse_items_to_fix`)
and every later comment (the author's responses), then per prior blocking
finding verifies with read-only worktree tools + `gh_issue` + `pr_diff` and
returns `RESOLVED` / `WAIVED_JUSTIFICATION` / `WAIVED_DEFERRED` / `STILL_OPEN`.
It **defaults to `STILL_OPEN`** (skeptical) and, on a genuine
acceptance-criterion interpretation fork, returns `STILL_OPEN` and asks the
author to pin the interpretation rather than guessing. The aggregate verdict is
`BLOCK` iff **any finding is `STILL_OPEN` or left un-adjudicated**, and is
emitted on **stderr** as `[adjudication verdict: PASS|BLOCK]` (not a PR marker —
#1716 derives the gate marker).

Its tool loop degrades on exhaustion the same way the answerer's does, to
`STILL_OPEN` for every prior finding — so running out of turns blocks rather
than clearing a finding it never actually adjudicated.

**Which comment supplies the prior findings (#1716).** Selection is structural,
not a substring search: a comment qualifies only if it *is* a rendered report —
an **unquoted, unfenced** `## qa-review — PR #` heading **and** an
`### Items to fix` section (fenced code blocks and `>` blockquotes are ignored,
because that is how a comment quotes a report it is discussing) — and, when
`--report-author` / `QA_REPORT_AUTHOR` is set, only if that login posted it. The
author restriction is strict: on a public repository, anything looser lets a
commenter supply a report-shaped comment with an empty Items-to-fix section and
clear every outstanding finding.

Finding **no** report comment is **not** a `PASS`: `fetch_comments` returns
`None` (distinct from `[]`, a real report with nothing blocking) and the tool
exits **3** with no verdict line at all, so no consumer can derive a marker from
a review that was never found. The pre-#1716 rule keyed off the last comment
*containing* the banner, so a self-review quoting it hijacked the selection and —
having no Items-to-fix section of its own — produced zero findings and a vacuous
`PASS` (observed on PR #1736).

## The `--no-exec` seam

`answerer.py` and `adjudicator.py` accept an **off-by-default `--no-exec`** flag
that drops the code-executing `go` tool from **both** the implementation map and
the advertised tool schema, leaving `read_file`/`grep`/`list_dir` and the
read-only `gh_issue`/`pr_diff` tools intact — **both agents** now carry the gh
tools (the answerer since #1792). It is a pure `tools_for(no_exec) -> (impl,
schema)` selector.

**Flag absent ⇒ the `go` tool is present ⇒ verbatim prototype behavior.** The
seam exists so #1715/#1716 can run the answerer/adjudicator on the self-hosted
runner while preserving `deliver-verify.yml`'s invariant that the PR's code is
never compiled or executed there.

## Orchestration (from the prototype `SKILL.md`)

1. Resolve the PR → materialize the PR head into a throwaway **read-only**
   worktree.
2. `questioner.py` → questions JSON.
3. `answerer.py --worktree <wt> --pr N --issue M` (optionally `--no-exec`),
   passing the PR + closing-issue numbers so its `gh_issue`/`pr_diff` tools can
   verify issue-completeness for F1 (#1792) → answers JSON.
4. `render_report.py --questions … --answers … --pr N [--post-to-pr]`.
5. Clean up the throwaway worktree.

For a re-verify round (#1716, adjudicate-only): skip the questioner/answerer and
run `adjudicator.py --worktree <wt> --pr N` against the prior findings + the
author's responses. (There is no `--adjudicate` flag; running `adjudicator.py`
instead of the questioner/answerer IS the adjudicate-only mode — the workflow
selects it by the round counter.)

## Environment surface

| Var | Meaning | Default |
|---|---|---|
| `OPENAI_BASE_URL` | LiteLLM proxy base URL | — (required) |
| `OPENAI_API_KEY` | proxy key; falls back to `LITELLM_KEY` | — (required) |
| `QA_QUESTIONER_MODEL` | questioner model | `gcp/gemini-3.6-flash` |
| `QA_ANSWERER_MODEL` | answerer model | `azure/gpt-5.6-sol` |
| `QA_ADJUDICATOR_MODEL` | adjudicator model | `azure/gpt-5.6-sol` |
| `QA_REPORT_AUTHOR` | adjudicator: only adjudicate a prior report posted by this comment author login | empty (any author) |
| `QA_REPO` | `owner/repo` for `gh` calls | `inference-sim/inference-sim` |
| `QA_REPO_DIR` | local clone the worktree is cut from | — |

## Tests

`scripts/qa_review_test.go` (`package scripts_test`) shells out to `python3`,
mirroring how `scripts/deliver_gate_test.go` shells out to `bash`, so every
qa-review test runs under `go test ./scripts/...` with no new test framework.
It covers only the **model-free** surface: `render_report.py`'s verdict rule and
output shape, `questioner.repair_json()`, `adjudicator.parse_items_to_fix()` and
its block rule, `adjudicator.fetch_comments()`'s report-comment selection
(including the quoting shapes that used to hijack it, and the fail-closed
refusal when no report exists), the `--no-exec` `tools_for()` seam for both agents
(now asserting the answerer carries the same read-only `gh_issue`/`pr_diff` set,
#1792), the answerer surfacing the `--pr`/`--issue` numbers into its prompt (#1792 —
`answerer_user_message` preamble structure plus a `main()` path that passes the
flags), and both agents' tool-loop exhaustion degradation (the one
`post_chat_completion` stub is the only model-dependent piece). The model-calling
paths need the live proxy and are not unit-tested here.

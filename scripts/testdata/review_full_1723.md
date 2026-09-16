**Claude finished @mtoslalibu's task in 18m 9s** —— [View job](https://github.com/inference-sim/inference-sim/actions/runs/35118608981)

---
### BLIS PR Review: hardening/1722-deliver-implement-crash-survivable

**Tasks:**
- [x] Parse trigger and understand round-4 review context
- [x] Read all changed files at current head (`f984bb9c`)
- [x] Run full BLIS-spec review (10 perspectives + Q/A phase)
- [x] Contract / issue-coverage verification
- [x] Post findings and verdict

---

## Archon Context

No `/archon-pr-review` output exists. #1722 has no `archon-plan:` and no `## Target branch` section; the dist ratchet reads `absent`, which is correct — this is a standalone hardening issue.

**Scope:** CI/workflow only — 2 workflow/docs files, 1 shell script, 2 Go test files. No simulator code, no CLI surface, no INV-1…INV-13 in scope. Perspectives 6–9 (DES / vLLM / Distributed / Performance) are N/A.

---

## Round-3 Findings — All Verified Addressed

| Round-3 finding | Status | Verification |
|---|---|---|
| Fence toggle discarded body on odd marker count (CommonMark fix) | Fixed as correctness bug | awk records delimiter char + run length; closes only on same char ≥ that length; `*_seen` on UNSTRIPPED body; plan loss is hard error at `:382-385`. Two round-3 test expectations **correctly reversed** since they pinned the buggy behavior. **Two residuals: F2, F4 below** |
| Prompt contradicted `pr-workflow.md` | Addressed | Three explicit overrides named at `:532-548`; comment at `:435-451` no longer overstates what the allowlist buys. **One wording gap: N5** |
| Seeding single point of failure for ratchet | Addressed | Agent's independent path restored at `:515-516`; hard error from fence fix makes seeding-miss unreachable silently |
| "for every phase" false | Addressed | Scoped to implement+verify; explicit paragraph naming the not-yet-fixed correct phase + link to #1737 |
| Guard delivery-scoped | Addressed | Test now globs `.github/workflows/*.yml`; verified to cover all 10 workflow files including `archon.yml` |
| `base != ''` silent stop | Addressed | Clause removed from work step `:621`; failure-reporter already handles seed failure |
| Push/verify interaction undocumented | Addressed in substance, **broken in mechanism → F1** | Note was written as YAML `#` inside a `prompt: |` block scalar |

---

## Review — 10 Perspectives

### Perspective 1 — Substance & Design

Design is right. The branch-first seeding / agent-opens-PR / incremental-push decomposition correctly maps to the two distinct failure modes (work survivability vs. delivery visibility). The deviation from #1722 item 2 (agent opens the PR using the App token instead of the workflow using GITHUB_TOKEN) is better argued than the issue's own prescription: the `can_approve_pull_request_reviews` single-toggle argument is correct, empirically grounded by PR authorship of #1680/#1708/#1713, enforced by a test, and its cost is stated rather than glossed. I accept that deviation. The empty seed commit being load-bearing (GitHub refuses a PR with no commits between base and head) is correctly identified and documented. The hand-off gating on file changes rather than PR existence is the right consequence of the ordering change.

**One behavioral observation (not a defect):** A deliberately-closed PR on the delivery branch will be replaced by the agent opening a new one (`:409` filters `--state open`), consistent with the "re-issuing resumes" semantics. Worth one sentence in the docs.

### Perspective 2 — Code Quality

`set -uo pipefail` without `-e` is deliberate and matches `archon-plan-resolve.sh`. Every failure path in the `seed` step has an explicit `|| { … exit 1; }` (lines 331, 335, 346, 358, 401-406, 409) — no silent continue, R1-clean. The four-line `key=value` protocol plus `sed -n 's/^key=//p'` is injection-safe by construction. Braced `${BRANCH}` variables (commit `6161675b`) prevent zsh history-modifier misreads when copying to local shells.

**N1 (minor):** `title=$(gh issue view … --json title --jq .title)` at `:335-338` is read and never used — one extra API call and one additional hard-failure mode for zero benefit.

### Perspective 3 — Test Behavioral Quality

Tests are behavioral on the script side: `TestDeliverSeedRefs` shells out to the real script and asserts observable stdout, so it would survive any awk/sed rewrite — the round-4 rewrite proves this, since only three expectations changed and they changed because the *behavior* was wrong. `TestDeliverImplementSeedsTheBranchBeforeTheAgentRuns` asserts step **order**, not mere presence — the right design for a positional invariant. `len(paths) < 5` and `checked == 0` anti-vacuity floors close the "test with no assertions" hole. The PR body's break-table (each assertion broken + confirmed to fail) is good discipline.

**Gaps:** No test asserts the prompt is free of stray `#` lines (**F1**). No test covers a heading-spanning fence (**F4**). Nothing connects `deliver-seed-refs.sh`'s `plan_seen=true, plan=''` output to the workflow's `exit 1` (**F2**) — the two live in different files.

### Perspective 4 — Getting-Started Experience

`automated-delivery.md` is unusually complete. The new PR-first section, three-consequences bullet, and resume-vs-dispatch-verify distinction all answer questions a maintainer would actually have. The Setup section's explicit "the toggle should stay disabled, here's why" block is the right place for it.

Two rough edges: the "tracked separately" dead end (**F3**), and a legitimate delivery being refused with a confusing error message (**F2**).

### Perspective 5 — Automated Reviewer Simulation

F1 is exactly what a YAML-aware linter would flag first (`#` inside a block scalar). F2 is the raw-vs-processed asymmetry that a static analysis tool would flag as "condition computed on a different input than the value it guards." N1 (assigned and never used) is shellcheck SC2034.

### Perspectives 6–9 — N/A

No events, no clock, no batching, no KV, no routing. The PR body's note that `timeout-minutes` is unrelated to BLIS's `TimeoutEvent` / `--timeout` pre-empts the one plausible false positive here.

### Perspective 10 — Security & Robustness

Strongest area. Shape allowlist `^[A-Za-z0-9._/-]+$` + leading-dash guard + `..`/`.lock`/absolute-slash refusals + all 11 hostile refs tested. The shape validation is also what makes the `github-script` interpolations (`const base = '${{ steps.seed.outputs.base }}'`) injection-safe. Invocation is collaborator-gated. The `can_approve_pull_request_reviews` toggle reasoning is correct: verified by three existing delivery PRs authored by `app/claude`, not `github-actions[bot]`.

---

## Q/A Phase

15 questions generated; all investigated with file:line evidence. Key results:

| Question | Verdict |
|---|---|
| Can `infence` be entered with `fencelen < 3`? | **CONFIDENT — no** (`deliver-seed-refs.sh:114`, `n >= 3` gate) |
| Does a closing fence with trailing text wrongly leave the fence open? | **CONFIDENT — no** (`gsub` + `tail != ""` check is CommonMark-correct) |
| Can a fenced `archon-plan:` declaration produce `plan_seen=true` + hard error? | **FLAW_FOUND → F2** (verified live) |
| Can fence stripping delete a `##` boundary and let the wrong ref win silently? | **FLAW_FOUND → F4** (verified live) |
| CRLF handling on both piped and file paths? | **CONFIDENT — yes** (`:64`, `:345`; pinned by CRLF test case) |
| Can the agent re-read the issue for the plan fallback? | **CONFIDENT — yes** (`Bash` in `--allowed-tools` `:432`; but path is unreachable per Finding 3 analysis) |
| Can `BRANCH` break the refspec? | **CONFIDENT — no** (always `deliver/issue-$number`, braced) |
| Can both reporters fire simultaneously? | **CONFIDENT — mutually exclusive** (`success()` vs `!success()`) |

---

## Issue Coverage Matrix

| #1722 item | Delivered | Notes |
|---|---|---|
| 1 · timeout 60 → 120 | ✅ `:249` | Tested against sweep window |
| 2 · Branch + draft PR before implementing | ✅ with deviation | Branch workflow-pushed; **PR opened by agent** (deviation argued + tested). Residual visibility hole: **F3** |
| 2b · Move `Depends on:` check before PR | ✅ by correcting the issue | Check was already workflow logic in `check-permissions`; prompt's contradictory copy removed |
| 3 · Push incrementally | ✅ `:560-563` | **Undermined by F1**: a cost-of-pushing sentence currently sits in the agent's instructions immediately above this bullet |
| 4 · Resume rather than restart | ✅ resume path `:395-402` | Explicit-refspec fetch; `RESUMING` in prompt `:526-530` |
| 5 · Stop duplicating `ci.yml` | ✅ `:564-568` | Both banned patterns tested; previously-dead `golangci-lint` assertion fixed |
| 6 · Non-interactive + skill allowlist | ✅ | Three explicit overrides + `UNATTENDED`; all six skills asserted by name |
| — · Fix reporters | ✅ `:677-774` | No guessed causes; `changed == ''` third state ("could not be determined") is a good touch |

---

## Findings Summary

| Severity | Finding | Location |
|---|---|---|
| **Important** | Six intended YAML comments sit inside `prompt: \|` and are sent to the agent; one argues against the adjacent `PUSH AS YOU GO` instruction | `.github/workflows/deliver-implement.yml:554-559` |
| **Important** | Plan hard-error fires on a body that only *quotes* `archon-plan:` inside a fence, refusing a legitimate delivery with a misdescriptive error; asymmetric with the heading half which warns | `deliver-implement.yml:382-385` + `deliver-seed-refs.sh:203-206`; test at `deliver_seed_refs_test.go:212-218` pins the output but not its consequence |
| **Important** | The PR's central residual hole (PR-less delivery branch invisible to the sweep) is claimed "tracked separately" in three places, but no issue exists for it | `deliver-implement.yml:319-320`, `automated-delivery.md`, PR body |
| **Important** | Fence stripping deletes a `##` section boundary; a heading-spanning fence lets a later section's ref become the base **silently** (no warning, no error) | `scripts/deliver-seed-refs.sh:95-150` |
| Minor | `title` read from the API at `:335-338` and never used — one extra API call, one extra failure mode | `deliver-implement.yml:335-338` |
| Minor | Repo-wide guard globs `*.yml` only; missing `gh api -X POST .../pulls` variant | `deliver_guards_test.go:176, 215` |
| Nit | Override 2 disclaims only the full-repository lint; `pr-workflow.md:227` also mandates per-package lint | `deliver-implement.yml:541-545` |

---

## Verdict

**Not quite ready to merge — one fix required, three worth resolving first.**

The design is right and the arguments are sound. Five of @jgchn's seven findings are cleanly addressed, and the two remaining are addressed in substance. CI is green on `f984bb9c`.

**Required before merge:**
- **F1** — move lines 554-559 out of the `prompt: |` block scalar (the six lines are at the same indentation as the surrounding prompt bullets; YAML does not recognize `#` inside a block scalar). Place them above `prompt:` alongside the existing plugin-list comment at `:435-451`. Optionally add `TestDeliverImplementPromptHasNoStrayYamlComments` so this class of mistake can't recur.

**Strongly recommended (author's call on approach):**
- **F2** — either downgrade to a `::warning::` and lean on the restored agent fallback (finding 3 makes this safe), narrow the trigger to genuine unclosed-fence loss, or rewrite the message to accurately describe the fenced-example case.
- **F3** — file the sweep follow-up issue and cite the number in all three places, exactly as #1737 was handled.
- **F4** — one test case for a heading-spanning fence, and consider blanking (rather than deleting) stripped fence lines to preserve section boundaries.

---

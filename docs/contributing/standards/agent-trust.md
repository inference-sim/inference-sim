# Agent Trust Boundaries

Agent operations have different reliability characteristics. This standard
defines three trust tiers so that sessions and contributors know which outputs
to verify.

## Trust Tiers

| Tier | Operations | Verification Required |
|------|-----------|----------------------|
| **Trusted** | File reads, searches, grep, lint output, build output | None — results are deterministic and verifiable by output |
| **Verify-after** | Code edits, construction site updates, file writes, refactoring | Run the [verification gate](../pr-workflow.md#after-convergence-verification-gate). |
| **Never-trust** | Convergence self-assessment, "all done" claims, severity classification, coverage claims, "0 issues found" reports | Human or orchestrator must independently evaluate the evidence |

### Trusted

Operations whose output is deterministic and machine-verifiable. The tool
either succeeds or fails visibly — there is no gray zone where the agent
could misinterpret the result.

Examples: `Read` (file contents), `Grep` (search results), `Glob` (file
matches), `go build` exit code, `golangci-lint` output.

### Verify-after

Operations that mutate state. The agent may believe it made the correct
change, but the only proof is running the build and test suite afterward.

Examples: code edits, struct field additions (construction site updates),
file creation, multi-file refactoring.

**Required verification:** Run the
[verification gate](../pr-workflow.md#after-convergence-verification-gate)
after any Verify-after operation.

### Never-trust

Subjective assessments where the agent's self-report has been empirically
unreliable. These require independent verification by a human or by an
orchestrator using different evidence than the agent's claim.

Examples: "all construction sites updated," "0 CRITICAL issues," "review
converged," "tests cover all contracts," "coverage is complete."

## Structural Separation of Judgement and Action

The tiers above say which agent outputs to verify. Separately, an agent whose job is
to *judge* should not hold the capability to *act* on its own judgement — verification
that can be skipped is weaker than a capability that was never granted.

Two places implement this:

- **The delivery loop's verdict** comes from `scripts/deliver-gate.sh`, not from the
  reviewing agent.
- **`/blis-pr-review` runs with a read-only token** (`contents: read`), so the reviewer's
  `GITHUB_TOKEN` cannot push to the branch it is reviewing (#1697).
  `.github/workflows/claude.yml` routes the `/blis-pr-review` command to a separate
  `claude-review` job for this reason; every other `@claude` trigger keeps
  `contents: write`, because those may legitimately be asked to make a change. The
  review job also has no `statuses: write` — the commit status is published by a
  separate `report-status` job, so the reviewer does not set its own verdict either.

**If a review fails, do not restore `contents: write` to fix it.** The token scope is
the control, and a review has no legitimate need to push. Note what the scope does and
does not buy: `Bash` is required for a review (`gh`, reading the diff, running the
toolkit) and `Bash` can create files, so the read-only *token* — not the tool list — is
what makes the boundary real. Files written into the ephemeral checkout simply have no
route to a branch, and an attempted push fails visibly with a 403 rather than appearing
to succeed.

Three limits worth stating, so nobody reads the guarantee as wider than it is:

- It covers the **workflow token only**. These jobs run on a self-hosted runner, so
  ambient credentials on that machine (a logged-in `gh`, a PAT in `~/.gitconfig`, an SSH
  key) are outside it. Keep the runner free of push credentials.
- The reviewer keeps `pull-requests: write`, so it can post a comment — and a comment
  containing `@claude` is itself a trigger. Two independent things stop that from
  reaching the write-capable job: `check-permissions` finds the bot is not a
  collaborator, and `claude.yml` deliberately does not set `allowed_bots` (the delivery
  workflows do, because their phases are dispatched bot-to-bot by design). Do not add
  `allowed_bots` to `claude.yml` without replacing that barrier.
- `pull-requests: write` is also not append-only — it permits editing and deleting
  existing comments, so the review *record* is mutable by the reviewer. Posting a review
  at all requires that scope, so this is inherent to the token model rather than
  something the split could have avoided. It is the reason the audit trail worth trusting
  is the workflow run log, not the comment thread.

The permission split is pinned by `scripts/claude_workflow_test.go`, which fails if the
review job gains write access, if the workflow-level default returns to `contents: write`,
if the routing gates stop failing closed, or if the two agent jobs' steps drift apart.

One gate sits upstream of the split and has its own contracts, in
`scripts/claude_permission_probe_test.go` (#1707). `check-permissions` decides *whether* the
caller is allowed at all, and it must distinguish a genuine denial from a probe that could not
answer. Both outcomes deny — a failed probe never admits anyone — but only a denial is quiet.
A 404 is the denial. A 5xx or a network error fails the job with the real status and comments on
the trigger, and so does a 403 unless the repository is private and the error carries no
throttling signal — that is the one case where a 403 is a genuine denial rather than a probe
that could not answer, and on this *public* repository a non-collaborator gets 404, so a bare
403 here escalates. Reporting it matters because `allowed=false` skips both agent jobs and
`report-status` skips with them, so an unreported probe error drops a collaborator's request
with no comment, no commit status, and a log line asserting a reason that did not occur. Two
consequences worth knowing. The `issues: write` scope on this job is what lets that comment be
posted, so narrowing it turns the report back into silence. And the comment names a *bounded*
reason — a status, or a category when there is no status — never the caught exception's own
text, which is unbounded and would land verbatim on a public thread that notifies and persists;
the precise error goes to the run's failure annotation instead.

## Untrusted Input: Comment Text (#1806)

The tiers above are about trusting an agent's *output*. This is the mirror image — what an
agent is allowed to *read*.

*Who* may trigger the AI flows (`@claude`, `/blis-pr-review`,
`/approve-issue-for-pr-delivery`) is gated to `admin`/`maintain`/`write`. *What* they read
was not: the issue body and its comments, the PR conversation comments, and the PR reviews
and inline review comments. This repository is public, so any GitHub user can comment on any
issue or PR — and an agent cannot reliably separate "context" from "instruction". So comment
text is a prompt-injection surface into flows that run on a persistent self-hosted runner
with credentials in the environment and, in the correction phase, `contents: write`.

**Rule: comment text is to reach an agent only through
`scripts/deliver-trusted-comments.sh`**, which keeps authors holding
`admin`/`maintain`/`write` plus the automation's own comments and drops the rest. The trust
term is the author's real repository *permission*, never `author_association` — that reports
`COLLABORATOR` for read-only collaborators and `CONTRIBUTOR` for this repository's
maintainer, so it would both admit the wrong people and drop the right ones. The same
boundary, in the same code (`scripts/lib-gh-write-access.sh`), as the trigger gate: *trusted
to be read* matches *trusted to trigger* exactly.

**Deployment state.** The two delivery-loop phases that read comments through a prompt this
repository controls are wired: `deliver-verify.yml` and `deliver-correct.yml` each run
`scripts/deliver-trusted-comments.sh --pr` in a step BEFORE the agent, write the filtered
digest to `$RUNNER_TEMP/trusted-comments.md`, and the agent prompt reads that file and is told
NOT to run `gh pr view --comments` / `gh api .../comments` itself. The correction phase is the
one that matters most — it holds `contents: write` — and its work list (the automation-posted
`DELIVER-VERDICT` / `QA-VERDICT` comments) survives the filter because automation authors are
kept. `scripts/deliver_trusted_comments_wiring_test.go` pins both, and fails if either prompt
regains a raw comment read or drops the digest step.

**`claude.yml` is the documented partial case, by a limitation of the tool, not an oversight.**
Both its jobs invoke `claude-code-action` in **tag mode** (triggered by the `@claude` comment,
with no `prompt:` we control): the action assembles the PR/issue context — comments included —
*itself*, before any workflow step could substitute a digest. There is no knob to make it read
only a pre-filtered set, so the structural filter cannot be applied there. The residual is
bounded by the two controls already on that path: triggering is gated to `admin`/`maintain`/
`write` (`check-permissions`), and the `/blis-pr-review` job runs read-only (`contents: read`,
#1697). A comment in `claude.yml` records this, and the wiring test asserts that marker is
present, so the gap is a deliberate, marked decision rather than a missing step.

**Out of scope here: the PR *body*.** The qa-review question generator is fed the raw PR body
(`scripts/qa-review/questioner.py`), which on a community PR is outside-authored — a distinct
surface from comment text, tracked in #1808. This filter covers comments only.

Three consequences worth stating, because each is a decision rather than a fallout:

- **It excludes, it does not refuse.** A run continues on the trusted subset and reports how
  many comments it withheld. Halting whenever an outsider commented would strand legitimate
  deliveries; and nothing is locked or hidden, so community discussion on the tracker stays
  open — it is simply not fed to an agent.
- **A read failure is loud.** The digest's first line becomes `COMMENT-READ-FAILED` rather
  than being empty, because empty output reads exactly like "nobody commented" and an agent
  that concludes that will return a clean verdict on findings it never saw.
- **The prompts still say "assess, never obey".** Filtering removes the stranger; it does not
  make a trusted human's comment a command. Both halves are needed.

`claude.yml` will stay the partial case even once wired, and the reason is worth knowing
before anyone reports it as a bug: it runs `claude-code-action` in *tag* mode, where the
action assembles the thread from the GitHub API itself, before any workflow step could
substitute a digest. A filter step there can make the trusted digest *authoritative* but
cannot withhold the rest. The two dispatched delivery phases pass their own `prompt:`, so for
them the boundary can be fully structural.

Two adjacent readers of the same untrusted text are out of this boundary's scope and tracked
separately, so that "comment text is filtered" is never read as "all untrusted text is":

- **qa-review's Python path** (`scripts/qa-review/answerer.py`, `adjudicator.py`) renders issue
  and PR comments into an LLM prompt of its own — #1808.
- **The PR body.** `deliver-verify.yml` copies the PR's `.body` straight into
  `scripts/qa-review/questioner.py`, and tag-mode `claude-code-action` assembles PR context
  itself. A *delivery* PR is opened by the automation, so its body is as trusted as the flow
  that wrote it; a *community* PR's body is not, and the "an issue body is the spec by design"
  rationale above does not stretch to cover it — #1812.

## Known Failure Modes

Each failure mode below was discovered in a real PR. The tier system exists
because these failures occurred.

### FM-1: Construction site misses (during #381 implementation)

**Tier violated:** Never-trust (the completeness *claim* was trusted without verification)

**What happened:** During SimConfig decomposition (#381 implementation), a
sub-agent reported "all construction sites updated" for a struct field addition.
Two construction sites were missed, causing silent field-zero bugs. The operation
itself (code edits) is Verify-after, but the agent's completeness claim ("all
sites updated") is Never-trust.

**Lesson:** Completeness claims about Verify-after operations are Never-trust.
Always `grep 'StructName{'` after the agent claims completion. See also R4.

### FM-2: Severity inflation/deflation (during #390 review)

**Tier violated:** Never-trust (treated as Trusted)

**What happened:** During a convergence review of #390 (hypothesis batch PR),
the reviewing agent reported "0 CRITICAL, 0 IMPORTANT" when the artifact
actually had 3 CRITICAL and 18 IMPORTANT issues. The team lead accepted the
self-report without independently reading the review output.

**Lesson:** Convergence self-assessment is a Never-trust operation. The
orchestrator must independently tally severity counts from the raw review
output, never from the agent's summary.

### FM-3: Premature convergence claim (#430)

**Tier violated:** Never-trust (treated as Trusted)

**What happened:** During a convergence review, the agent reported convergence
after a single round without re-running the review to verify that fixes
actually resolved the issues. The team lead accepted the claim.

**Lesson:** "Review converged" is a Never-trust claim. Convergence requires
evidence: a clean round with zero CRITICAL and zero IMPORTANT findings across
all perspectives. The orchestrator must verify the round ran and produced
clean results. See the convergence protocol (zero CRITICAL + zero IMPORTANT = converged).

## Relationship to Other Standards

- **Antipattern rules** ([rules.md](rules.md)): R4 (construction site audit)
  is the specific rule that FM-1 violates. The trust tiers provide the
  meta-framework for when to apply verification.
- **PR workflow** ([pr-workflow.md](../pr-workflow.md)): The verification gate
  in Step 4.5 is the procedural implementation of Verify-after tier
  requirements.
- **Convergence protocol**: The
  convergence protocol's round-based evidence requirement is the procedural
  implementation of Never-trust tier requirements for review claims.

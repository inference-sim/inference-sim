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

Two limits worth stating, so nobody reads the guarantee as wider than it is:

- It covers the **workflow token only**. These jobs run on a self-hosted runner, so
  ambient credentials on that machine (a logged-in `gh`, a PAT in `~/.gitconfig`, an SSH
  key) are outside it. Keep the runner free of push credentials.
- The reviewer keeps `pull-requests: write`, so it can post a comment — and a comment
  containing `@claude` is itself a trigger. Two independent things stop that from
  reaching the write-capable job: `check-permissions` finds the bot is not a
  collaborator, and `claude.yml` deliberately does not set `allowed_bots` (the delivery
  workflows do, because their phases are dispatched bot-to-bot by design). Do not add
  `allowed_bots` to `claude.yml` without replacing that barrier.

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

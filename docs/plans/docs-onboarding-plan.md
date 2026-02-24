# Documentation Onboarding Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the BLIS codebase welcoming to new contributors by closing 8 documentation gaps that increase friction for first-time contributors.

**The problem today:** New contributors face a cognitive load cliff: the path from "I want to contribute" to "I opened my first PR" requires reading 1500+ lines of process documentation. Key information is missing (lint tool installation, correct Contributing link), scattered (no table of contents in 777-line README), or conflated (Claude Code instructions mixed with architecture reference in CLAUDE.md). Human contributors without AI tooling have no simplified workflow path.

**What this PR adds:**
1. Quick wins — fix the README Contributing link, add golangci-lint install instructions, annotate Quick Start with expected output
2. Navigation — Tables of Contents in README.md and docs/process/pr-workflow.md
3. Priority guidance — tiered rules in docs/standards/rules.md so newcomers know what matters most
4. Human-contributor workflow — simplified 5-step path without Claude Code skills
5. First contribution walkthrough — complete tutorial for adding a trivial admission policy
6. CLAUDE.md reorganization — separate behavioral instructions from reference material

**Why this matters:** The standards and process documentation is excellent but the onboarding ramp is steep. These changes lower the barrier for the project's first external contributors without changing any code.

**Architecture:** Documentation-only changes to 5 files: README.md, CONTRIBUTING.md, CLAUDE.md, docs/standards/rules.md, docs/process/pr-workflow.md. No Go code, no new packages, no interface changes.

**Source:** GitHub issue #422

**Closes:** Fixes #422

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR modifies 5 documentation files to improve new-contributor onboarding. The changes range from single-line fixes (README Contributing link) to substantial new sections (first-contribution walkthrough in CONTRIBUTING.md). No Go code is modified. All existing tests and lint continue to pass unchanged.

The changes are ordered by impact: quick fixes first, then navigation aids, then substantial new content. Each task is independently useful — partial completion still improves onboarding.

Four deviations from the issue description (see Deviation Log): task regrouping, tier assignment corrections, heading name generalization, and scope simplification.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: README Contributing Link
- GIVEN a reader reaches the Contributing section of README.md
- WHEN they look for how to contribute
- THEN they MUST find a link to CONTRIBUTING.md (not `docs/plans/`)
- MECHANISM: Replace the Contributing section text with a direct link to CONTRIBUTING.md

BC-2: Lint Tool Installation
- GIVEN a new contributor follows CONTRIBUTING.md Quick Start
- WHEN they encounter `golangci-lint run ./...`
- THEN the preceding text MUST include installation instructions for golangci-lint
- MECHANISM: Add install command before the lint step in Quick Start

BC-3: Quick Start Interpretation
- GIVEN a reader runs the Quick Start command
- WHEN they see JSON output
- THEN the README MUST explain what the key metrics mean, immediately after the command
- MECHANISM: Add "What you should see" annotation with key metric definitions

BC-4: README Navigation
- GIVEN a reader opens the 777-line README.md
- WHEN they want to find a specific section
- THEN a Table of Contents at the top MUST list all major sections with anchor links
- MECHANISM: Add markdown ToC after the project description

BC-5: PR Workflow Navigation
- GIVEN a reader opens the 1119-line pr-workflow.md
- WHEN they want to find a specific step
- THEN a Table of Contents at the top MUST list all major sections with anchor links
- MECHANISM: Add markdown ToC after the status line

BC-6: Rule Priority Tiers
- GIVEN a new contributor reads rules.md
- WHEN they want to know which rules matter most
- THEN rules MUST be grouped into priority tiers (Critical/Important/Hygiene) with rationale
- MECHANISM: Add a Priority Tiers section after the introductory paragraph

BC-7: Human Contributor Path
- GIVEN a human contributor not using Claude Code
- WHEN they read the development workflow
- THEN CONTRIBUTING.md MUST describe a simplified 5-step workflow without Claude Code skills
- MECHANISM: Add "Human Contributor Quick Path" section to CONTRIBUTING.md

BC-8: First Contribution Walkthrough
- GIVEN a first-time contributor to BLIS
- WHEN they want to make their first PR
- THEN CONTRIBUTING.md MUST provide a complete tutorial showing every file, test, and command
- MECHANISM: Add "Your First Contribution" section with a worked example

BC-9: CLAUDE.md Section Organization
- GIVEN a reader (human or AI) opens CLAUDE.md
- WHEN they look for behavioral instructions vs reference material
- THEN behavioral instructions (Context Management, Task Agent, Macro Plan Updates) MUST be grouped under a distinct heading separate from architecture reference
- MECHANISM: Add "## Agent Behavioral Instructions" heading and move relevant sections under it

**Negative Contracts:**

BC-10: No Content Loss
- GIVEN any documentation change in this PR
- WHEN compared to the original file
- THEN no existing content MUST be deleted — only reorganized, augmented, or corrected
- MECHANISM: Each task explicitly preserves existing content

BC-11: No Code Changes
- GIVEN this is a documentation-only PR
- WHEN `go test ./...` and `golangci-lint run ./...` are run
- THEN both MUST pass with identical results to before the PR
- MECHANISM: Only .md files are modified

### C) Component Interaction

No component interaction — documentation-only changes. Five .md files modified independently.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue lists 8 items numbered in a specific order | Plan regroups into 7 tasks with different ordering (quick wins first) | SIMPLIFICATION: Grouping related items (e.g., README Contributing fix + Quick Start annotation) reduces task count and improves execution flow |
| Issue proposes Critical: R1,R4,R5,R6,R11,R19; Important: R2,R3,R7-R10,R13,R14; Hygiene: R12,R15-R18,R20 | Plan moves R17,R18,R20 from Hygiene to Important | CORRECTION: R17 (signal freshness — 200x worse distribution per H3), R18 (flag precedence — caused invalid H9 results), R20 (degenerate inputs — false negatives in detectors) produce wrong results (Important, not Hygiene). Critical and remaining Important tiers match the issue. |
| Issue item 8 says "Claude Code Behavioral Instructions" heading | Plan uses "Agent Behavioral Instructions" | CORRECTION: More generic heading includes non-Claude AI assistants |
| Issue item 8 says "Separate CLAUDE.md behavioral instructions from reference material" | Plan groups 3 specific sections under a new heading but does not restructure the entire file | SIMPLIFICATION: Moving 3 sections under a new heading achieves the goal with minimal disruption |

### E) Review Guide

1. **The tricky part:** The first-contribution walkthrough (Task 6) must be technically accurate — every file path, interface name, and test command must match the current codebase. A stale path would be worse than no walkthrough.
2. **What to scrutinize:** BC-8 (walkthrough accuracy), BC-4/BC-5 (ToC anchor links must match actual headings).
3. **What's safe to skim:** BC-1 (one-line fix), BC-2 (install command), BC-11 (no code changes).
4. **Known debt:** The walkthrough teaches adding an admission policy, which is the lightest extension type (~3 files). A routing policy walkthrough would be more realistic but heavier.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `README.md` — fix Contributing link, add ToC, annotate Quick Start (Tasks 1, 3)
- `CONTRIBUTING.md` — add lint install, human path, first-contribution walkthrough (Tasks 2, 5, 6)
- `docs/standards/rules.md` — add priority tiers (Task 4)
- `docs/process/pr-workflow.md` — add ToC (Task 3)
- `CLAUDE.md` — reorganize sections (Task 7)

**Key decisions:**
- Tables of Contents are manually maintained (no tooling dependency)
- First-contribution walkthrough uses admission policy (lightest extension type)
- Priority tiers use 3 levels: Critical (correctness), Important (quality), Hygiene (maintenance)

**Confirmation:** No dead code, all changes are immediately useful.

### G) Task Breakdown

---

### Task 1: Fix README Contributing Section + Add Quick Start Annotation

**Contracts Implemented:** BC-1, BC-3

**Files:**
- Modify: `README.md:69-77` (Quick Start section), `README.md:769-771` (Contributing section)

**Step 1: Fix Contributing section**

In `README.md`, replace the Contributing section (near line 769-771):

Current:
```markdown
## Contributing

Contributions are welcome! Please see the design documents in `docs/plans/` for ongoing work and architectural decisions.
```

Replace with:
```markdown
## Contributing

Contributions are welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for the engineering standards, development workflow, and step-by-step guides for adding new components. For ongoing work and architectural decisions, see `docs/plans/`.
```

**Step 2: Add Quick Start expected output annotation**

In `README.md`, after the Quick Start code block (after line 74), add:

```markdown
You should see JSON output like:

```json
{
  "completed_requests": 100,
  "tokens_per_sec": 492.02,
  "ttft_mean_ms": 25.08,
  "e2e_mean_ms": 4541.01,
  ...
}
```

**Key metrics:** TTFT (Time to First Token) measures how quickly the first output token arrives. E2E (End-to-End) is the total request latency. `tokens_per_sec` is output token throughput (decode tokens per simulated second). See [Example Output](#example-output) for the full schema.
```

**Step 3: Verify no build/test regression**

Run: `go build ./... && go test ./... 2>&1 | tail -3`
Expected: All packages pass (documentation-only change)

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): fix Contributing link and annotate Quick Start (BC-1, BC-3)

- Point Contributing section to CONTRIBUTING.md instead of docs/plans/
- Add expected output annotation after Quick Start command
- Add brief metric definitions (TTFT, E2E, tokens/sec)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add golangci-lint Installation Instructions

**Contracts Implemented:** BC-2

**Files:**
- Modify: `CONTRIBUTING.md:7-16` (Quick Start section)

**Step 1: Add install instructions**

In `CONTRIBUTING.md`, replace the Quick Start section:

Current:
```markdown
## Quick Start

```bash
# Build
go build -o simulation_worker main.go

# Test
go test ./...

# Lint
golangci-lint run ./...
```

All three must pass before submitting a PR.
```

Replace with:
```markdown
## Quick Start

```bash
# Build
go build -o simulation_worker main.go

# Test
go test ./...

# Install linter (one-time setup)
go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.9.0

# Lint
golangci-lint run ./...
```

All three must pass before submitting a PR. CI uses golangci-lint v2.9.0 (see `.github/workflows/ci.yml`).
```

**Step 2: Verify no build/test regression**

Run: `go build ./... && go test ./... 2>&1 | tail -3`
Expected: All packages pass

**Step 3: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs(contributing): add golangci-lint install instructions (BC-2)

- Add one-time setup command for golangci-lint in Quick Start
- Note CI version (v2.9.0) for version pinning awareness

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add Tables of Contents to README.md and pr-workflow.md

**Contracts Implemented:** BC-4, BC-5

**Files:**
- Modify: `README.md` (add ToC after line 7)
- Modify: `docs/process/pr-workflow.md` (add ToC after line 5)

**Step 1: Add Table of Contents to README.md**

After the project description paragraph (after line 6, before the `---` on line 7), insert:

```markdown

## Table of Contents

- [Features](#features)
- [Supported Models](#supported-models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Preset Workloads](#preset-workloads)
  - [Custom GPU, TP, vLLM Versions](#custom-gpu-tp-vllm-versions)
  - [Custom Workload Distribution](#custom-workload-distribution)
  - [Custom vLLM Configs](#custom-vllm-configs)
  - [Replay Workload Traces](#replay-workload-traces)
  - [Multi-Instance Cluster Simulation](#multi-instance-cluster-simulation)
  - [Tiered KV Cache](#tiered-kv-cache-gpu--cpu-offloading)
  - [Priority and Scheduling Policies](#priority-and-scheduling-policies)
  - [Fitness Evaluation and Anomaly Detection](#fitness-evaluation-and-anomaly-detection)
  - [Policy Configuration Files (YAML)](#policy-configuration-files-yaml)
  - [ServeGen-Informed Workload Generation](#servegen-informed-workload-generation)
  - [Decision Tracing and Counterfactual Analysis](#decision-tracing-and-counterfactual-analysis)
- [Latency Estimation Approaches](#latency-estimation-approaches)
- [Example Output](#example-output)
- [Debugging and Observability](#debugging-and-observability)
- [Evolutionary Policy Optimization](#evolutionary-policy-optimization-in-progress)
- [Project Structure](#project-structure)
- [CLI Reference](#cli-reference)
- [Contributing](#contributing)
- [License](#license)

```

**Step 2: Add Table of Contents to pr-workflow.md**

After the status line (line 3), before "This document describes" (line 5), insert:

```markdown

## Table of Contents

- [Current Template Versions](#current-template-versions)
- [Prerequisites](#prerequisites)
- [Overview](#overview)
- [Quick Reference: Simplified Invocations](#quick-reference-simplified-invocations)
- [Step-by-Step Process](#step-by-step-process)
  - [Step 1: Create Isolated Worktree](#step-1-create-isolated-worktree-using-using-git-worktrees-skill)
  - [Step 2: Create Implementation Plan](#step-2-create-implementation-plan-using-writing-plans-skill)
  - [Step 2.5: Multi-Perspective Plan Review](#step-25-multi-perspective-plan-review-rounds)
  - [Step 3: Human Review of Plan](#step-3-human-review-of-plan)
  - [Step 4: Execute Plan](#step-4-execute-plan-using-executing-plans-skill)
  - [Step 4.5: Multi-Perspective Code Review](#step-45-multi-perspective-code-review-rounds)
  - [Step 4.75: Pre-Commit Self-Audit](#step-475-pre-commit-self-audit-no-agent--deliberate-thinking)
  - [Step 5: Commit, Push, and Create PR](#step-5-commit-push-and-create-pr-using-commit-commandscommit-push-pr)
- [Workflow Variants](#workflow-variants)
- [Skill Reference Quick Guide](#skill-reference-quick-guide)
- [Example A: Macro Plan PR](#example-a-macro-plan-pr-workflow-same-session-with-worktrees)
- [Example B: Issue/Design-Doc PR](#example-b-issuedesign-doc-pr-workflow)
- [Tips for Success](#tips-for-success)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Appendix: Workflow Evolution](#appendix-workflow-evolution)

```

**Step 3: Verify no build/test regression**

Run: `go build ./... && go test ./... 2>&1 | tail -3`
Expected: All packages pass

**Step 4: Commit**

```bash
git add README.md docs/process/pr-workflow.md
git commit -m "docs: add Tables of Contents to README and pr-workflow (BC-4, BC-5)

- README.md: 20-entry ToC covering all major sections
- pr-workflow.md: 15-entry ToC covering all steps and appendices

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add Priority Tiers to rules.md

**Contracts Implemented:** BC-6

**Files:**
- Modify: `docs/standards/rules.md` (add section after line 8)

**Step 1: Add Priority Tiers section**

After the enforcement checkpoints paragraph (after line 8, before `## Rules`), insert:

```markdown

## Priority Tiers

New contributors: focus on **Critical** rules first. These protect correctness — violating them produces wrong results or crashes. **Important** rules protect code quality and maintainability. **Hygiene** rules keep the codebase clean over time.

| Tier | Rules | Why |
|------|-------|-----|
| **Critical** (correctness) | R1, R4, R5, R6, R11, R19 | Violations produce silent data loss, panics, conservation invariant breaks, or infinite loops |
| **Important** (quality) | R2, R3, R7, R8, R9, R10, R13, R14, R17, R18, R20 | Violations produce non-determinism, validation gaps, silent misconfig, interface debt, or undetected anomalies |
| **Hygiene** (maintenance) | R12, R15, R16 | Violations produce stale references, config sprawl, or misleading test baselines |

All 20 rules apply to every PR. The tiers help you prioritize during review — check Critical rules first.

```

**Step 2: Verify no build/test regression**

Run: `go build ./... && go test ./... 2>&1 | tail -3`
Expected: All packages pass

**Step 3: Commit**

```bash
git add docs/standards/rules.md
git commit -m "docs(standards): add priority tiers to antipattern rules (BC-6)

- Three tiers: Critical (correctness), Important (quality), Hygiene (maintenance)
- Helps new contributors focus on what matters most during review

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Add Human Contributor Quick Path to CONTRIBUTING.md

**Contracts Implemented:** BC-7

**Files:**
- Modify: `CONTRIBUTING.md` (add section after Development Workflow, before Engineering Principles)

**Step 1: Add Human Contributor Quick Path**

After the Development Workflow section (after line 34, before `## Engineering Principles` on line 35), insert:

```markdown

## Human Contributor Quick Path

If you are not using Claude Code, here is the simplified workflow:

1. **Branch** — `git checkout -b feature/my-change`
2. **Plan** — write an implementation plan following `docs/templates/micro-plan.md`. Include behavioral contracts (GIVEN/WHEN/THEN) and a task breakdown. Post the plan as a PR draft or issue comment for review.
3. **Implement** — follow TDD: write a failing test, implement the minimal code to pass it, run `go test ./...`, run `golangci-lint run ./...`, commit. Repeat for each contract.
4. **Self-review** — check the [Antipattern Checklist](#antipattern-checklist) below. Run `go build ./... && go test ./... && golangci-lint run ./...` one final time.
5. **PR** — push your branch and open a PR. Maintainers will run the automated review protocols (convergence-review with 10 perspectives).

The automated review tools (convergence-review, pr-review-toolkit) are run by maintainers — you do not need Claude Code installed. Your PR will go through the same quality gates regardless of tooling.

```

**Step 2: Verify no build/test regression**

Run: `go build ./... && go test ./... 2>&1 | tail -3`
Expected: All packages pass

**Step 3: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs(contributing): add Human Contributor Quick Path (BC-7)

- 5-step simplified workflow for contributors not using Claude Code
- Clarifies that maintainers run automated review protocols

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Add "Your First Contribution" Walkthrough to CONTRIBUTING.md

**Contracts Implemented:** BC-8

**Files:**
- Modify: `CONTRIBUTING.md` (add section after Quick Start, before Development Workflow)

**Step 1: Add the walkthrough section**

After the Quick Start section (after line 18, before `## Development Workflow`), insert:

```markdown

## Your First Contribution

This walkthrough adds a trivial admission policy — the lightest extension type (~3 files). Follow it step-by-step to learn the patterns, then apply them to your own contribution.

**What we'll build:** A `CountingAdmit` admission policy that admits the first N requests and rejects the rest.

### Step 1: Create a branch

```bash
git checkout -b feature/counting-admit
```

### Step 2: Write the failing test

Add a test to `sim/admission_test.go`:

```go
func TestCountingAdmit_RejectsAfterLimit(t *testing.T) {
	// GIVEN a CountingAdmit policy with limit=2
	policy := &CountingAdmit{Limit: 2}
	req := &Request{ID: "test", InputTokens: make([]int, 3)}
	state := &RouterState{Clock: 0}

	// WHEN 3 requests arrive
	r1, _ := policy.Admit(req, state)
	r2, _ := policy.Admit(req, state)
	r3, reason := policy.Admit(req, state)

	// THEN the first 2 are admitted and the 3rd is rejected
	if !r1 {
		t.Error("first request should be admitted")
	}
	if !r2 {
		t.Error("second request should be admitted")
	}
	if r3 {
		t.Errorf("third request should be rejected, got reason: %s", reason)
	}
}
```

Run: `go test ./sim/... -run TestCountingAdmit -v`
Expected: **FAIL** (type `CountingAdmit` does not exist yet)

### Step 3: Implement the policy

In `sim/admission.go`, add after the existing policies:

```go
// CountingAdmit admits the first Limit requests, then rejects all subsequent ones.
type CountingAdmit struct {
	Limit int
	count int
}

func (c *CountingAdmit) Admit(_ *Request, _ *RouterState) (bool, string) {
	c.count++
	if c.count <= c.Limit {
		return true, ""
	}
	return false, "counting-admit limit exceeded"
}
```

### Step 4: Register in the factory

Two files need changes:

In `sim/bundle.go`, add `"counting-admit"` to the `validAdmissionPolicies` map:

```go
validAdmissionPolicies = map[string]bool{"": true, "always-admit": true, "token-bucket": true, "reject-all": true, "counting-admit": true}
```

In `sim/admission.go`, add a case to the `NewAdmissionPolicy` factory switch:

```go
case "counting-admit":
    return &CountingAdmit{Limit: 100} // hardcoded for tutorial simplicity
```

> **Note:** In a real policy, you would wire the limit through the factory parameters (e.g., `Limit: int(capacity)`) or via `PolicyBundle` YAML config. Hardcoded defaults would fail code review — see how `token-bucket` uses `capacity` and `refillRate`.

### Step 5: Verify tests pass

```bash
go test ./sim/... -run TestCountingAdmit -v   # Your new test
go test ./...                                    # All tests still pass
golangci-lint run ./...                          # No lint issues
```

### Step 6: Commit and open a PR

```bash
git add sim/admission.go sim/admission_test.go sim/bundle.go
git commit -m "feat(sim): add counting-admit admission policy

- Admits first N requests, rejects the rest
- Registered in factory with default limit=100"
git push -u origin feature/counting-admit
gh pr create --title "feat: add counting-admit admission policy" --body "My first BLIS contribution!"
```

**That's it!** You've added a complete, tested, registered policy. Real contributions follow the same pattern — just with more contracts and a formal implementation plan.

> **Important:** This example is for learning only. Do **not** submit this as a real PR — `CountingAdmit` is a toy policy with no practical use. For your actual first contribution, check [open issues](https://github.com/inference-sim/inference-sim/issues) for tasks labeled `good first issue`.

```

**Step 2: Verify no build/test regression**

Run: `go build ./... && go test ./... 2>&1 | tail -3`
Expected: All packages pass (the walkthrough code is in markdown, not executed)

**Step 3: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs(contributing): add Your First Contribution walkthrough (BC-8)

- Complete tutorial: branch → test → implement → register → verify → PR
- Uses CountingAdmit admission policy (lightest extension type)
- Teaches factory registration, BDD testing, and commit patterns

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Reorganize CLAUDE.md Behavioral Instructions

**Contracts Implemented:** BC-9

**Files:**
- Modify: `CLAUDE.md` (move 3 sections under new heading)

**Step 1: Add heading and move sections**

Find the following three sections in CLAUDE.md:
- `### Context Management`
- `### Task Agent Guidelines`
- `### Macro Plan Updates`

These are currently interspersed with architecture reference material under `## Development Guidelines`. Move them to a new section:

After the entire `## Development Guidelines` section ends (after `### CI/CD`), but before `## File Organization`, insert a new top-level section:

```markdown

## Agent Behavioral Instructions

The following instructions are for Claude Code and other AI assistants working on this codebase. Human contributors can skip this section.

### Context Management

[existing content unchanged]

### Task Agent Guidelines

[existing content unchanged]

### Macro Plan Updates

[existing content unchanged]
```

Remove these three sections from their original locations under Development Guidelines.

**Step 2: Verify the CLAUDE.md file organization tree is unchanged**

The File Organization section, invariants, rules table, and all other content must remain in their original locations.

**Step 3: Verify no build/test regression**

Run: `go build ./... && go test ./... 2>&1 | tail -3`
Expected: All packages pass

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude-md): separate agent instructions from reference material (BC-9)

- Move Context Management, Task Agent Guidelines, Macro Plan Updates
  under new 'Agent Behavioral Instructions' heading
- Human contributors can now skip this section
- No content deleted — only reorganized

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Verification |
|----------|------|-----------|--------------|
| BC-1     | Task 1 | Manual | Grep README.md for "CONTRIBUTING.md" |
| BC-2     | Task 2 | Manual | Grep CONTRIBUTING.md for "golangci-lint" install |
| BC-3     | Task 1 | Manual | Grep README.md for "You should see" |
| BC-4     | Task 3 | Manual | Verify ToC anchor links match heading slugs |
| BC-5     | Task 3 | Manual | Verify ToC anchor links match heading slugs |
| BC-6     | Task 4 | Manual | Verify all 20 rules appear in exactly one tier |
| BC-7     | Task 5 | Manual | Verify 5-step workflow is complete and self-contained |
| BC-8     | Task 6 | Manual | Verify walkthrough code matches current codebase APIs |
| BC-9     | Task 7 | Manual | Verify 3 sections moved, no content deleted |
| BC-10    | All   | Diff    | `git diff --stat` shows only additions/moves |
| BC-11    | All   | CI      | `go test ./...` and `golangci-lint run ./...` pass |

No golden dataset changes. No new Go code. No invariant tests needed.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Walkthrough code (Task 6) references stale APIs | Medium | High | Verify against current `sim/admission.go` before writing |
| ToC anchor links break on heading renames | Low | Low | GitHub auto-generates slugs; manually verify top 5 |
| CLAUDE.md reorganization breaks AI assistant behavior | Low | Medium | Only move sections, don't change content. Note: heading hierarchy changes (sections move from under "Development Guidelines" to under "Agent Behavioral Instructions") which could affect tools that parse CLAUDE.md by section path |
| Rules.md tier assignments are subjective | Medium | Low | Use objective criteria: "produces wrong results" = Critical |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates
- [x] No hidden global state impact
- [x] N/A: All new code will pass golangci-lint (no new Go code)
- [x] N/A: Shared test helpers (no new tests)
- [x] CLAUDE.md updated (Task 7 reorganizes it)
- [x] No stale references left in CLAUDE.md
- [x] Documentation DRY: rules.md is a canonical source; CLAUDE.md working copy's compact table is not changed
- [x] Deviation log reviewed — no unresolved deviations
- [x] Each task produces useful documentation (no scaffolding)
- [x] Task dependencies are correctly ordered (Tasks 1-7 are independent; Task 3 adds ToC which must be updated if later tasks add sections, but ToC is added to README/pr-workflow which are modified in Tasks 1/3 respectively)
- [x] All contracts mapped to specific tasks
- [x] N/A: Golden dataset regeneration (no output changes)
- [x] N/A: Construction site audit (no struct changes)
- [x] N/A: Antipattern rules R1-R20 (documentation-only PR)

---

## Appendix: File-Level Implementation Details

### File: `README.md`

**Purpose:** Project README — first thing new users and contributors see.

**Changes:**
1. (Task 1) Replace Contributing section text to link to CONTRIBUTING.md
2. (Task 1) Add "What you should see" annotation after Quick Start command
3. (Task 3) Add Table of Contents after project description

### File: `CONTRIBUTING.md`

**Purpose:** Contributor guide — engineering standards and workflow.

**Changes:**
1. (Task 2) Add golangci-lint install command in Quick Start
2. (Task 5) Add "Human Contributor Quick Path" section after Development Workflow
3. (Task 6) Add "Your First Contribution" walkthrough after Quick Start

**Insertion order matters:** Task 6 inserts after Quick Start (line 18). Task 5 inserts after Development Workflow (line 34, which shifts after Task 6's insertion). Execute in task order.

### File: `docs/standards/rules.md`

**Purpose:** Canonical antipattern rules (R1-R20).

**Changes:**
1. (Task 4) Add Priority Tiers section with 3-tier table after introductory paragraph

### File: `docs/process/pr-workflow.md`

**Purpose:** End-to-end PR development workflow.

**Changes:**
1. (Task 3) Add Table of Contents after status line

### File: `CLAUDE.md`

**Purpose:** AI assistant onboarding and codebase reference.

**Changes:**
1. (Task 7) Move Context Management, Task Agent Guidelines, and Macro Plan Updates under new "Agent Behavioral Instructions" heading

# Template Decision Tree — Implementation Plan

**Goal:** Add issue-filing guidance to CLAUDE.md so AI agents know which of the 5 issue templates to use.
**Source:** [#1072](https://github.com/inference-sim/inference-sim/issues/1072), parent [#1071](https://github.com/inference-sim/inference-sim/issues/1071)
**Closes:** `Fixes #1072`
**PR Size Tier:** Small (1 file modified + this plan, no Go code, no behavioral logic changes)

**Clarifications:**
- Issue #1072 suggests "optionally also in `docs/contributing/`" — we place it only in CLAUDE.md to avoid duplication. CLAUDE.md is what agents read first; adding it elsewhere would create a working-copy maintenance burden with no added value.

## Behavioral Contracts

BC-1: Decision tree present in CLAUDE.md
- GIVEN an AI agent reads CLAUDE.md
- WHEN it reaches the Agent Behavioral Instructions section
- THEN it finds a subsection titled "Issue Filing" with a numbered decision tree mapping 5 situations to 5 issue templates

BC-2: Each branch links to the correct template
- GIVEN a decision tree entry (e.g., "Porting a feature from an external repo?")
- WHEN the agent follows the arrow
- THEN it reaches the correct template file path (e.g., `.github/ISSUE_TEMPLATE/cross_repo_feature.md`)

BC-3: Agent can file an issue without prior template knowledge
- GIVEN an AI agent that has never seen the repo's issue templates before
- WHEN it reads only the CLAUDE.md Agent Behavioral Instructions section
- THEN it has enough information to: (1) determine which template to use, (2) find the template file, and (3) understand the key requirement of that template (e.g., "requires GitHub permalinks")

## Tasks

### Task 1: Add "Issue Filing" subsection to CLAUDE.md Agent Behavioral Instructions (BC-1, BC-2, BC-3)

**Files:** modify `CLAUDE.md`

**What to do:**

Add a new subsection `### Issue Filing` inside the `## Agent Behavioral Instructions` section of CLAUDE.md, right after the existing `### Macro Plan Updates` subsection (line 231) and before `## Speckit Feature-Development Toolkit` (line 234).

The subsection must contain:

1. A one-sentence intro: "When filing a GitHub issue, pick the template that matches your situation:"

2. A numbered decision tree with exactly 5 entries:
   - Found a bug or wrong simulation result? → `.github/ISSUE_TEMPLATE/bug_report.md`
   - Porting a feature from an external repo (llmd, gaie, vllm, sglang)? → `.github/ISSUE_TEMPLATE/cross_repo_feature.md` — requires GitHub permalinks to source code
   - Proposing a new BLIS-native capability? → `.github/ISSUE_TEMPLATE/feature_request.md`
   - Testing a hypothesis or running an experiment? → `.github/ISSUE_TEMPLATE/hypothesis.md`
   - Fixing an antipattern, hardening, or refactoring? → `.github/ISSUE_TEMPLATE/custom.md`

3. A one-line reminder: "Every issue must have at least one label. Use `gh issue create --template <filename>` to pre-fill the template."

**Style guide:** Match the terse, directive style of the existing Agent Behavioral Instructions subsections (Context Management, Task Agent Guidelines, Macro Plan Updates). No lengthy explanations. Keep it under 15 lines.

**Verify:** Read CLAUDE.md and confirm the new subsection is between "Macro Plan Updates" and "Speckit Feature-Development Toolkit". Confirm all 5 templates are listed with correct file paths. Confirm the paths match the actual files in `.github/ISSUE_TEMPLATE/`.

**Lint:** N/A (no Go code)

**Commit:** `docs(claude): add issue template decision tree for agents (BC-1, BC-2, BC-3)`

## Sanity Checklist

- [ ] CLAUDE.md is modified — check source-of-truth map for working copies. The Agent Behavioral Instructions section is original CLAUDE.md content, not a working copy of any canonical source. No other files need updating.
- [ ] All 5 template file paths match actual files in `.github/ISSUE_TEMPLATE/`: `bug_report.md`, `cross_repo_feature.md`, `feature_request.md`, `hypothesis.md`, `custom.md`
- [ ] No new antipattern rules apply (no Go code)
- [ ] mkdocs.yml does not need updating (CLAUDE.md is not part of the docs site)
- [ ] No feature creep — this adds only the decision tree, not broader workflow changes
- [ ] The cross-repo entry mentions "requires GitHub permalinks" — this is the key differentiator from feature_request and the main reason the template exists

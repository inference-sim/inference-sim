# Cross-Repo Feature Issue Template — Implementation Plan

**Goal:** Add a GitHub issue template that requires code proofs (GitHub permalinks) when porting features from external repos into BLIS.
**Source:** [#1053](https://github.com/inference-sim/inference-sim/issues/1053), [#1071](https://github.com/inference-sim/inference-sim/issues/1071)
**Closes:** `Fixes #1053`
**PR Size Tier:** Small (1 new template file, no Go code, no behavioral logic changes)

## Behavioral Contracts

BC-1: Cross-repo template available in GitHub issue picker
- GIVEN a contributor opens "New Issue" on GitHub
- WHEN they view the template picker
- THEN they see a "Cross-repo feature" option with description "Port a feature or behavior from an external platform (llmd, gaie, vllm, sglang) into BLIS"

BC-2: Template requires GitHub permalinks as code proofs
- GIVEN a contributor selects the cross-repo feature template
- WHEN they view the template body
- THEN it contains a mandatory "Reference implementation" section that asks for GitHub permalinks to the exact source code lines

BC-3: Template warns AI agents to verify their references
- GIVEN an AI agent reads the template
- WHEN it processes the HTML comment at the top
- THEN it finds explicit instructions to: (1) provide GitHub permalinks, not paraphrased code, (2) verify each permalink resolves, and (3) ask itself "Are you absolutely sure these references are correct and complete?"

## Tasks

### Task 1: Create cross-repo feature issue template (BC-1, BC-2, BC-3)

**Files:** create `.github/ISSUE_TEMPLATE/cross_repo_feature.md`

**What to do:**

Create the file `.github/ISSUE_TEMPLATE/cross_repo_feature.md` with this exact content:

```yaml
---
name: Cross-repo feature
about: Port a feature or behavior from an external platform (llmd, gaie, vllm, sglang) into BLIS
title: ''
labels: 'enhancement, cross-repo'
assignees: ''

---
```

Followed by the template body (in plain markdown, not inside a code fence). The template body must contain these sections in order:

1. **HTML comment block at the top** — a STOP warning explaining:
   - This template is for features replicating behavior from an external system
   - MANDATORY: must have access to actual source code (no prose descriptions)
   - AI AGENTS: must provide GitHub permalinks, not paraphrased code. Must verify each permalink resolves. Must ask: "Are you absolutely sure these references are correct and complete?" before submitting.

2. **"What external behavior are we replicating?"** — asks for system, component, and specific behavior name.

3. **"Reference implementation (MANDATORY)"** — explains GitHub permalinks are the preferred format. Provides placeholder permalink examples. Includes HTML comment about manual attribution fallback for non-GitHub repos.

4. **"Key behaviors to preserve"** — checklist linking behaviors to specific permalink lines.

5. **"Known intentional deviations"** — where BLIS should deliberately differ from the reference.

6. **"Target parity version"** — which version/commit of the external system is targeted.

7. **"Which components are affected?"** — same checklist as `feature_request.md` (Core sim, Cluster, Workload, KV cache, Trace, CLI, New package).

8. **"Extension friction check"** — same as `feature_request.md` (files to change, new interface needed, invariant impact).

9. **"Relationship to existing work"** — same as `feature_request.md`.

**Style guide:** Match the formatting style of the existing templates (`feature_request.md`, `bug_report.md`). Use `**bold**` for section headers. Use `- [ ]` for checklists. Keep it concise.

**Verify:** Open the raw file and confirm YAML frontmatter is valid (name, about, title, labels, assignees fields present). Confirm all 9 sections listed above are present.

**Commit:** `docs: add cross-repo feature issue template (BC-1, BC-2, BC-3)`

## Sanity Checklist

- [ ] No Go code changes — this is a docs/template-only PR
- [ ] No source-of-truth map entries need updating (issue templates are not tracked in the map)
- [ ] CLAUDE.md does not reference issue templates in its file organization — no update needed
- [ ] `project-structure.md` does not list `.github/ISSUE_TEMPLATE/` contents — no update needed
- [ ] Template YAML frontmatter matches the format of existing templates (`feature_request.md`, `bug_report.md`)
- [ ] Template labels field includes both `enhancement` and `cross-repo`
- [ ] The `cross-repo` label may need to be created in the GitHub repo if it doesn't exist
- [ ] No new antipattern rules apply (R1-R23 are for Go code)
- [ ] mkdocs.yml does not need updating (issue templates are not part of the docs site)

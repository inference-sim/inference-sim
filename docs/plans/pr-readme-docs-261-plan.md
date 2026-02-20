# docs(readme): add missing --kv-transfer-base-latency flag — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

| Field | Value |
|-------|-------|
| **Goal** | Add `--kv-transfer-base-latency` to README's Tiered KV Cache section and CLI Reference table |
| **Architecture** | Documentation-only change — no code, no tests |
| **Source Reference** | GitHub issue #261 |
| **Closes:** | Fixes #261 |

---

## Part 1: Design Validation

### A) Executive Summary

Issue #261 reports that the README is missing the `--kv-transfer-base-latency` CLI flag, which was added by PR #253. The flag appears correctly in `cmd/root.go` (line 592) and CLAUDE.md, but was never added to README.md. This PR adds the flag to two locations: the Tiered KV Cache usage section's flags table and the CLI Reference table at the bottom of the file.

### B) Behavioral Contracts

**BC-1: Tiered KV Cache flags table completeness**
- GIVEN a reader viewing the "Tiered KV Cache" section of README.md
- WHEN they look at the flags table
- THEN `--kv-transfer-base-latency` appears with default `0` and description matching `cmd/root.go`

**BC-2: CLI Reference completeness**
- GIVEN a reader viewing the "CLI Reference > vLLM Server Parameters" section
- WHEN they look at the flags table
- THEN `--kv-transfer-base-latency` appears with default `0` and description matching `cmd/root.go`

### C) Component Interaction

Single file change: `README.md`. No cross-component interaction.

### D) Deviation Log

None — issue #261 is implemented exactly as specified.

### E) Review Guide

Verify:
1. Both tables include the new row
2. Default value matches `cmd/root.go` (0)
3. Description is accurate and concise
4. Table formatting is consistent with adjacent rows

---

## Part 2: Implementation

### F) Implementation Overview

Two insertions into existing markdown tables in `README.md`:
1. Row in the Tiered KV Cache flags table (after `--kv-transfer-bandwidth`)
2. Row in the CLI Reference vLLM Server Parameters table (after `--kv-transfer-bandwidth`)

### G) Task Breakdown

#### Task 1: Add `--kv-transfer-base-latency` to both README tables

**Step 1: Edit Tiered KV Cache flags table**

In `README.md`, after the `--kv-transfer-bandwidth` row (line ~266), add:

```markdown
| `--kv-transfer-base-latency` | 0 | Fixed per-transfer latency in ticks (0 = no fixed cost) |
```

**Step 2: Edit CLI Reference vLLM Server Parameters table**

In `README.md`, after the `--kv-transfer-bandwidth` row (line ~731), add:

```markdown
| `--kv-transfer-base-latency` | 0 | Fixed per-transfer latency in ticks for CPU↔GPU KV transfers (0 = no fixed cost) |
```

**Step 3: Verify**

```bash
grep "kv-transfer-base-latency" README.md
```

Expected: 2 matches (one per table).

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): add missing --kv-transfer-base-latency flag (#261)

Add --kv-transfer-base-latency to Tiered KV Cache flags table and
CLI Reference vLLM Server Parameters table. The flag was added in
PR #253 but omitted from README.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### H) Test Strategy

Documentation-only change — no automated tests. Verification is visual:
- `grep` confirms flag appears in 2 locations
- Table formatting consistent with surrounding rows

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Table formatting broken | Low | Low | Visual inspection of markdown |

---

## Part 3: Sanity Checklist

- [x] No code changes — docs only
- [x] Flag default matches `cmd/root.go` (0)
- [x] Description matches flag help text
- [x] Both insertion points identified with exact surrounding context
- [x] No dead code or scaffolding

---

## Appendix: File-Level Details

| File | Action | What Changes |
|------|--------|-------------|
| `README.md:266` | Insert row | Add `--kv-transfer-base-latency` to Tiered KV Cache table |
| `README.md:731` | Insert row | Add `--kv-transfer-base-latency` to CLI Reference table |

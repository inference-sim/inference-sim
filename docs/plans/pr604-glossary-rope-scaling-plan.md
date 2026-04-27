# Plan: Add 12 Missing Glossary Entries

**Goal:** Add 12 standalone glossary entries to `docs/concepts/glossary.md` identified as gaps in issue #604 comment.
**Source:** Issue #604 and gap audit comment posted 2026-04-27.
**Closes:** #604
**Tier:** Small (docs-only, single file)

---

## Behavioral Contracts

**BC-1 — Twelve new entries appear at correct alphabetical positions**
- GIVEN the glossary is sorted alphabetically (case-insensitive)
- WHEN a user browses the glossary
- THEN the 12 new entries appear in these positions:
  - Cohort: between Chunked Prefill and Continuous Batching
  - GAIE: between Fitness Score and Gateway Queue
  - Gateway Queue: between GAIE and HOL Blocking
  - HOL Blocking: between Gateway Queue and Horizon
  - llm-d: between Latency Model and MaxModelLen
  - MFU: between MaxModelLen and MoE
  - MoE: between MFU and Observe/Replay/Calibrate Pipeline
  - rope_scaling: between Roofline Model and Routing Policy
  - Scheduling Delay: between Routing Snapshot and Scorer
  - SLO Class: between Seed and Step
  - Tensor Parallelism (TP): between Step and Tick
  - TPOT: between Tiered KV Cache and TraceV2

**BC-2 — rope_scaling entry is accurate**
- Types that apply the factor: `linear`, `dynamic`, `yarn`, `default`, `mrope`
- Types that are excluded: `su`, `longrope`, `llama3`
- `yarn` special case: uses `original_max_position_embeddings` as base when present
- `gemma3` special case: skips `rope_scaling` entirely
- Links to MaxModelLen, latency-models guide, configuration reference

**BC-3 — SLO Class entry lists all five tiers with correct priorities**
- critical=4, standard=3, batch=−1, sheddable=−2, background=−3
- Explains sheddability (priority < 0 = sheddable)
- Links to workloads guide and configuration reference

**BC-4 — All 12 entries cross-link to existing authoritative sections**
- No links to non-existent anchors
- Anchors verified against actual section headings in target files

**BC-5 — MaxModelLen entry cross-links rope_scaling**
- GIVEN the MaxModelLen entry contains inline `rope_scaling`
- THEN the backtick reference becomes an anchor link to `#rope_scaling`

---

## Tasks

### Task 1 — Add all 12 entries to glossary.md and update MaxModelLen cross-reference

**Verification (manual):** After write, confirm:
1. All 12 headings present: `grep "^### " docs/concepts/glossary.md`
2. Alphabetical order preserved (verify the 12 insertion points listed in BC-1)
3. MaxModelLen entry contains `[#rope_scaling]` link anchor
4. All cross-link anchors are valid (spot-check 3 anchors in target files)

**Implementation:** Write the complete updated `docs/concepts/glossary.md` with all 12 entries inserted in the correct alphabetical positions.

**Commit:** `docs(glossary): add 12 missing glossary entries (rope_scaling, SLO Class, TP, MoE, etc.), closes #604`

---

## Sanity Checklist

- [ ] 12 new headings present in file
- [ ] Alphabetical order correct for all 12 insertions
- [ ] rope_scaling content matches cmd/root.go:186-191 and latency-models.md:87 verbatim
- [ ] SLO class priorities match configuration.md SLO Tier Priorities table
- [ ] MoE interleaved vs uniform distinction present
- [ ] All cross-link anchors verified against target files
- [ ] MaxModelLen cross-link to rope_scaling present
- [ ] No process/workflow docs changed

---

## Deviation Log

**Scope expansion from source:** Issue #604 requested only `rope_scaling`. Based on the gap audit posted in the same issue thread, this plan adds all 12 identified Tier 1+2 gaps in one PR to avoid 12 separate single-entry PRs for related documentation work. No deviations from the individual entry content proposed in the audit comment.

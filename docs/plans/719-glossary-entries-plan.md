# 719: Add Glossary Entries for Observe/Replay/Calibrate Pipeline

**Goal:** Add 5 glossary entries to `docs/concepts/glossary.md` for concepts introduced by the observe/replay/calibrate pipeline. Also fix pre-existing alphabetical ordering issue (Work-Conserving vs Workload Specification).

**Source:** [GitHub Issue #719](https://github.com/inference-sim/inference-sim/issues/719), child of #715.

**Closes:** #719

**Tier:** Small (docs-only, 1 file, no behavioral changes)

**No clarifications needed** — the source document is unambiguous and complete.

---

## Behavioral Contracts

**BC-1: Alphabetical ordering preserved**
GIVEN the existing glossary has entries in mostly alphabetical order (with one pre-existing violation: "Workload Specification" before "Work-Conserving")
WHEN 5 new entries are added and the pre-existing order is fixed
THEN all entries (existing + new) are in strict alphabetical order

**BC-2: Cross-references to pipeline guide**
GIVEN each new entry describes a pipeline concept
WHEN the entry is rendered
THEN it contains a markdown link to the pipeline guide page (`../guide/observe-replay-calibrate.md` or a section anchor within it)

**BC-3: Terminology consistency**
GIVEN the pipeline guide uses specific terminology (TraceV2, distribution synthesis, warmup requests)
WHEN the glossary defines these terms
THEN the definitions are consistent with the guide's usage (no contradictions)

---

## Tasks

### Task 1: Add 5 glossary entries in alphabetical order and fix existing order

1. **Edit** `docs/concepts/glossary.md` to add these entries at their correct alphabetical positions:
   - **Calibration Report** — between "Block (KV Block)" and "Chunked Prefill" ("Cal" < "Ch")

     > The JSON output of `blis calibrate` comparing real observed latencies against simulator predictions. Contains per-request TTFT and E2E deltas, aggregate error metrics (MAPE, Pearson correlation, percentile comparisons), bias assessment, and a quality rating. See [Observe / Replay / Calibrate](../guide/observe-replay-calibrate.md#blis-calibrate).

   - **Distribution Synthesis** — between "Discrete Event Simulation (DES)" and "E2E (End-to-End Latency)" ("Disc" < "Dist")

     > The `--rate` mode of `blis observe` that generates workload from statistical distributions (prompt/output token counts, arrival rate) instead of a workload spec YAML file. Useful for quick single-client experiments without crafting a full workload specification. See [Observe / Replay / Calibrate: Distribution Synthesis Flags](../guide/observe-replay-calibrate.md#distribution-synthesis-flags).

   - **Observe/Replay/Calibrate Pipeline** — between "MaxModelLen" and "Oracle Knowledge Boundary (INV-9)" ("Ob" < "Or")

     > The end-to-end workflow of `blis observe` → `blis replay` → `blis calibrate` for validating simulator accuracy against real inference servers. Each stage is independently useful: observe collects latency baselines, replay tests simulator behavior on real traces, and calibrate compares results. See [Observe / Replay / Calibrate](../guide/observe-replay-calibrate.md).

   - **TraceV2** — between "Tiered KV Cache" and "TTFT (Time To First Token)" ("Ti" < "Tr" < "TT")

     > The trace format used by the observe/replay/calibrate pipeline, consisting of two files: a header YAML file (recording model, server config, and observation metadata) and a data CSV file (recording per-request timing: arrival time, TTFT, E2E latency, token counts). See [Observe / Replay / Calibrate](../guide/observe-replay-calibrate.md).

   - **Warmup Requests** — between "TTFT (Time To First Token)" and "Work-Conserving" ("War" < "Wor")

     > Initial requests dispatched by `blis observe` that are excluded from trace output (`--warmup-requests N`). Warmup requests allow the server to populate its KV cache and reach steady-state scheduling before measurement begins, avoiding cold-start artifacts in the recorded trace. See [Observe / Replay / Calibrate: blis observe](../guide/observe-replay-calibrate.md#blis-observe).

2. **Fix pre-existing order:** Swap "Work-Conserving" (currently line 151) before "Workload Specification" (currently line 147). "Work-C" < "Workl" in lexicographic order.

3. Each entry follows the existing format: `### Term` heading, 1–3 sentence description, `See [link text](relative-path).` cross-reference.

4. **Verify** alphabetical order by scanning all `###` headings in sequence.

5. **Verify** MkDocs build (or link check) — no broken links.

6. **Commit.**

---

## Sanity Checklist

- [ ] All 5 entries present
- [ ] Pre-existing order fixed (Work-Conserving before Workload Specification)
- [ ] Alphabetical order preserved across entire file
- [ ] Each entry cross-references the pipeline guide
- [ ] No broken markdown links
- [ ] Definitions consistent with `docs/guide/observe-replay-calibrate.md`
- [ ] No unrelated changes

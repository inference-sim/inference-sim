# Documentation: CLAUDE.md gaps + architecture pipeline diagram

**Goal:** Close documentation gaps in CLAUDE.md's Build and Run Commands section and add an observe/replay/calibrate data flow diagram to the architecture page.
**Source:** [#720](https://github.com/inference-sim/inference-sim/issues/720), [#721](https://github.com/inference-sim/inference-sim/issues/721) (parent: #715).
**Closes:** Fixes #720, fixes #721.

## Behavioral Contracts

BC-1: CLAUDE.md observe section covers key flags
- GIVEN the CLAUDE.md Build and Run Commands section
- WHEN a contributor reads the `blis observe` examples
- THEN examples demonstrate `--api-key`, `--max-concurrency`, `--warmup-requests`, `--no-streaming`, `--unconstrained-output`, `--rate` mode with distribution flags (`--prompt-tokens`, `--output-tokens`), and `--prefix-tokens`

BC-2: CLAUDE.md convert section covers inference-perf
- GIVEN the CLAUDE.md Build and Run Commands section
- WHEN a contributor reads the `blis convert` examples
- THEN a `convert inference-perf` example is present alongside the existing `preset` and `servegen` examples

BC-3: CLAUDE.md run section covers --trace-output
- GIVEN the CLAUDE.md Build and Run Commands section
- WHEN a contributor reads the `blis run` examples
- THEN a `--trace-output` flag usage example is present

BC-4: Architecture page documents observe/replay/calibrate pipeline
- GIVEN `docs/concepts/architecture.md`
- WHEN a reader looks for the real-server observation data flow
- THEN a section with a mermaid data flow diagram shows observe → replay → calibrate, noting that only replay engages the DES

## Tasks

### Task 1: Update CLAUDE.md Build and Run Commands (BC-1, BC-2, BC-3)

**Files:** modify `CLAUDE.md`

**Impl:**
In the `## Build and Run Commands` bash block:
1. Add a second `blis run` example with `--trace-output <prefix>` (note: takes a prefix, not a file path — auto-appends `.yaml`/`.csv`; keep existing minimal example intact)
2. Add a third `blis observe` example showing `--rate` mode with key distribution flags (`--rate`, `--num-requests`, `--prompt-tokens`, `--output-tokens`, `--prefix-tokens`) and optional flags (`--warmup-requests`, `--no-streaming`, `--api-key`, `--max-concurrency`, `--unconstrained-output`)
3. Add `blis convert inference-perf --spec spec.yaml` example after the existing convert examples

**Verify:** Read CLAUDE.md and confirm all flags from BC-1/BC-2/BC-3 appear.
**Lint:** N/A (markdown only)
**Commit:** `docs(claude-md): add missing observe flags, convert inference-perf, trace-output (BC-1, BC-2, BC-3)`

### Task 2: Add observe/replay/calibrate section to architecture.md (BC-4)

**Files:** modify `docs/concepts/architecture.md`

**Impl:**
Add a new `## Observe / Replay / Calibrate Pipeline` section at the end of the file (after `## Online Routing Pipeline Walkthrough`). This is an offline validation workflow, not a DES internal — placing it after all DES content avoids interrupting the cluster simulation flow. Include:
1. Brief prose (3-5 sentences) describing the three stages, clarifying that only Replay engages the DES event loop; Observe is an HTTP client and Calibrate is statistical comparison
2. A mermaid flowchart showing dual inputs: WorkloadSpec + ServerURL → Observe → TraceV2; TraceV2 + `--model` + sim flags → Replay (DES) → SimResult JSON (`--results-path`); TraceV2 + SimResult JSON → Calibrate → Report
3. Cross-reference to the pipeline guide ([Observe / Replay / Calibrate](../guide/observe-replay-calibrate.md)) and the flag reference ([Configuration Reference](../reference/configuration.md))

**Verify:** Read architecture.md and confirm the new section exists with mermaid diagram.
**Lint:** N/A (markdown only)
**Commit:** `docs(architecture): add observe/replay/calibrate data flow diagram (BC-4)`

## Sanity Checklist

- [ ] R4 (canonical constructors): N/A — no code changes
- [ ] R10 (strict YAML): N/A — no code changes
- [ ] Source-of-truth map: CLAUDE.md is itself a working copy — changes here don't require syncing elsewhere. Architecture.md is not in the source-of-truth map.
- [ ] No behavioral code changes — docs only
- [ ] Verify flag names against `cmd/observe_cmd.go`, `cmd/root.go`, `cmd/convert.go` before writing examples

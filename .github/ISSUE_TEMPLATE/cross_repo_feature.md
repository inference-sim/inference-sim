---
name: Cross-repo feature
about: Port a feature or behavior from an external platform (llmd, gaie, vllm, sglang) into the simulator
title: ''
labels: 'enhancement, cross-repo'
assignees: ''

---

<!-- ⚠️ STOP — READ THIS BEFORE OPENING THIS ISSUE ⚠️

This template is for features that replicate behavior from an external system
(llmd, gaie, vllm, sglang, etc.) into the BLIS simulator.

MANDATORY REQUIREMENT: You must have access to the actual source code of the
reference implementation. Prose descriptions, documentation summaries, or
second-hand explanations are NOT sufficient. If you cannot provide the real
code, do not open this issue — ask for access first.

AI AGENTS: If you are an AI agent opening this issue, you MUST read the
actual source files from the reference repo and provide GitHub permalinks
to the relevant code below. Do NOT paraphrase, summarize, or describe code
from memory. If you do not have access to the source repo, STOP and ask
the user to provide the code or grant access before proceeding.

After assembling your code proofs, verify each permalink resolves to the
code you cited. Ask yourself: "Are you absolutely sure these references
are correct and complete?" Re-examine before submitting. -->

**What external behavior are we replicating?**
Name the system, component, and specific behavior (e.g., "llmd gateway GAIE admission saturation formula").

**Reference implementation (MANDATORY)**

Provide GitHub permalinks to the actual source code that defines the behavior being ported. Include ALL relevant code paths — main logic, edge cases, defaults, error handling.

GitHub permalinks are the preferred format — they encode repo, commit, file, and lines in a single verifiable URL, and GitHub auto-renders them as code blocks.

<!-- Paste as many permalinks as needed. Each must point to real code, not pseudocode. -->

Permalink 1:
https://github.com/org/repo/blob/abc1234/path/to/file.go#L42-L68

Permalink 2 (if applicable):
https://github.com/org/repo/blob/abc1234/path/to/file.py#L100-L120

<!-- For non-GitHub repos, fall back to manual attribution:
Source: <repo>/<file>@<commit> (lines X-Y)
followed by a fenced code block containing the actual source code. -->

**Key behaviors to preserve**
List the specific behaviors from the reference code that BLIS must match exactly:
- [ ] Behavior 1 (reference: permalink 1)
- [ ] Behavior 2 (reference: permalink 2)

**Known intentional deviations**
List any places where BLIS should deliberately differ from the reference, and why:
- Deviation 1: [what differs] — [why]

**Target parity version**
What version/commit of the external system are we targeting? (e.g., "llmd v0.8.2, commit abc1234")

**Which components are affected?**
- [ ] Core simulator (`sim/`)
- [ ] Cluster simulation (`sim/cluster/`)
- [ ] Workload generation (`sim/workload/`)
- [ ] KV cache (`sim/kv/`)
- [ ] Decision tracing (`sim/trace/`)
- [ ] CLI (`cmd/`)
- [ ] New package needed

**Extension friction check**
- How many files would need to change to add this? (Estimate)
- Does this require a new interface, or can it extend an existing one?
- Does this affect any invariants (conservation, causality, determinism)?

**Relationship to existing work**
Does this relate to any open issues, the macro plan, or a design document?

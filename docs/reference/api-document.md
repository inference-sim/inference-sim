# Declarative API Document (`llm-d-perf-simulator/v1`)

BLIS is increasingly driven from inside other tools rather than by a human at a prompt, so it
is growing a single, versioned, self-describing document as its canonical input *and* output —
one format across every verb, replacing today's ~117 flags plus heterogeneous config files and
mixed JSON/text stdout. The design lives in
[Discussion #1853](https://github.com/inference-sim/inference-sim/discussions/1853); the work is
tracked by [epic #1855](https://github.com/inference-sim/inference-sim/issues/1855).

!!! warning "Not yet wired into any command"
    This page documents the **envelope only** — the part delivered by API-1
    ([#1856](https://github.com/inference-sim/inference-sim/issues/1856)). No `blis` command
    reads or writes a document yet, no flag changed, and `blis run` output is unchanged. The
    per-verb `spec` sections and the `result` body arrive with API-2 onward, at which point
    this page grows with them.

## The envelope

Every document carries an `apiVersion` and a `kind`. An **input** document carries a `spec`; an
**output** document — a `…Result` kind — carries a `result`:

```yaml
apiVersion: llm-d-perf-simulator/v1
kind: Run
spec: {}
```

```yaml
apiVersion: llm-d-perf-simulator/v1
kind: RunResult
result: {}
```

JSON is accepted wherever YAML is: YAML is a superset, one parser reads both, and the same
document in either encoding gets the same verdict.

### Fields

| Field | Required | Meaning |
|-------|----------|---------|
| `apiVersion` | always | Exactly `llm-d-perf-simulator/v1`. One value is defined; anything else is refused. |
| `kind` | always | One of the ten kinds below. |
| `spec` | input kinds | The input body. An input document must carry it and must **not** carry a `result`. |
| `result` | output kinds | The output body. An output document must carry it and must **not** carry a `spec`. |

An unknown top-level key is refused rather than ignored, so a typo (`specs:`) is an error and
not a silently-empty document.

### Kinds

One input kind per CLI verb, each paired with its `…Result` output form, so a reader can tell
which verb produced a result document from its `kind` alone.

| Verb | Input kind | Output kind |
|------|------------|-------------|
| `blis run` | `Run` | `RunResult` |
| `blis replay` | `Replay` | `ReplayResult` |
| `blis observe` | `Observe` | `ObserveResult` |
| `blis calibrate` | `Calibrate` | `CalibrateResult` |
| `blis convert` | `Convert` | `ConvertResult` |

### Bodies are opaque at this stage

`spec` and `result` are **open objects** today: any contents validate. That is deliberate —
constraining them now would freeze fields the later PRs own. Each later PR replaces one section
with typed, strictly-parsed fields:

- `spec` sections (`deployment`, `latency`, `admission`, …) — API-3 and API-4
- `result` body (`summary`, `conservation`, `kv`, …) — API-2
- the remaining verbs — API-5 through API-8

## The JSON Schema

The published contract is [`api/schema/llm-d-perf-simulator-v1.json`](https://github.com/inference-sim/inference-sim/blob/main/api/schema/llm-d-perf-simulator-v1.json),
a JSON Schema 2020-12 document that external tools can validate against directly.

It is **generated from the Go types** in `api/` — never hand-edited. After changing those
types, regenerate it:

```bash
go generate ./api/...
```

A test fails if the committed schema differs from the one the types derive, so the published
contract cannot drift from the code that implements it. The example documents in
`api/examples/` (one minimal envelope per kind) are validated against the committed schema in
the same test run.

Both gates reach CI through `api_schema_gate_test.go` in the repository root, because
`.github/workflows/ci.yml` lists its test packages explicitly and does not yet list
`./api/...`; that bridge also runs the api package's own suite as a subprocess. Two caveats
until [#1866](https://github.com/inference-sim/inference-sim/issues/1866) adds the matrix
entry: the workflow does not trigger on pull requests targeting the `api` branch that epic
[#1855](https://github.com/inference-sim/inference-sim/issues/1855) uses, so those PRs are
exercised by a manual `workflow_dispatch` run rather than automatically; and `go test ./...`
locally is the only way to run the api package directly.

Validating from Go uses the same schema the published file holds:

```go
import "github.com/inference-sim/inference-sim/api"

if err := api.ValidateDocument(data); err != nil { /* every problem, named */ }
```

`api.Document` is the Go form of the envelope, and `api.Document.Validate` applies the same
rules as the schema — a test holds the two to the same verdict on every envelope shape that
has a `Document` form: each valid and invalid combination of `apiVersion`, `kind`, `spec` and
`result`. A schema that disagreed with the implementation would be worse than no schema.

The schema checker is additionally tested on raw documents the Go types cannot express — a
malformed scalar, a non-string object key, an unknown top-level key. Those cases have no
`Document` form, so there is no Go verdict to compare them against, and they are checked
against the schema alone.

## Invariants

API-1 changes no invariant. Two are reframed by later PRs, and are stated in
[the invariant registry](../contributing/standards/invariants.md) when they land:

- **INV-6 (determinism)** — the result document becomes byte-deterministic given
  (resolved spec + catalog revision + seed), excluding a quarantined `result.runtime` block
  (API-2).
- **INV-13 (run/replay parity)** — feeding a `RunResult`'s embedded input back to `replay`
  yields a byte-identical result document, minus `result.runtime` (API-5).

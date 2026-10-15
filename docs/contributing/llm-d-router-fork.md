# The llm-d Router Fork

BLIS routes with the real llm-d router (see [Routing](../guide/routing.md)). It imports the
router from the [inference-sim fork of llm-d-router](https://github.com/inference-sim/llm-d-router),
branch `llm-d-r`, through a `replace` directive in `go.mod`:

```
replace github.com/llm-d/llm-d-router => github.com/inference-sim/llm-d-router <pseudo-version>
```

While a fork change is under review, the `replace` names the PR's head in its author's fork
(`github.com/<author>/llm-d-router <pseudo-version>`), because a commit that is only on a pull
request is not fetchable from `inference-sim/llm-d-router` directly. Once the fork PR merges
with a merge commit (which keeps its commits' hashes), the `replace` moves to
`inference-sim/llm-d-router` at the same hash.

This page is for whoever maintains that fork and the BLIS side that hosts it.

## What the fork changes

The fork changes **how the router is hosted, never what it decides**. Every change is a seam
whose default (the `Real` runtime) reproduces upstream behavior exactly; the router's
decisions under a hosted runtime are the same decisions over the same inputs.

| Seam | Where | What it does |
|---|---|---|
| `pkg/common/hosting` (new) | `Runtime` interface, `Real` runtime, context `Into`/`From`, `Transport` | The router reads time, starts background work, sets deadlines, blocks, draws randomness and makes outbound HTTP calls through the `Runtime` on its context. |
| Clock and deadlines | director, candidates cache, plugin state, data-layer extractor, utilization detector, request evictor, session affinity, flow control (controller deadlines, processor and registry clocks) | `time.Now`, `time.Since`, `context.WithTimeout`/`WithDeadlineCause` on decision paths go through the runtime. Observability-only timing (metric latencies, log rate limits) is left alone and recorded in the lint baseline. |
| Goroutines and loops | data producers, response-body queue, collectors, cleanup loops, flow-control processor and sweep, token-producer warmup | `go` statements become `rt.Go`; ticker loops use `rt.Clock().NewTicker` and call `rt.Park` before blocking; the response-body queue runs inline when hosted; the flow-control worker pool runs sequentially when hosted. |
| Blocking waits | flow-control `awaitFinalization`, processor loop, cleanup sweep | `rt.Park(ready)` before each blocking `select`, so a hosted runtime knows when the goroutine waits and what it waits for. |
| Randomness | picker random source, model-rewrite split, probabilistic admitter | Per-runtime named streams (`rt.Rand`); the max-score picker's round-robin counter is per runtime (`rt.Local`). |
| Map order | scheduler profile order, picker candidate order, datastore pod list, flow-control queue iteration | Under a hosted runtime: sorted, and where upstream's order is effectively random tie-breaking, a seeded shuffle of the sorted order. Under `Real`: unchanged. |
| Outbound HTTP | metrics data source, vLLM render backend | `hosting.Transport`: a request whose context carries a hosted runtime is served by the runtime. |
| Embedding | `cmd/epp/runner/embed.go` (new), `assembleDirector` (extracted) | `runner.Embed` builds the Director through the runner's own file-discovery assembly path, so config semantics (defaults, feature gates, DAG ordering) are the runner's by construction. `EmbedOptions.AfterRegister` runs before any plugin is instantiated; BLIS uses it to replace the factory of every unsupported plugin type with one that refuses it, so an unsupported plugin's constructor never runs. |
| Guard | `hack/hosting-lint` (new), `.github/workflows/hosting-lint.yml` (new) | The lint and the patch budget below. The lint is type-aware (go/packages, go/types) and records each accepted use individually (file, function, kind, line text), so a new use cannot replace a removed one. Besides clock, timer, deadline, goroutine and random-draw uses it tracks `sync.WaitGroup.Go`/`errgroup.Group.Go`, `clock.RealClock`, library timers (`ttlcache`), map and `sync.Map` iteration, outbound HTTP and the process-global `PickerRand`. |

Everything simulator-specific lives in BLIS (`sim/llmd`): the discrete-event runtime, the
coroutine scheduler, the simulated model servers' HTTP surface, request translation and
outcome mapping.

## The hosted runtime in BLIS

`sim/llmd/runtime.go` implements `hosting.Runtime` on the cluster's event loop:

- `Now()` is a fixed epoch (2000-01-01 UTC) plus the simulation clock; time does not pass
  inside the router.
- `Every`, tickers, timers and deadlines are cluster events. Background events are *daemon*
  events: they never keep a simulation alive on their own.
- `Go` starts a coroutine and runs it until it finishes or parks (`sim/llmd/coroutine.go`).
  Exactly one of the simulation goroutine and the router's coroutines runs at a time; a parked
  coroutine resumes, in spawn order, once its `ready` predicate holds. A coroutine that blocks
  on something it did not `Park` for trips a watchdog with an explanation instead of hanging.
- `WithTimeout` never fires (router work takes zero simulated time); `WithDeadlineCause` fires
  as an event at its deadline.
- `Rand(stream)` streams are seeded from the run seed and the stream name.
- `RoundTrip` serves `/metrics` (each engine's published vLLM series) and
  `/v1/completions/render` (the prompt's token IDs).

## Rebasing onto a new llm-d release

The fork is a permanent patch set on llm-d-router release tags; nothing is upstreamed.

1. Fetch upstream and rebase `llm-d-r` onto the release tag:
   `git fetch upstream --tags && git rebase --onto vX.Y.Z <old-base> llm-d-r`.
2. Resolve conflicts. Most patches are a few lines per file; most of the seam is in new files
   that do not conflict.
3. Run `go run ./hack/hosting-lint`. A failure names each file where the release added a
   direct clock read, timer, deadline, goroutine or random draw. Route each one through the
   runtime, or, if it cannot affect a routing decision (a metric, a log), annotate the line
   `//blis:wallclock-ok <reason>` or regenerate the baseline (`-update`) on review.
4. Update `upstream_base` in `hack/hosting-lint/budget.txt` and run
   `go run ./hack/hosting-lint -budget`.
5. Run the fork's tests, push, then in BLIS run `go get` on the new pseudo-version (the
   `replace` line) and `go test ./sim/llmd/...`. New plugin types in the release are refused by
   BLIS until they are audited and added to `sim/llmd/support.go`'s supported list.

## Keeping the rebase cheap

- **The patch budget.** `hack/hosting-lint/budget.txt` caps the lines the fork may change in
  files it shares with upstream (new files are free). CI fails when a change exceeds it, so a
  growing rebase cost is visible on the change that causes it. Prefer new files, and wrappers
  in new files, over edits to upstream code.
- **The lint.** It is the main defense against a release silently reintroducing wall-clock or
  concurrency on a router path.
- **The determinism tests in BLIS.** `sim/llmd/determinism_test.go` runs the same simulations
  in separate processes at different `GOMAXPROCS`; a map-order or scheduling dependence the
  lint cannot see shows up there.

## Supporting another plugin

A plugin type runs in BLIS only once it is in `sim/llmd/support.go`'s supported list (`llmd.IsSupported`). To add
one: read every clock read, goroutine, blocking wait, random draw and map iteration it makes,
convert the decision-relevant ones in the fork, check that every signal it consumes is produced
by the simulated engines, and add a test in `sim/llmd`. Until then it stays refused with the
reason in the unsupported list (`llmd.UnsupportedReason`).

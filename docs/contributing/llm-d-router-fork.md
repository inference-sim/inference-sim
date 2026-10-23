# The llm-d Router Fork

BLIS routes with the real llm-d router (see [Routing](../guide/routing.md)). It imports the
router from the [inference-sim fork of llm-d-router](https://github.com/inference-sim/llm-d-router),
branch `llm-d-r`, through a `replace` directive in `go.mod`:

```
replace github.com/llm-d/llm-d-router => github.com/inference-sim/llm-d-router vX.Y.Z-blis.N
```

This is the one `replace` that merges into BLIS: the fork keeps upstream's module path so that
upstream releases merge into it, so a `replace` is the only way to substitute it.

While a fork change is under review, the BLIS pull request that needs it names the fork PR's head
in its author's fork (`github.com/<author>/llm-d-router <pseudo-version>`), because a commit that
is only on a pull request is not fetchable from `inference-sim/llm-d-router` directly. Such a
`replace`, and any review-only `replace` of a blis-* module it needs, is marked TEMPORARY and keeps
the BLIS PR in draft. Once the fork PR merges with a merge commit and is tagged, the `replace`
moves to `inference-sim/llm-d-router` at the tag, and the PR can leave draft.

## The contract

BLIS programs against one API: package `embed` of
[blis-schemas](https://github.com/inference-sim/blis-schemas) (introduced in
inference-sim/blis-schemas#99), versioned by blis-schemas' own tags. `embed.Runtime` (with the
optional `embed.RoundTripper`) is what BLIS implements; `embed.Router` and `embed.Flight` are
what the router implements; the rest is plain data (`Request`, `Decision`, `Rejection`,
`KVBatch`, `Requirements`, ...). The fork implements the contract with one constructor,
`pkg/embed.New` (an `embed.Constructor`), which is the only identifier BLIS takes from the
fork. `sim/llmd/imports_test.go` fails on any other llm-d-router import or identifier.

The contract is defined in blis-schemas and changes there, never in the fork alone.

**Tags.** The fork tags `vX.Y.Z-blis.N`: `X.Y.Z` is the upstream release merged into
`llm-d-r`, and `N` counts fork releases on that upstream release, from 1. Tags never move.

**Compatibility.** Between two `-blis` tags with the same upstream major, `pkg/embed.New` and
the blis-schemas `embed` package it implements change only compatibly. The fork's
`hosting-lint.yml` enforces this with `apidiff` on `pkg/embed` against the latest `v*-blis*`
tag, and on blis-schemas `embed` between the version that tag pinned and the one now pinned; it
fails on an incompatible change unless the upstream major changed. A same-major upstream
release, or a fork-only fix, therefore reaches BLIS as a `go.mod`/`go.sum` change only.

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
| KV events | `pkg/kvevents/pool_hosted.go`, `preciseprefixcache/hosted.go` (new) | Under a hosted runtime the precise-prefix producer starts no ZMQ subscriber, indexer goroutine or event-pool workers; the embedder hands it event batches synchronously (`Producer.HostedIngest`, `HostedClearPod`), and speculative entries expire on the runtime's clock. |
| Outbound HTTP | metrics data source, vLLM render backend | `hosting.Transport`: a request whose context carries a hosted runtime is served by the runtime. |
| Embedding | `pkg/embed` (new), over `cmd/epp/runner/embed.go` (new) and `assembleDirector` (extracted) | `pkg/embed.New` implements blis-schemas `embed`: it parses the EPP flags, configuration and manifests, builds the Director through the runner's own file-discovery assembly path (so config semantics are the runner's by construction), adapts `embed.Runtime` to `hosting.Runtime`, and answers in the contract's vocabulary (`Decision`, `Rejection.Kind`, `Requirements`, `KVBatch`). Before any plugin is instantiated it refuses every type the fork has not audited for hosting, or that the host's `Config.Allow` refuses, so an unsupported plugin's constructor never runs. |
| Guard | `hack/hosting-lint` (new), `.github/workflows/hosting-lint.yml` (new) | The lint and the patch budget below. The lint is type-aware (go/packages, go/types) and records each accepted use individually (file, function, kind, line text), so a new use cannot replace a removed one. Besides clock, timer, deadline, goroutine and random-draw uses it tracks `sync.WaitGroup.Go`/`errgroup.Group.Go`, `clock.RealClock`, library timers (`ttlcache`), map and `sync.Map` iteration, outbound HTTP and the process-global `PickerRand`. |

Everything simulator-specific lives in BLIS (`sim/llmd`): the discrete-event runtime, the
coroutine scheduler, pod labels from the simulated placement, prompt canonicalization, the
simulated model servers' HTTP surface, the OpenAI request and response bodies, the mapping of
`Rejection.Kind` onto INV-1 buckets, and BLIS's signal policy (`Config.Allow`).

## The hosted runtime in BLIS

`sim/llmd/runtime.go` implements blis-schemas `embed.Runtime` (and `embed.RoundTripper`) on the
cluster's event loop:

- `Now()` is a fixed epoch (2000-01-01 UTC) plus the simulation clock; time does not pass
  inside the router.
- `Every`, tickers, timers and deadlines are cluster events. Background events are *daemon*
  events: they never keep a simulation alive on their own.
- `Go` starts a coroutine and runs it until it finishes or parks (`sim/llmd/coroutine.go`).
  Exactly one of the simulation goroutine and the router's coroutines runs at a time; a parked
  coroutine resumes, in spawn order, once its `ready` predicate holds. A coroutine that blocks
  on something it did not `Park` for trips a watchdog with an explanation instead of hanging.
- `NewTimer` and `NewTicker` are cluster events delivering on buffered channels
  (`sim/llmd/clock.go`).
- `WithTimeout` never fires (router work takes zero simulated time); `WithDeadlineCause` fires
  as an event at its deadline.
- `Rand(stream)` streams are seeded from the run seed and the stream name.
- `RoundTrip` serves `/metrics` (each engine's published vLLM series) and
  `/v1/completions/render` (the prompt's token IDs).

## Updating to a new llm-d release

The fork is a permanent patch set on llm-d-router release tags; nothing is upstreamed. A
release is **merged** into `llm-d-r`, never rebased: BLIS pins a tag on `llm-d-r`, and a
rebase would rewrite the hashes and orphan it.

1. On a branch from `llm-d-r`, merge the tag: `git fetch upstream --tags && git merge --no-ff vX.Y.Z`.
   Resolve conflicts. Most patches are a few lines per file; most of the seam is in new files
   that do not conflict.
2. Run `go run ./hack/hosting-lint`. A failure names each file where the release added a
   direct clock read, timer, deadline, goroutine or random draw. Route each one through the
   runtime, or, if it cannot affect a routing decision (a metric, a log), annotate the line
   `//blis:wallclock-ok <reason>` or regenerate the baseline (`-update`) on review.
3. Update `upstream_base` in `hack/hosting-lint/budget.txt` to the release and run
   `go run ./hack/hosting-lint -budget`. Fix `pkg/embed` where the release moved an internal it
   uses; its tests, the blis-schemas `embedtest` conformance run and the `apidiff` gate catch
   what the release broke.
4. Run the fork's tests, and BLIS's against the branch (a local `replace`). Open a PR into
   `llm-d-r`, merge it with **a merge commit**, and tag the merge `vX.Y.Z-blis.1`.
5. In BLIS, move the `replace` to the tag and run `go mod tidy` and `go test ./...`. For a
   same-major release this `go.mod`/`go.sum` change is the only BLIS change. New plugin types
   in the release stay refused until the fork audits them and BLIS allows them (below).

## Keeping each update cheap

- **The patch budget.** `hack/hosting-lint/budget.txt` caps the lines the fork may change in
  files it shares with upstream (new files are free). CI fails when a change exceeds it, so a
  growing update cost is visible on the change that causes it. Prefer new files, and wrappers
  in new files, over edits to upstream code.
- **The lint.** It is the main defense against a release silently reintroducing wall-clock or
  concurrency on a router path.
- **The determinism tests in BLIS.** `sim/llmd/determinism_test.go` runs the same simulations
  in separate processes at different `GOMAXPROCS`; a map-order or scheduling dependence the
  lint cannot see shows up there.

## Supporting another plugin

A plugin type runs in BLIS only when the fork has audited it for hosting **and** BLIS's signal
policy allows it (`sim/llmd/support.go`, passed to the router as `embed.Config.Allow`;
`llmd.IsSupported`; `TestSupportedTypesAreHostedByTheFork` holds every allowed type within what the fork can host). To add one: in the fork, read every clock read,
goroutine, blocking wait, random draw and map iteration it makes, convert the
decision-relevant ones and add it to the audited set; in BLIS, check that every signal it
consumes is produced by the simulated engines, allow it, and add a test in `sim/llmd`. Until
then it stays refused with its reason (`llmd.UnsupportedReason`).

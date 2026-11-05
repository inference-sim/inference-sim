# Engine interface

<p class="lede"><code>engine.Engine</code> is the interface an inference-engine model
implements and a discrete-event simulator calls: the counterpart of the
<a href="kernel.md">kernel interface</a> for scheduling rather than pricing. One value is one
engine instance, the state machine behind one server process.</p>

The boundary follows from who owns time. The simulator owns the clock, the event queue and
the decision of when to step. The engine owns every scheduling, KV-cache and connector
decision, and the metrics those decisions produce. An engine reads no clock and schedules no
event: every call carries the simulated instant as a value (`engine.Time`, microseconds), and
where an engine rule makes work wait until a later instant, the engine reports that instant
through `NextWake`.

Nothing in the package names an engine. Configuration reaches an engine as an opaque
`engine.Spec` that the simulator hands over unread, so a second engine implements the
interface in its own repository with no change here or in the simulator.

## Methods

| Method | Does |
|---|---|
| `Add(r Request, now Time) error` | Hands over a request that arrived at `now`. Schedules nothing. |
| `Schedule(now Time) (Batch, error)` | Decides one batch: which requests run, how many tokens each, which are preempted, which KV transfers start. `ErrNoCapacity` when `InFlightLimit` batches are in flight. |
| `StepTime(b Batch) Duration` | Prices the forward pass of a batch that runs requests; at least one microsecond. An `Empty` batch is not priced. |
| `Complete(id BatchID, outcome Outcome, end Time) (Result, error)` | Applies a batch's outcome (tokens sampled, drafts accepted) and returns what became visible: tokens emitted, first tokens, finishes, KV events. |
| `TransferDone(id TransferID, ok bool, now Time) error` | Reports that an asynchronous KV load or store finished, successfully or not. |
| `Finish(id string, why FinishReason, now Time) error` | Ends a request for a reason the engine did not decide (client abort, timeout). A no-op for an unknown or finished request. |
| `NextWake(now Time) (Time, bool)` | The earliest instant after `now` at which state may change with no new input; false if none. |
| `InFlightLimit() int` | How many batches may be in flight at once (for vLLM, `max_concurrent_batches`); at least one. |
| `Metrics(now Time) Metrics` | The engine's exported metrics. |
| `Snapshot() State` | Current state (waiting, running, in-flight, KV capacity), a copy the caller may keep. |

Calls are sequential: a simulator never calls an engine from two goroutines, and an engine
starts none.

### Pipeline stages

An engine whose batches run through more than one pipeline stage, each on its own devices,
also implements `engine.Pipelined`:

| Method | Does |
|---|---|
| `StageTimes(b Batch) []Duration` | `StepTime(b)` split by stage, in pipeline order: one entry per stage, each at least one microsecond, summing exactly to `StepTime(b)`. |

A simulator that honours it starts stage *s* of a batch once stage *s*-1 of that batch has
ended and stage *s* has finished the batch ahead of it, and completes the batch when its last
stage ends. The interface is optional: a simulator that does not look for it times each batch
as one `StepTime` on one device, which serializes the stages. `enginetest.Drive` honours it and
checks its law, including that every batch has the same number of stages.

## Batches

A batch that runs requests gets a `BatchID` greater than every earlier one of the same engine
and stays in flight until `Complete` names it. Any number of batches may be in flight, up to
`InFlightLimit`. They complete in the order they were scheduled (`ErrOutOfOrder` otherwise),
which is how vLLM's engine core consumes its batch queue: it appends each scheduled batch at
one end and takes results from the other. A batch that runs nothing (`Empty`) is not in
flight, but may still start transfers.

## The wake protocol

- The simulator calls `NextWake` after every state-changing call (`Add`, `Schedule`,
  `Complete`, `TransferDone`, `Finish`, and a wake firing). An engine never cancels a wake: it
  reports a different instant, or false, the next time it is asked.
- An earlier instant than the pending wake supersedes it; a later one, or false, replaces it.
- The instant is always after `now`. A wake at or before `now` breaks the contract.
- An engine with unfinished work, nothing in flight, no transfer outstanding and an empty
  `Schedule` must report a wake, or nothing would step it again.
- At one instant a simulator processes, in this order: arrivals; transfer completions; batch
  completions; wakes and steps; timeouts last, so a request that completes at the instant its
  deadline passes has completed. An engine may rely on that order.

## Configuration: `engine.Spec`

| Field | Meaning |
|---|---|
| `Kind` | The engine model, such as `vllm`; the simulator looks it up in the factory registry. |
| `UpstreamVersion` | The real engine's release the configuration was written for. An engine refuses a release it does not model. |
| `Config` | The engine's configuration block, as YAML only the engine decodes. A factory decodes the node itself, so an error names its line in the document. |
| `Roots` | Directories the engine may read further inputs from, by role (catalog, registry, the configuration's own directory). |
| `Pool` | Which pool of a multi-pool document this engine is. |
| `Seed` | The run's seed. |

`engine.Register(f Factory)` makes a factory available under its kind (it panics on an
empty or a duplicate kind), `engine.New(spec)` builds an engine through it, and
`engine.Kinds()` lists the registered kinds.

## Checking an implementation: `engine/enginetest`

`enginetest.Drive` is a minimal discrete-event driver that follows the contract exactly.
Driving an engine through it checks the laws every engine keeps, and reports the first one
broken: no request is lost or finished twice, and none is finished that was never added;
a batch runs only requests the engine holds, each once, and preempts only requests it holds
and does not run, each once; batch IDs never repeat; batches
complete in order; no more batches are in flight than the limit; a wake is always after the
instant it was asked at; a step always takes time; a batch emits output only for its own
requests, at most one sampled token plus its accepted drafts each; a finish reports the output
tokens its batches emitted; an engine with work left always has a way to be stepped again;
and the same inputs give the same run. `enginetest.TestEngine` runs those checks as a Go test, and
`enginetest.CheckProtocolErrors` checks the error values. `enginetest.FIFO` is a deliberately
simple engine built only through `Spec` and the registry: the proof that a second engine plugs
in without changing this module.

## Rules: `rules.EngineRules`

What one engine release accepts and refuses reaches a consumer, such as a cost model, through
the `rules.EngineRules` interface rather than a concrete pack type, so the packs can live with
the engine model that owns them. [Rules packs](rules.md) lists the rules the registered packs
check.

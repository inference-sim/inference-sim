# Router contract

<p class="lede">Package <code>embed</code> is the interface between a simulator that hosts a
request router and the router it hosts. The simulator implements <code>Runtime</code>; the
router implements <code>Constructor</code>, <code>Router</code> and <code>Flight</code>. The
llm-d router fork is the implementation in use, and inference-sim hosts it.</p>

The boundary is ownership. The host owns simulated time, background work and randomness, the
endpoints' lifecycle, and what each engine publishes. The router owns every routing decision
and the meaning of its configuration. Router configuration (scorers, weights, profiles) never
appears in the contract: it travels as the deployment's own bytes in `Config`, and what the host
needs to know about it comes back as facts in `Requirements`.

A router is a deterministic function of its `Config`, the `Runtime` it runs on and the sequence
of calls the host makes. It reads time, starts concurrent work, blocks, draws randomness and
reaches the network only through the `Runtime`.

The package uses the standard library only. A test enforces that rule.

## Host side

| Method | Does |
|---|---|
| `Now() time.Time` | The current time. |
| `NewTimer(d) Timer`, `NewTicker(d) Ticker` | Timers and tickers on the runtime clock. |
| `Go(fn)` | Runs `fn` concurrently. The runtime may run it until it finishes or parks before returning. |
| `Every(ctx, name, interval, fn)` | Calls `fn` every interval until `ctx` is done. |
| `WithTimeout`, `WithDeadlineCause` | The `context` functions on the runtime clock. |
| `Park(ready)` | Called by a router goroutine before it blocks; it blocks only while `ready()` is false. |
| `Rand(stream) Rand` | A named random stream. |
| `Local(key, init) any` | State the router would otherwise keep process-wide. |

`RoundTripper` is an optional capability, found by type assertion, for a runtime that serves
the router's outbound HTTP (engine metrics, prompt rendering).

## Router side

| Method | Does |
|---|---|
| `Constructor(ctx, rt, cfg) (Router, error)` | Builds a router with no endpoints. Neither it nor work it starts calls `Rand` before it returns, because the host derives the stream scope from the constructed router. If it fails, it has already stopped what it started. A refused plugin type wraps a `*RefusalError`. |
| `Config.Allow(pluginType) (bool, string)` | The host's plugin policy. A type the router cannot host for a stated reason is refused with that reason; otherwise Allow decides; a type the router has not audited is refused even when Allow permits it. |
| `Config.Logger *slog.Logger` | Receives the router's logs; nil discards them. |
| `Config.Check(Requirements, Pool) error` | The host's own check, which the router calls after parsing the configuration and before it builds any plugin. A failure is returned unwrapped. |
| `Requirements()` | Facts about the configuration, known from the parsed configuration before any plugin exists: metric flags, inert flags, metrics extractors and sources, KV-event indexes, topology labels, whether it disaggregates. |
| `Pool()` | The pool served. `Pool.ID()` renders it as `Kind.group namespace/name`. |
| `FlowControl()` | Whether flow control is on, the longest bounded queue-wait budget and the expiry sweep. |
| `IdleSkippable()` | Whether the host may skip background ticks over an idle period. |
| `SetEndpoint`, `RemoveEndpoint` | The endpoints and their readiness. |
| `Route(ctx, req, done)` | One routing decision. `done` receives a `Flight` and a `Decision`, or a `Rejection`, exactly once. |
| `WantsKVEvents`, `IngestKVEvents`, `ClearKVEvents` | Engines' KV events, in vLLM's vocabulary (`BlockStored`, `BlockRemoved`). |
| `Stop()` | Ends every loop and goroutine the router started. |

`KVIndexInspector` is an optional capability, found by type assertion, for a router that can
report its KV-event index per endpoint and tier (`KVBlocks`). It serves conformance and
fidelity tests; a host does not depend on it in production.

A `Flight` is a dispatched request. The host hands it the response headers
(`ResponseHeaders`, which returns the headers after the router's handling) and the body
(`ResponseBody`), then ends it once: with the last body chunk, or with `Terminate`
(`client-disconnect`, `evicted`, `error`). `EvictionSignal` is closed when the router revokes
the request. It is nil when the router never evicts.

## Requests

A `Request` is what a client sends; the router renders its request headers from it, in sorted
key order. `:method` is `POST`, `:path` is `Path`, `content-type` is `application/json`, and
`x-request-id` is `ID`, always. The others are sent only when the field is set:

| Field | Header | Rendering |
|---|---|---|
| `Objective` | `x-llm-d-inference-objective` | verbatim |
| `FairnessID` | `x-llm-d-inference-fairness-id` | verbatim |
| `TTFTSLO` | `x-llm-d-slo-ttft-ms` | whole milliseconds, rounded up, when positive |
| `TTL` | `x-llm-d-inference-ttl` | `time.Duration.String`, when positive; a flow-controlled request's queue-wait budget is the shorter of it and the configured one |

## Rejections

| `RejectKind` | Meaning |
|---|---|
| `no-endpoints` | No endpoint could serve the request. |
| `ttl-expired` | It waited in flow control past its queue-wait budget. |
| `saturated` | Flow control's queues were at capacity. |
| `context-cancelled` | The client gave up while the router held it. |
| `admission` | An admission check refused it. |
| `unavailable` | The router could not serve it and gave no more specific reason. |
| `unclassified` | Any other outcome. A host treats it as a defect, not as load shedding. |

## Conformance

`embed/embedtest` checks a `Constructor` against the contract, as `testing/fstest` checks an
`fs.FS`. It drives the router through a fixed scenario on a deterministic `Runtime` that it
provides, and it plays the engines. A `Suite` supplies one router configuration per behaviour:
plain routing, flow control that holds requests, KV-event routing, and optionally eviction and
an always-unavailable configuration. It also names a plugin type to refuse. A router passes
`embedtest.Run(t, suite)` when every part behaves as this page describes.

## Compatibility

`Runtime`, `Router` and `Flight` are implemented in other repositories, so none of them gains a
method within a major version. A new capability is a separate interface. Struct types may gain
fields in a minor version; their zero value keeps the earlier meaning. A new `RejectKind` or
`Termination` value is a minor change, and a host maps a kind it does not know as it maps
`unclassified`.

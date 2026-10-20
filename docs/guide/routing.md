# Routing with the llm-d Router

Every cluster simulation routes with the **llm-d router**, the Endpoint Picker (EPP) of
[llm-d-router](https://github.com/llm-d/llm-d-router). BLIS has no routing, admission or
flow-control logic of its own: it runs the router's real code in-process, configured by the
same files a real deployment gives it, and feeds it signals produced by the simulated engines.
Two consequences follow.

- **Configuration parity.** A deployment's `EndpointPickerConfig`, its `InferencePool`,
  `InferenceObjective` and `InferenceModelRewrite` manifests, and its EPP flags run in BLIS
  unchanged, and the other way around. There is no BLIS-specific routing configuration.
- **Behavioral fidelity.** BLIS makes the routing, admission and flow-control decisions the
  router would make given the same signals. The remaining differences between a simulation and
  a real deployment come from engine modeling, not from routing.

For instance-level request ordering (the engine's own scheduler), see
[Scheduling & Priority](scheduling.md).

```bash
# The router's built-in configuration over four instances (the scenario states no router)
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml --scenarios testdata/scenarios \
  --registry $PWD/blis-registry --num-instances 4 --rate 100 --num-requests 500

# A real deployment's router configuration, unchanged, named by the scenario's router block
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node-inferencepool.yaml \
  --scenarios examples/llm-d/scenarios \
  --registry $PWD/blis-registry --num-instances 4 --rate 100 --num-requests 500
```

## The router's inputs

The router's inputs are a deployment property, so the scenario states them, as `parallel:`
states the tensor-parallel width: the deployment's `router:` block and each pool's `labels`
(blis-schemas `spec/deployment`). There are no router flags for them; `--epp-flags`,
`--epp-config`, `--epp-manifests` and `--pod-labels` were removed, as `--tp` was, and are
refused as unknown flags.

```yaml
kind: Deployment
name: my-deployment
pools:
  - role: colocated
    nodes: 4
    labels: {app: vllm}          # labels on this pool's model-server pods
    parallel: {tp: 4, pp: 1, dp: 1, enable_expert_parallel: false}
    engine: {...}
router:
  epp_flags: [--pool-name=vllm, --refresh-metrics-interval=20ms]
  endpoint_picker_config: epp-config.yaml   # relative to this scenario file
  manifests: [inferencepool.yaml]
```

| Input | Scenario field | What it is |
|---|---|---|
| EPP flags | `router.epp_flags` | The router's command-line flags, verbatim, one entry per flag (a Helm chart's `epp.flags` rendered as `--name=value`). They select the pool (`--pool-name` or `--endpoint-selector`) and set the router's own knobs (`--refresh-metrics-interval`, `--metrics-staleness-threshold`, `--feature-gates`, ...). A `--config-file` flag is read relative to the scenario file. |
| EndpointPickerConfig | `router.endpoint_picker_config` | The router configuration file, as the EPP reads it with `--config-file`. Naming it here and with an EPP `--config-file`/`--config-text` flag is refused. |
| Manifests | `router.manifests` | Kubernetes YAML files. The router reads `InferencePool`, `InferenceObjective` and `InferenceModelRewrite` objects through its own reconcilers; other kinds (the model servers' Deployments, Services, ...) are ignored. |
| Pod labels | `pools[].labels` | Labels on the pool's simulated model-server pods, so an `InferencePool` selector or a label filter from a real deployment matches. A label every pool states with one value is on every pod; any other is on its own pool's pods only. BLIS sets `llm-d.ai/role` from each prefill and decode pool's role (blis-schemas refuses a pool label that contradicts it) and `llm-d.ai/inference-serving=true` on every pod. |

Paths are relative to the scenario file's directory. A file that cannot be read is refused at
startup, naming the scenario and the field. Each input is parsed by the router's own code: the
flags by the EPP's flag set, the configuration by its config loader, the manifests by its
reconcilers. A value the real router rejects is rejected here with the router's message.
`examples/llm-d/scenarios/` holds scenarios that route with each configuration in
`examples/llm-d/`.

`blis run --trace-output` records the resolved inputs, inline (the configuration's contents, not
its path), in the trace header, with the seed the router ran with, and `blis replay` routes with
them, so a replay reproduces the run's routing whatever `--seed` it is given (INV-13). A replay
whose scenario states no router replays the recorded one. A replay whose scenario does state one
(a router block or pool labels) must state the recorded router, and a router environment flag
(below) given to `replay` must match the header's; otherwise the replay is refused, never
silently preferred over the header.

**Defaults.** With no EPP flags, BLIS runs a standalone router whose pool is every simulated
pod: `--endpoint-selector=llm-d.ai/inference-serving=true --endpoint-target-ports=8000`. With
no configuration, the router uses its built-in default configuration (queue, KV-cache
utilization and prefix-cache scorers). BLIS installs no `InferenceObjective` of its own: request
priorities come only from the deployment's objectives, which the router reconciles for an
`InferencePool` (`--pool-name`). A request whose objective has no `InferenceObjective`, and every
request in standalone mode, gets priority 0, as in a deployment.
`examples/llm-d/inferencepool.yaml` defines one objective per BLIS SLO class.

**As an API server would.** The manifests are admitted the way a cluster admits them: custom
resources get their CRD defaults (an `InferenceObjective` `poolRef` without a `group` gets
`inference.networking.k8s.io`), both served `InferenceObjective` versions (`llm-d.ai/v1` and
`llm-d.ai/v1alpha2`) are reconciled, unknown or duplicate fields are errors (strict field
validation), and a router resource at a version no CRD serves is an error. The CRDs are those of the llm-d-router and Gateway API Inference Extension versions BLIS
builds against.

## The environment around the router

Three BLIS flags model the environment, not routing:

| Flag | Default | Meaning |
|---|---|---|
| `--gateway-latency` | 0 µs | Time from a request's arrival to the router's first look at it: the network and proxy hops before the Endpoint Picker. |
| `--epp-scrape-jitter` | 0 µs | Upper bound of a seeded phase offset for each endpoint's metrics collection. At 0 every endpoint is scraped in phase. |
| `--kv-event-delay` | 0 µs | Time from an engine publishing its KV-cache events (at the end of a step) to the router's `precise-prefix-cache-producer` ingesting them: the event transport and the router's processing. At 0 the router has every event of the steps that have ended. |

The router itself takes zero simulated time to decide.

**EPP flags with no effect.** Flags that change how a deployed router is served or observed,
never what it decides (`--ha-*`, `--drain-timeout`, `--tracing`, `--health-checking`, TLS,
certificate, gRPC and metrics-endpoint flags, `--emit-endpoint-scores`), are accepted and have no
effect in a simulation; a run that sets one says so on stderr. The legacy metric-name flags
(`--total-queued-requests-metric`, ...) must name a series the simulated engines publish.

**Router logs.** The router's errors go to stderr as BLIS errors, so they show at every `--log`
level except `fatal` and `panic`; its informational messages appear with `--log debug` or
`--log trace`.

**One instance.** A single-instance cluster also runs the router: it can only pick that
instance, but its admission and flow control still apply. With the default configuration it
admits every request at once, and results are identical to a run without a router.

## What the router sees

Everything the router reads comes from the simulated engines, with real engine semantics.

- **Requests.** Each arrival is the request Envoy would hand the router: an OpenAI
  `/v1/completions` body whose `prompt` is the request's tokens, with `max_tokens` and `stream`.
  The model is the deployment's model, or the request's LoRA adapter. The headers carry
  `x-llm-d-inference-objective` (the request's SLO class), `x-llm-d-inference-fairness-id` (its
  tenant), when the request has a TTFT target `x-llm-d-slo-ttft-ms`, and when it has a client
  timeout `x-llm-d-inference-ttl`: its lifetime (deadline minus arrival) as a Go duration. Flow
  control takes the shorter of that and the configured queue budget (60 s by default), and an
  `edf-ordering-policy` orders by it; the client still gives up at its deadline, counted as
  timed out. The router's request handler, parser and Director process it exactly as in a
  deployment.
- **Multimodal requests.** A request carrying image, audio or video tokens is sent as a client
  sends media: a chat completion (`/v1/chat/completions`) whose message has an `image_url`,
  `input_audio` or `video_url` part per modality. The router's token producer sees the media
  items (its estimate backend from the parts; a vLLM render backend from BLIS's simulated
  renderer, which renders the request's tokens with each modality's placeholder span, after the
  text, and a hash of that span). With an encode pool, `always-disagg-multimodal-decider` then
  sends such a request to the encode pods the router picks (every one in `x-llm-d-encoder-hosts-ports`, in order); as in
  BLIS's own encode stage, encoding takes no simulated time, and the choice is recorded in the
  decision trace and on the request.
- **Prompt content.** BLIS token values are placeholders: a workload models prompt lengths and
  which prompts share a prefix, nothing more, and a replayed trace draws new placeholder values.
  The router, which hashes prompt content, is therefore shown canonical tokens: each KV block of
  a prompt (the last one possibly partial) is identified by its prefix chain, and every chain
  has fixed canonical tokens. Two prompts are equal exactly where they share a chain of whole
  blocks, as the engines' prefix caches see them, so a router block size that is a multiple of
  the engine's block size hashes the canonical prompt as it would the original, and a run and its
  replay show the router identical prompts.
- **Metrics.** Each engine publishes vLLM's Prometheus series at the end of every step, with
  vLLM's semantics: `vllm:num_requests_waiting`, `vllm:num_requests_running`,
  `vllm:kv_cache_usage_perc`, `vllm:num_preemptions_total`, `vllm:cache_config_info` and, when
  LoRA is active, `vllm:lora_requests_info` with the running and waiting adapter sets. The
  router scrapes them on its own schedule (`--refresh-metrics-interval`) through its own data
  source and extractor, so a scrape that lands mid-step sees the previous step's values, as in
  vLLM. Endpoints present at the start of a run have been scraped before the first request
  arrives; their periodic scrapes then keep their `--epp-scrape-jitter` phase. Endpoints that
  appear later (autoscaling, lifecycle) start cold, as on discovery.
- **KV-cache events.** With a `precise-prefix-cache-producer` in the configuration, each engine
  publishes vLLM's KV events: a stored block when a block hash enters its prefix cache (a full
  prefill or decode block, or one reloaded from a CPU tier) and a removed block when one leaves,
  at the end of the step, as vLLM does. The router receives them `--kv-event-delay` later, in
  order per engine, and ingests them synchronously (no ZMQ subscriber). The events carry the
  blocks' tokens as the router saw them, so the router's index holds, per pod, exactly the
  blocks the engine can serve from its cache. A terminated instance's blocks are cleared.
  Speculative indexing (`speculativeIndexing`) expires its entries on simulated time. With a CPU
  KV tier (`--kv-cpu-blocks`, or the `--kv-offload-config` chain), a block that becomes ready on
  the CPU tier, or leaves it, is published as a location event on tier `cpu`, in the same
  ordered stream, so the router indexes each block on every tier that holds it. A file-system
  tier behind the CPU tier publishes only when its `enable_kv_events` is set (vLLM's default is
  off): a block written to it is a location event on tier `shared_storage`, the medium llm-d's
  file-system backend publishes, which the prefix scorer weighs 0.4 by default
  (`kvCacheBackendConfigs`). With it off, the router cannot follow a block whose only copy is on
  that tier: when the block is promoted back to the CPU tier, its location event names a block
  the router has forgotten and is skipped, as llm-d skips it. A terminated instance's clear
  travels the same delayed stream, after every batch the instance published. A chain's
  canonical tokens stay fixed until the router has received the last removal of its block.
- **Prompt rendering.** The `token-producer` plugin's vLLM backend calls a render endpoint; the
  simulated model server answers it with the request's token IDs.
- **Responses.** The router sees the response headers when the engine accepts a streaming
  request, the first chunk when the first token is delivered, and the final chunk, with
  `usage` (prompt tokens and the tokens actually generated), when the response completes. A
  non-streaming response arrives whole at completion. A request that times out or is dropped
  ends with the termination cause the router's stream handler would report.
- **Endpoints.** Instances are pods of the pool, reconciled by the router's pod reconciler:
  ready while the instance is routable, not ready while it loads, drains or terminates.

## Decisions

The router's decision becomes a BLIS dispatch:

- the endpoint it picks serves the request;
- under disaggregation, the prefill endpoint the `disagg-profile-handler` names in
  `x-llm-d-prefiller-host-port` runs the prefill, and the KV moves to the picked endpoint, which
  serves the response. KV transfer is priced by the latency kernel, as before;
- a request the router does not dispatch lands in exactly one INV-1 bucket:

| Router outcome | BLIS bucket |
|---|---|
| Admission denial: an admitter plugin, or the legacy admission controller shedding a sheddable request (priority < 0) from a saturated pool | `rejected_requests` (admission) |
| No endpoint can serve it (no candidates, every filter or screener eliminated all) | `routing_rejections` |
| Flow control rejects it: queues at capacity | `gateway_queue_rejected` |
| It waits in flow control past its queue-wait budget (TTL) | `gateway_expired` |
| Its client deadline passes while the router holds it (the client disconnects; the router cancels it) | `timed_out_requests` |
| It is still waiting in flow control when the run ends | `gateway_queue_depth` |

The outcomes are an explicit allow-list keyed on the router's status code and its
`x-llm-d-request-dropped-reason`. A rejection that matches none of them (an internal error, a
malformed request) is not a routing outcome: it ends the run with an error rather than being
counted as shedding. So does a run in which the router holds requests that nothing can release.

With `--trace-level decisions`, each routing record carries the scores the router's primary
scheduling profile gave every endpoint; `--counterfactual-k` ranks them.

## Fidelity, and BLIS's former router

The acceptance criterion is behavioral fidelity to llm-d: given the same configuration and the
same signals, BLIS makes the decisions the router makes. It is not agreement with BLIS's former
native router, which ranked instances on instantaneous engine state that no real router has. The
two agree where routing cannot matter and differ, as they should, where signal freshness does.

Measured on `llama-3.1-70b-instruct-h200-tp4-4node`, seed 42 (BLIS main at `0d092d4e` vs this
branch, the same kernel and schemas; mean TTFT / p99 TTFT in ms):

| Workload | Former native router | llm-d default configuration | llm-d with `examples/llm-d/inflight-load.yaml` |
|---|---|---|---|
| 4 instances, 10 req/s, 800 requests | 87.7 / 119.4 | 87.5 / 118.6 | 87.7 / 119.4 |
| 4 instances, 40 req/s, 2000 requests | 96.8 / 129.6 | 184.6 / 385.1 | 105.5 / 189.7 |
| 8 instances, 80 req/s, 3000 requests | 95.5 / 129.5 | 401.8 / 964.5 | 105.4 / 186.0 |

Completed requests agree exactly, and E2E latency and throughput within 1.2%, in every cell. The
difference is placement under load:

- The default configuration ranks endpoints only on scraped metrics. llm-d scrapes at
  `--refresh-metrics-interval` (50 ms, which is also the minimum it accepts), and a vLLM engine
  updates its gauges only at the end of a step, so between scrapes every request sees the same
  ranking and goes to the same endpoint. That herding is what the real router does with this
  configuration; it is the effect a capacity plan should see.
- The router's own in-flight bookkeeping is current at every decision. A configuration that
  scores on it, as `examples/llm-d/inflight-load.yaml` does with llm-d's
  `inflight-load-producer` and `active-request-scorer`, comes within 10% of the former
  router's mean TTFT.
- `--epp-scrape-jitter` (here 50 ms) de-phases the endpoints' scrapes and moves the default
  configuration's mean TTFT by 6-7%; staleness, not phase, dominates.

What is not yet measured: a comparison against traces of a real llm-d deployment.

## Supported plugins

A configuration may use these plugin types. Each was audited in the router fork: every clock
read, background loop, blocking wait and random draw it makes goes through the hosting runtime,
and every signal it consumes is produced by the simulated engines.

| Kind | Types |
|---|---|
| Profile handlers and pickers | `single-profile-handler`, `disagg-profile-handler`, `max-score-picker`, `random-picker`, `weighted-random-picker` |
| Filters | `decode-filter`, `prefill-filter`, `encode-filter`, `label-selector-filter` |
| Scorers | `queue-scorer`, `kv-cache-utilization-scorer`, `prefix-cache-scorer`, `load-aware-scorer`, `running-requests-size-scorer`, `lora-affinity-scorer`, `token-load-scorer`, `no-hit-lru-scorer`, `active-request-scorer`, `context-length-aware` |
| P/D deciders | `always-disagg-pd-decider`, `prefix-based-pd-decider` |
| Data producers | `approx-prefix-cache-producer`, `precise-prefix-cache-producer` (fed by the engines' KV events), `inflight-load-producer`, `token-producer` (estimate backend, or vLLM backend with a fixed URL) |
| Admission | `probabilistic-admitter` |
| Saturation detectors | `utilization-detector`, `concurrency-detector`, `max-saturation-detector` |
| Flow control | `global-strict-fairness-policy`, `round-robin-fairness-policy`, `fcfs-ordering-policy`, `edf-ordering-policy`, `slo-deadline-ordering-policy`, `static-usage-limit-policy`, `strict-band-selection-policy`, `priority-holdback-policy`, `soft-reflective-ceiling-policy` |
| Data layer | `metrics-data-source`, `core-metrics-extractor`, `endpoint-notification-source` |
| Request parsers | `openai-parser`, `vllmhttp-parser`, `sglanghttp-parser`, `anthropic-parser`, `passthrough-parser`, `vertexai-parser` |

The flow-control layer (`featureGates: [flowControl]`) runs in full: priority bands, capacity
limits, queue-wait budgets, the 1 ms dispatch tick and the 1 s expiry sweep.

## Refused at startup

Anything that cannot be simulated faithfully is refused before the simulation starts, with the
reason. Nothing degrades silently. Plugins are checked after the configuration is loaded and
before any of them does work, wherever the configuration references them: under `plugins:`, or
only through a reference such as a flow-control `fairnessPolicyRef`.

| What | Why |
|---|---|
| `burst-prefix-cache-producer` | It blocks a request on a sealing timer the hosting runtime does not model. |
| `predicted-latency-producer`, `latency-scorer`, `latency-slo-admitter`, `slo-headroom-tier-filter`, `prefix-cache-affinity-filter` | They call or consume an external latency-predictor service. |
| `latency-observer-producer-hub`, `latency-observation-scorer-hub`, `session-state-producer`, `program-aware-fairness`, `disaggregatedset-rollout-screener`, `utilization-filter`, `request-attribute-reporter` | Not yet audited for, or not yet moved onto, the hosting runtime. |
| `session-affinity-*`, `session-id-producer`, `agent-identity`, `outlen-bucket`, `header-profile-handler`, `header-label-affinity-scorer`, `endpoint-pin-screener` | The simulated workload does not send the headers they read. |
| `mm-embeddings-*` | The simulated media have no content of their own, so there is no embedding to cache. |
| `p2p-source-producer`, `selective-kv-policy` | Peer KV pulls are not modeled by the simulated engines. |
| `topology-*`, `endpoint-attribute-*`, `label-producer`, `dcgm-*` | The simulated pods carry no topology labels, custom attributes or GPU telemetry. |
| `models-data-source`, `models-data-extractor`, `models-responder` | The simulated engines do not serve `/v1/models` yet; the workload sends only completions. |
| `redis-syncer`, `thunder-agent`, `k8s-notification-source`, `k8s-peer-discovery`, `file-discovery`, `multicluster-*` | They need external services or another discovery source than the simulated deployment. |
| More than one `InferencePool`; objective or rewrite manifests with `--endpoint-selector` | A run has one router for one pool; in standalone mode the router does not reconcile objectives or rewrites. |
| A pool with more than one target port | Several ports make each pod several endpoints (data-parallel ranks behind one pod); BLIS simulates each rank as its own pod. |
| Prefill and decode pools with a configuration that has no `disagg-profile-handler` | The router would serve every request from one flat pool; pass a disaggregating configuration (for example `examples/llm-d/pd.yaml`). |
| An encode pool on the command line (`--encode-instances`) | No scenario describes the engine an encode pod runs (a blis-schemas deployment's roles are colocated, prefill and decode), as on BLIS `main`. Encode pools are available through the Go API (`DeploymentConfig.EncodeInstances`). |
| A `token-producer` whose vLLM backend discovers endpoints (`vllm.endpointDiscovery`) | It probes pods for model limits on its own schedule, outside the simulation. Use a fixed `url`, or the estimate backend. |
| A `precise-prefix-cache-producer` whose `tokenProcessorConfig.blockSizeTokens` (default 16) is not the engines' KV block size | Its keys would never match the engines' blocks. |
| A `precise-prefix-cache-producer` with a Redis or cost-aware index, or non-vLLM `kvEventsConfig.engineType`; with LoRA adapters in the deployment | Redis is an external service; the cost-aware index's lookups have no fixed order; the engines publish vLLM's events and their prefix caches do not key blocks by adapter. |
| A `core-metrics-extractor` that, for some pod, reads an engine other than vLLM (by `defaultEngine` or the pod's engine label, `llm-d.ai/engine-type` by default), or a vLLM mapping naming series the engines do not publish; a `metrics-data-source` path other than `/metrics` | The simulated engines publish vLLM's series at `/metrics`; anything else would read nothing. |

Also outside the model: conditional decode (`Prefer: if-available`), and the P/D sidecar's
MoRIIO parallel prefill-and-decode dispatch (BLIS always runs prefill, then the KV transfer,
then decode). BLIS hosts the router's Endpoint Picker only: llm-d's coordinator and its E/P/D
`pipeline:` (multimodal encode, then prefill, then decode, as in llm-d's coordinator
disaggregation guide) are not simulated, so E/P/D parity is the EPP's decisions alone.

## From BLIS's former routing features

BLIS no longer has its own routing, admission or flow control. Each former feature either maps
to an llm-d configuration or is dropped.

| Former BLIS feature | llm-d configuration |
|---|---|
| `--routing-policy weighted --routing-scorers ...` | A scheduling profile with the matching scorers and weights, and `max-score-picker`. |
| scorer `prefix-affinity` | `approx-prefix-cache-producer` + `prefix-cache-scorer` |
| scorer `precise-prefix-cache` | `precise-prefix-cache-producer` + `prefix-cache-scorer` (with a `token-producer`), indexing the engines' KV events |
| scorers `queue-depth`, `kv-utilization`, `active-requests`, `running-requests`, `load-aware`, `no-hit-lru` | `queue-scorer`, `kv-cache-utilization-scorer`, `active-request-scorer`, `running-requests-size-scorer`, `load-aware-scorer`, `no-hit-lru-scorer` |
| `--lora-scorer-weight` | `lora-affinity-scorer` in the profile |
| `--routing-policy round-robin`, `least-loaded`, `always-busiest`; scorers `load-balance`, `vllm-dp` | Dropped: the llm-d router has no such policy. The nearest configurations are `random-picker`, and `queue-scorer` or `load-aware-scorer` with `max-score-picker`. |
| `--admission-policy gaie-legacy` | The router's default admission: with the `flowControl` gate off, the legacy admission controller sheds sheddable requests (priority < 0) when the saturation detector reports a saturated pool. Its thresholds are `utilization-detector` parameters. |
| `--admission-policy tier-shed` | `InferenceObjective` priorities with the legacy admission controller, or flow-control priority bands and a usage-limit policy. |
| `--admission-policy token-bucket`, `reject-all`; `tenant_budgets` | Dropped. Per-tenant fairness is the fairness ID (the request's tenant) with `round-robin-fairness-policy`. |
| `--flow-control` | `featureGates: [flowControl]` |
| `--saturation-detector`, `--queue-depth-threshold`, `--kv-cache-util-threshold`, `--max-concurrency` | `utilization-detector` or `concurrency-detector` and their parameters |
| `--dispatch-order fifo`, `slo-deadline`; `--slo-targets` | `fcfs-ordering-policy`, `slo-deadline-ordering-policy` |
| `--fairness-policy` | `global-strict-fairness-policy`, `round-robin-fairness-policy` |
| `--max-gateway-queue-depth`, `--per-band-capacity` | `flowControl.maxRequests`, a priority band's capacity |
| `--usage-limit-threshold` | `static-usage-limit-policy` |
| `--request-ttl` | `flowControl.defaultRequestTTL` |
| `--queue-shedding`, `--dispatch-tick-interval`, `--in-flight-eviction` | Dropped: BLIS-only behaviors. llm-d's dispatch tick is fixed at 1 ms; its in-flight eviction is `flowControl.enableEviction` ([admission](admission.md)). |
| `--pd-decider always`, `prefix-threshold`; `--pd-prefix-threshold` | `disagg-profile-handler` with `always-disagg-pd-decider` or `prefix-based-pd-decider` (`nonCachedTokens`) |
| `--prefill-routing-scorers`, `--decode-routing-scorers` | The `prefill` and `decode` scheduling profiles |
| `--encode-decider` | The disaggregation handler's encode decider (`always-disagg-multimodal-decider` for `multimodal`) |
| `--admission-latency`, `--routing-latency` | `--gateway-latency` |
| `--snapshot-refresh-interval` | The EPP flag `--refresh-metrics-interval`: staleness now comes from the scrape itself |
| `--cache-signal-delay` | `--kv-event-delay`: a fixed delay from an engine publishing its KV events to the router ingesting them (the old flag refreshed a snapshot of every cache at most every 50 ms) |
| `examples/epp-*.yaml` | The llm-d configurations themselves (`deploy/config/` in llm-d-router) |

## Further reading

- [Cluster Simulation](cluster.md) — the cluster pipeline around the router.
- [Architecture](../concepts/architecture.md) — how the router is embedded.
- [The llm-d router fork](../contributing/llm-d-router-fork.md) — how the fork hosts the router
  and how it is kept current.
- [Invariants](../contributing/standards/invariants.md) — INV-1, INV-6, INV-7, INV-9 and INV-13
  as they apply to the router.

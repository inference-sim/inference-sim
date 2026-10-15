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
# The router's built-in configuration over four instances
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml --scenarios testdata/scenarios \
  --registry $PWD/blis-registry --num-instances 4 --rate 100 --num-requests 500

# A real deployment's router configuration, unchanged
./blis run ... --epp-config epp-config.yaml --epp-flags epp-flags.yaml --epp-manifests inferencepool.yaml
```

## The router's inputs

| Input | CLI flag | Deployment key | What it is |
|---|---|---|---|
| EPP flags | `--epp-flags FILE` | `router.epp_flags` | The router's command-line flags, verbatim: a YAML list of flags, or a map of flag name to value as in the router Helm chart's `epp.flags`. They select the pool (`--pool-name` or `--endpoint-selector`) and set the router's own knobs (`--refresh-metrics-interval`, `--metrics-staleness-threshold`, `--feature-gates`, ...). |
| EndpointPickerConfig | `--epp-config FILE` | `router.endpoint_picker_config` | The router configuration, as the EPP reads it with `--config-file`. |
| Manifests | `--epp-manifests FILE[,FILE...]` | `router.manifests` | Kubernetes YAML. The router reads `InferencePool`, `InferenceObjective` and `InferenceModelRewrite` objects through its own reconcilers; other kinds (the model servers' Deployments, Services, ...) are ignored. |
| Pod labels | `--pod-labels k=v[,...]` | `pools[].labels` | Labels on the simulated model-server pods, so an `InferencePool` selector from a real deployment matches. BLIS also sets `llm-d.ai/role` (`prefill`, `decode`, `prefill-decode`, `encode`) on disaggregated deployments and `llm-d.ai/inference-serving=true` on every pod. |

Each input is parsed by the router's own code: the flags by the EPP's flag set, the
configuration by its config loader, the manifests by its reconcilers. A value the real router
rejects is rejected here with the router's message.

**Defaults.** With no EPP flags, BLIS runs a standalone router whose pool is every simulated
pod: `--endpoint-selector=llm-d.ai/inference-serving=true --endpoint-target-ports=8000`. With
no configuration, the router uses its built-in default configuration (queue, KV-cache
utilization and prefix-cache scorers). With `--pool-name` and no manifests other than the
`InferencePool`, BLIS installs one `InferenceObjective` per BLIS SLO class at the Gateway API
Inference Extension default priorities (`critical` 4, `standard` 3, `batch` -1, `sheddable` -2,
`background` -3).

## The environment around the router

Three BLIS knobs model the environment, not routing:

| Flag | Default | Meaning |
|---|---|---|
| `--gateway-latency` | 0 µs | Time from a request's arrival to the router's first look at it: the network and proxy hops before the Endpoint Picker. |
| `--epp-scrape-jitter` | 0 µs | Upper bound of a seeded phase offset for each endpoint's metrics collection. At 0 every endpoint is scraped in phase. |

The router itself takes zero simulated time to decide.

## What the router sees

Everything the router reads comes from the simulated engines, with real engine semantics.

- **Requests.** Each arrival is the request Envoy would hand the router: an OpenAI
  `/v1/completions` body whose `prompt` is the request's token IDs, with `max_tokens` and
  `stream`. The model is the deployment's model, or the request's LoRA adapter. The headers
  carry `x-llm-d-inference-objective` (the request's SLO class) and
  `x-llm-d-inference-fairness-id` (its tenant). The router's request handler, parser and
  Director process it exactly as in a deployment.
- **Metrics.** Each engine publishes vLLM's Prometheus series at the end of every step, with
  vLLM's semantics: `vllm:num_requests_waiting`, `vllm:num_requests_running`,
  `vllm:kv_cache_usage_perc`, `vllm:num_preemptions_total`, `vllm:cache_config_info` and, when
  LoRA is active, `vllm:lora_requests_info` with the running and waiting adapter sets. The
  router scrapes them on its own schedule (`--refresh-metrics-interval`) through its own data
  source and extractor, so a scrape that lands mid-step sees the previous step's values, as in
  vLLM. Endpoints present at the start of a run have been scraped once already; endpoints that
  appear later (autoscaling, lifecycle) start cold, as on discovery.
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
| It is still waiting in flow control when the run ends | `gateway_queue_depth` |

With `--trace-level decisions`, each routing record carries the scores the router's primary
scheduling profile gave every endpoint; `--counterfactual-k` ranks them.

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
| Data producers | `approx-prefix-cache-producer`, `inflight-load-producer`, `token-producer` (estimate backend, or vLLM backend with a fixed URL) |
| Admission | `probabilistic-admitter` |
| Saturation detectors | `utilization-detector`, `concurrency-detector`, `max-saturation-detector` |
| Flow control | `global-strict-fairness-policy`, `round-robin-fairness-policy`, `fcfs-ordering-policy`, `edf-ordering-policy`, `slo-deadline-ordering-policy`, `static-usage-limit-policy`, `strict-band-selection-policy`, `priority-holdback-policy`, `soft-reflective-ceiling-policy` |
| Data layer | `metrics-data-source`, `core-metrics-extractor`, `endpoint-notification-source` |
| Request parsers | `openai-parser`, `vllmhttp-parser`, `sglanghttp-parser`, `anthropic-parser`, `passthrough-parser`, `vertexai-parser` |

The flow-control layer (`featureGates: [flowControl]`) runs in full: priority bands, capacity
limits, queue-wait budgets, the 1 ms dispatch tick and the 1 s expiry sweep.

## Refused at startup

Anything that cannot be simulated faithfully is refused before the simulation starts, with the
reason. Nothing degrades silently.

| What | Why |
|---|---|
| `precise-prefix-cache-producer` | It consumes vLLM KV-cache events, which the simulated engines do not publish yet. |
| `burst-prefix-cache-producer` | It blocks a request on a sealing timer the hosting runtime does not model. |
| `predicted-latency-producer`, `latency-scorer`, `latency-slo-admitter`, `slo-headroom-tier-filter`, `prefix-cache-affinity-filter` | They call or consume an external latency-predictor service. |
| `latency-observer-producer-hub`, `latency-observation-scorer-hub`, `session-state-producer`, `program-aware-fairness`, `disaggregatedset-rollout-screener`, `utilization-filter`, `request-attribute-reporter` | Not yet audited for, or not yet moved onto, the hosting runtime. |
| `session-affinity-*`, `session-id-producer`, `agent-identity`, `outlen-bucket`, `header-profile-handler`, `header-label-affinity-scorer`, `endpoint-pin-screener` | The simulated workload does not send the headers they read. |
| `mm-embeddings-*`, `always-disagg-multimodal-decider` | The simulated workload carries no multimodal inputs. |
| `p2p-source-producer`, `selective-kv-policy` | Peer KV pulls are not modeled by the simulated engines. |
| `topology-*`, `endpoint-attribute-*`, `label-producer`, `dcgm-*` | The simulated pods carry no topology labels, custom attributes or GPU telemetry. |
| `models-data-source`, `models-data-extractor`, `models-responder` | The simulated engines do not serve `/v1/models` yet; the workload sends only completions. |
| `redis-syncer`, `thunder-agent`, `k8s-notification-source`, `k8s-peer-discovery`, `file-discovery`, `multicluster-*` | They need external services or another discovery source than the simulated deployment. |
| Flow-control in-flight eviction (`enableEviction`) | It revokes dispatched requests, and the simulation does not model that revocation yet. |
| More than one `InferencePool`; objective or rewrite manifests with `--endpoint-selector` | A run has one router for one pool; in standalone mode the router does not reconcile objectives or rewrites. |

Also outside the model: the token-producer's vLLM backend with endpoint discovery (it probes
pods for model limits), conditional decode (`Prefer: if-available`), and the P/D sidecar's
MoRIIO parallel prefill-and-decode dispatch (BLIS always runs prefill, then the KV transfer,
then decode).

## From BLIS's former routing features

BLIS no longer has its own routing, admission or flow control. Each former feature either maps
to an llm-d configuration or is dropped.

| Former BLIS feature | llm-d configuration |
|---|---|
| `--routing-policy weighted --routing-scorers ...` | A scheduling profile with the matching scorers and weights, and `max-score-picker`. |
| scorer `prefix-affinity` | `approx-prefix-cache-producer` + `prefix-cache-scorer` |
| scorer `precise-prefix-cache` | `precise-prefix-cache-producer` + `prefix-cache-scorer`, refused until the engines publish KV events |
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
| `--queue-shedding`, `--dispatch-tick-interval`, `--in-flight-eviction` | Dropped: BLIS-only behaviors. llm-d's dispatch tick is fixed at 1 ms; its in-flight eviction is refused (above). |
| `--pd-decider always`, `prefix-threshold`; `--pd-prefix-threshold` | `disagg-profile-handler` with `always-disagg-pd-decider` or `prefix-based-pd-decider` (`nonCachedTokens`) |
| `--prefill-routing-scorers`, `--decode-routing-scorers` | The `prefill` and `decode` scheduling profiles |
| `--encode-decider` | The disaggregation handler's encode decider; multimodal deciders are refused |
| `--admission-latency`, `--routing-latency` | `--gateway-latency` |
| `--snapshot-refresh-interval` | The EPP flag `--refresh-metrics-interval`: staleness now comes from the scrape itself |
| `--cache-signal-delay` | Dropped with the precise prefix cache, until KV events are simulated |
| `examples/epp-*.yaml` | The llm-d configurations themselves (`deploy/config/` in llm-d-router) |

## Further reading

- [Cluster Simulation](cluster.md) — the cluster pipeline around the router.
- [Architecture](../concepts/architecture.md) — how the router is embedded.
- [The llm-d router fork](../contributing/llm-d-router-fork.md) — how the fork hosts the router
  and how it is kept current.
- [Invariants](../contributing/standards/invariants.md) — INV-1, INV-6, INV-7, INV-9 and INV-13
  as they apply to the router.

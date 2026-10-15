package llmd

import (
	"fmt"
	"sort"
	"strings"

	"github.com/llm-d/llm-d-router/cmd/epp/runner"
)

// Supported lists the router plugin types a simulation runs faithfully. Each one was audited in
// the fork: every clock read, background loop, blocking wait and random draw it makes goes
// through the hosting runtime, and every signal it consumes is produced by the simulated
// engines. A configuration that instantiates any other type is refused at startup with the
// reason from Unsupported (or a generic one): the simulation never runs a plugin it cannot run
// as llm-d does.
var Supported = map[string]bool{
	// profile handlers and pickers
	"single-profile-handler": true,
	"disagg-profile-handler": true,
	"max-score-picker":       true,
	"random-picker":          true,
	"weighted-random-picker": true,
	// filters
	"decode-filter":         true,
	"prefill-filter":        true,
	"encode-filter":         true,
	"label-selector-filter": true,
	// scorers
	"queue-scorer":                 true,
	"kv-cache-utilization-scorer":  true,
	"prefix-cache-scorer":          true,
	"load-aware-scorer":            true,
	"running-requests-size-scorer": true,
	"lora-affinity-scorer":         true,
	"token-load-scorer":            true,
	"no-hit-lru-scorer":            true,
	"active-request-scorer":        true,
	"context-length-aware":         true,
	// P/D deciders
	"always-disagg-pd-decider": true,
	"prefix-based-pd-decider":  true,
	// data producers
	"approx-prefix-cache-producer": true,
	"inflight-load-producer":       true,
	"token-producer":               true,
	// admission
	"probabilistic-admitter": true,
	// saturation detectors
	"utilization-detector":    true,
	"concurrency-detector":    true,
	"max-saturation-detector": true,
	// flow control
	"global-strict-fairness-policy":            true,
	"round-robin-fairness-policy":              true,
	"fcfs-ordering-policy":                     true,
	"edf-ordering-policy":                      true,
	"slo-deadline-ordering-policy":             true,
	"static-usage-limit-policy":                true,
	"static-usage-limit-policy-type":           true, // the policy's TypedName type
	"strict-band-selection-policy":             true,
	"priority-holdback-policy":                 true,
	"soft-reflective-ceiling-policy":           true,
	"sheddable-eviction-filter":                true,
	"priority-then-time-eviction-order-policy": true,
	// data layer
	"metrics-data-source":          true,
	"core-metrics-extractor":       true,
	"endpoint-notification-source": true,
	// request parsing: the simulated workload sends only token-ID completions requests, and the
	// parsers are pure functions of the request and response bytes
	"openai-parser":      true,
	"vllmhttp-parser":    true,
	"sglanghttp-parser":  true,
	"anthropic-parser":   true,
	"passthrough-parser": true,
	"vertexai-parser":    true,
}

// Unsupported gives the reason a plugin type cannot be simulated faithfully.
var Unsupported = map[string]string{
	"burst-prefix-cache-producer":               "it blocks a request on a sealing timer the hosting runtime does not model",
	"precise-prefix-cache-producer":             "it consumes vLLM KV-cache events, which the simulated engines do not publish yet",
	"predicted-latency-producer":                "it calls an external latency-predictor service",
	"latency-scorer":                            "it consumes the external latency predictor",
	"latency-slo-admitter":                      "it consumes the external latency predictor",
	"slo-headroom-tier-filter":                  "it consumes the external latency predictor",
	"prefix-cache-affinity-filter":              "it consumes the external latency predictor",
	"latency-observer-producer-hub":             "its clock reads have not been moved onto the hosting runtime",
	"latency-observation-scorer-hub":            "its exploration draws from process-wide randomness",
	"session-state-producer":                    "its clock reads have not been moved onto the hosting runtime",
	"session-affinity-scorer":                   "the simulated workload sends no session header",
	"session-affinity-filter":                   "the simulated workload sends no session header",
	"session-id-producer":                       "the simulated workload sends no session header",
	"program-aware-fairness":                    "its background eviction loop and clock reads have not been moved onto the hosting runtime",
	"thunder-agent":                             "it calls an external agent service",
	"redis-syncer":                              "it synchronizes through an external Redis",
	"mm-embeddings-cache-producer":              "the simulated workload carries no multimodal inputs",
	"mm-embeddings-cache-scorer":                "the simulated workload carries no multimodal inputs",
	"always-disagg-multimodal-decider":          "the simulated workload carries no multimodal inputs",
	"p2p-source-producer":                       "peer KV pulls are not modeled by the simulated engines",
	"selective-kv-policy":                       "peer KV pulls are not modeled by the simulated engines",
	"dcgm-data-source":                          "the simulated engines publish no DCGM GPU telemetry",
	"dcgm-extractor":                            "the simulated engines publish no DCGM GPU telemetry",
	"k8s-notification-source":                   "it needs a Kubernetes controller manager",
	"k8s-peer-discovery":                        "it needs a Kubernetes API server",
	"file-discovery":                            "endpoints come from the simulated deployment, not from a discovery file",
	"multicluster-file-discovery":               "multi-cluster routing is not simulated",
	"disaggregatedset-rollout-screener":         "its revision split draws from process-wide randomness",
	"topology-affinity-filter":                  "the simulated pods carry no topology labels",
	"topology-affinity-scorer":                  "the simulated pods carry no topology labels",
	"topology-extractor":                        "the simulated pods carry no topology labels",
	"topology-stamp-handler":                    "the simulated pods carry no topology labels",
	"models-data-source":                        "the simulated engines do not serve /v1/models yet",
	"models-data-extractor":                     "the simulated engines do not serve /v1/models yet",
	"models-responder":                          "the simulated workload sends only completions requests",
	"header-based-testing-filter":               "it is a conformance-test fixture",
	"destination-endpoint-served-verifier":      "it is a conformance-test fixture",
	"agent-identity":                            "the simulated workload carries no agent identity",
	"outlen-bucket":                             "its request headers are not produced by the simulated workload",
	"header-profile-handler":                    "its request headers are not produced by the simulated workload",
	"header-label-affinity-scorer":              "its request headers are not produced by the simulated workload",
	"endpoint-attribute-filter":                 "the simulated pods carry no custom endpoint attributes",
	"endpoint-attribute-scorer":                 "the simulated pods carry no custom endpoint attributes",
	"endpoint-attribute-weight-scorer":          "the simulated pods carry no custom endpoint attributes",
	"label-producer":                            "the simulated pods carry no custom endpoint attributes",
	"utilization-filter":                        "it has not been audited for the hosting runtime",
	"request-attribute-reporter":                "it has not been audited for the hosting runtime",
	"endpoint-pin-screener":                     "its request headers are not produced by the simulated workload",
	"multicluster-approx-prefix-cache-producer": "multi-cluster routing is not simulated",
	"multicluster-kv-cache-utilization-scorer":  "multi-cluster routing is not simulated",
	"multicluster-metrics-data-source":          "multi-cluster routing is not simulated",
	"multicluster-metrics-extractor":            "multi-cluster routing is not simulated",
	"multicluster-prefix-cache-scorer":          "multi-cluster routing is not simulated",
	"multicluster-queue-scorer":                 "multi-cluster routing is not simulated",
	"multicluster-session-affinity-filter":      "multi-cluster routing is not simulated",
}

// validateEmbedded refuses a router whose plugins (including the producers the router created
// for them) include a type that cannot be simulated, and a router that needs services the
// simulation cannot provide.
func validateEmbedded(e *runner.Embedded) error {
	var problems []string
	for name, p := range e.Handle.GetAllPluginsWithNames() {
		typ := p.TypedName().Type
		if Supported[typ] {
			continue
		}
		reason := Unsupported[typ]
		if reason == "" {
			reason = "it has not been audited for the hosting runtime"
		}
		problems = append(problems, fmt.Sprintf("plugin %q (type %s): %s", name, typ, reason))
	}
	sort.Strings(problems)
	if len(problems) > 0 {
		return fmt.Errorf("the router configuration cannot be simulated faithfully:\n  %s", strings.Join(problems, "\n  "))
	}
	return nil
}

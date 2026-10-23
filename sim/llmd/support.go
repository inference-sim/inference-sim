package llmd

import (
	"cmp"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/inference-sim/blis-schemas/embed"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// supported is BLIS's signal policy: the router plugin types whose every consumed signal the
// simulated engines and workload produce. The router consults it (embed.Config.Allow) for every
// type the fork can host, before that type's constructor runs; the fork refuses the types it
// cannot host with its own reason. Each type here must also be in the fork's audited set, so a
// plugin runs only when both agree: the simulation never runs a plugin it cannot run as llm-d
// does.
var supported = map[string]bool{
	// profile handlers and pickers
	"single-profile-handler": true,
	"disagg-profile-handler": true,
	"max-score-picker":       true,
	"random-picker":          true,
	"weighted-random-picker": true,
	// filters
	"decode-filter":                    true,
	"prefill-filter":                   true,
	"encode-filter":                    true,
	"always-disagg-multimodal-decider": true,
	"label-selector-filter":            true,
	"topology-affinity-filter":         true,
	"topology-stamp-handler":           true,
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
	"topology-affinity-scorer":     true,
	// P/D deciders
	"always-disagg-pd-decider": true,
	"prefix-based-pd-decider":  true,
	// data producers
	"approx-prefix-cache-producer": true,
	"inflight-load-producer":       true,
	"token-producer":               true,
	// The engines publish vLLM's KV-cache events, ingested synchronously by the hosted producer
	// (sim/llmd/kvevents.go); checkPrecisePrefix refuses the configurations it cannot host.
	"precise-prefix-cache-producer": true,
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
	"topology-extractor":           true,
	// request parsing: the simulated workload sends only token-ID completions requests, and the
	// parsers are pure functions of the request and response bytes
	"openai-parser":      true,
	"vllmhttp-parser":    true,
	"sglanghttp-parser":  true,
	"anthropic-parser":   true,
	"passthrough-parser": true,
	"vertexai-parser":    true,
}

// signalReasons give, for plugin types BLIS's signal policy refuses, why: a signal the simulated
// engines or workload do not produce. The fork gives the reasons of the types it cannot host.
var signalReasons = map[string]string{
	"session-affinity-scorer":                   "the simulated workload sends no session header",
	"session-affinity-filter":                   "the simulated workload sends no session header",
	"session-id-producer":                       "the simulated workload sends no session header",
	"mm-embeddings-cache-producer":              "the simulated workload carries no multimodal inputs",
	"mm-embeddings-cache-scorer":                "the simulated workload carries no multimodal inputs",
	"p2p-source-producer":                       "peer KV pulls are not modeled by the simulated engines",
	"selective-kv-policy":                       "peer KV pulls are not modeled by the simulated engines",
	"dcgm-data-source":                          "the simulated engines publish no DCGM GPU telemetry",
	"dcgm-extractor":                            "the simulated engines publish no DCGM GPU telemetry",
	"file-discovery":                            "endpoints come from the simulated deployment, not from a discovery file",
	"multicluster-file-discovery":               "multi-cluster routing is not simulated",
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
	"endpoint-pin-screener":                     "its request headers are not produced by the simulated workload",
	"multicluster-approx-prefix-cache-producer": "multi-cluster routing is not simulated",
	"multicluster-kv-cache-utilization-scorer":  "multi-cluster routing is not simulated",
	"multicluster-metrics-data-source":          "multi-cluster routing is not simulated",
	"multicluster-metrics-extractor":            "multi-cluster routing is not simulated",
	"multicluster-prefix-cache-scorer":          "multi-cluster routing is not simulated",
	"multicluster-queue-scorer":                 "multi-cluster routing is not simulated",
	"multicluster-session-affinity-filter":      "multi-cluster routing is not simulated",
}

// notAudited is the reason for a type with no recorded one.
const notAudited = "it has not been audited for the hosting runtime"

// allow is BLIS's plugin policy, passed to the router as embed.Config.Allow.
func allow(typ string) (bool, string) {
	if supported[typ] {
		return true, ""
	}
	return false, cmp.Or(signalReasons[typ], notAudited)
}

// IsSupported reports whether BLIS's signal policy allows router plugins of type typ. Every type it
// allows must also be in the fork's audited set (the fork's pkg/embed tests hold it there); a type
// runs only when both agree.
func IsSupported(typ string) bool { return supported[typ] }

// UnsupportedReason is why BLIS's signal policy refuses plugins of type typ, or "" for a supported
// type or one with no signal-level reason. A type the fork cannot host is refused by the fork, with
// the fork's reason, at startup; this returns "" for it.
func UnsupportedReason(typ string) string { return signalReasons[typ] }

// legacyMetricFlags name the engine series the router reads when its configuration does not, in
// the order they are checked.
var legacyMetricFlags = []string{
	"total-queued-requests-metric", "total-running-requests-metric", "kv-cache-usage-percentage-metric",
	"lora-info-metric", "cache-info-metric",
}

// checkMetricFlags refuses router flags that name an engine series the simulated engines do not
// publish.
func checkMetricFlags(set map[string]string) error {
	for _, name := range legacyMetricFlags {
		if v := set[name]; v != "" && !isPublished(v) {
			return fmt.Errorf("EPP flag --%s=%s names an engine series the simulated engines do not publish (%s)",
				name, v, strings.Join(publishedSeries, ", "))
		}
	}
	return nil
}

// topologyLabels are the pod labels BLIS's synthetic Pods carry for each topology level.
var topologyLabels = []struct{ key, label string }{
	{"hostname", "kubernetes.io/hostname"},
	{"rack", "topology.kubernetes.io/rack"},
	{"zone", "topology.kubernetes.io/zone"},
	{"region", "topology.kubernetes.io/region"},
}

// checkTopologyLabels admits only topology extractors that read the labels BLIS sets on its
// synthetic Pods from the simulated placement.
func checkTopologyLabels(t embed.Topology) error {
	for _, x := range t.Extractors {
		for i, value := range []string{x.Hostname, x.Rack, x.Zone, x.Region} {
			level := topologyLabels[i]
			if value != "" && value != level.label {
				return fmt.Errorf("topology-extractor %s label %q is not supported; synthetic Pods use %q", level.key, value, level.label)
			}
		}
	}
	return nil
}

// checkEngineSignals refuses a configuration that reads a signal the simulated engines do not
// produce:
//   - a precise-prefix-cache-producer whose block size differs from the engines' (its keys would
//     never match the KV events), or that reads another engine's KV events;
//   - a metrics-data-source scraping another path than vLLM's /metrics, which would read nothing;
//   - a core-metrics-extractor that reads, for some pod, an engine's series the simulated engines
//     do not publish: the extractor picks each pod's mapping by its engine label or else its
//     default engine, and the simulated engines publish vLLM's series. The router's implicit
//     extractor is checked too.
func checkEngineSignals(req embed.Requirements, podLabels []map[string]string, engineBlockSize int) error {
	for _, p := range req.PrefixIndexes {
		if engineBlockSize > 0 && p.BlockSize != engineBlockSize {
			return fmt.Errorf("plugin %q cuts block keys every %d tokens, but the simulated engines' KV block size is %d; "+
				"set tokenProcessorConfig.blockSizeTokens to %d", p.Name, p.BlockSize, engineBlockSize, engineBlockSize)
		}
		if p.EngineType != "" && p.EngineType != "vllm" {
			return fmt.Errorf("plugin %q reads %s KV events; the simulated engines publish vLLM's", p.Name, p.EngineType)
		}
	}
	for _, m := range req.MetricsSources {
		if m.Path != "" && m.Path != "/metrics" {
			return fmt.Errorf("plugin %q scrapes %q; the simulated vLLM servers serve their metrics at /metrics", m.Name, m.Path)
		}
	}
	for _, x := range req.Extractors {
		if err := checkExtractor(x, podLabels); err != nil {
			return err
		}
	}
	return nil
}

// specKeys are an engine mapping's series keys, in the order they are reported.
var specKeys = []string{"queuedRequestsSpec", "runningRequestsSpec", "kvUsageSpec", "loraSpec", "cacheInfoSpec", "cacheBlockSizeSpec", "cacheNumBlocksSpec"}

func checkExtractor(x embed.MetricsExtractor, podLabels []map[string]string) error {
	labelKey := cmp.Or(x.EngineLabelKey, "llm-d.ai/engine-type")
	engines := map[string]bool{cmp.Or(x.DefaultEngine, "vllm"): true}
	for _, labels := range podLabels {
		if e := labels[labelKey]; e != "" {
			engines[e] = true
		}
	}
	names := make([]string, 0, len(engines))
	for e := range engines {
		names = append(names, e)
	}
	sort.Strings(names)
	for _, engine := range names {
		var mapping *embed.EngineMapping
		for i := range x.Engines {
			if x.Engines[i].Name == engine {
				mapping = &x.Engines[i] // the last mapping for an engine wins, as in the router
			}
		}
		if mapping == nil {
			if engine != "vllm" {
				return fmt.Errorf("plugin %q reads the %q engine's metrics (pod label %s or defaultEngine); the simulated engines publish vLLM's series (%s)",
					x.Name, engine, labelKey, strings.Join(publishedSeries, ", "))
			}
			continue // the built-in vLLM mapping
		}
		var missing []string
		for _, key := range specKeys {
			if s := mapping.Series[key]; s != "" && !isPublished(s) {
				missing = append(missing, fmt.Sprintf("%s %q", key, s))
			}
		}
		for _, c := range mapping.Custom {
			if !c.Optional && c.Series != "" && !isPublished(c.Series) {
				missing = append(missing, fmt.Sprintf("custom metric %q", c.Series))
			}
		}
		if len(missing) > 0 {
			return fmt.Errorf("plugin %q requires engine series the simulated engines do not publish: %s (published: %s)",
				x.Name, strings.Join(missing, ", "), strings.Join(publishedSeries, ", "))
		}
	}
	return nil
}

// checkTargetPorts refuses a pool whose pods each serve several ports. Several target ports make
// each pod several endpoints (data-parallel ranks behind one pod). BLIS simulates each rank as its
// own instance and pod (DP placement), so the ports would name endpoints no engine serves.
func checkTargetPorts(pool embed.Pool) error {
	if len(pool.TargetPorts) > 1 {
		return fmt.Errorf("the pool has %d target ports %v; BLIS simulates each data-parallel rank as its own pod "+
			"with one port, so give the pool a single target port", len(pool.TargetPorts), pool.TargetPorts)
	}
	return nil
}

// validateTopology refuses a deployment whose pools the router configuration cannot serve as
// that deployment: a P/D deployment needs a disaggregating configuration, or the router would
// treat the prefill and decode pods as one flat pool.
func validateTopology(req embed.Requirements, host cluster.GatewayHost) error {
	var prefill, decode bool
	for _, inst := range host.Instances() {
		role := host.PoolRole(inst.ID())
		prefill = prefill || role.Has(cluster.PoolRolePrefill)
		decode = decode || role.Has(cluster.PoolRoleDecode)
	}
	if !prefill || !decode || req.Disaggregates {
		return nil
	}
	return errors.New("the deployment has prefill and decode pools, but the router configuration has no " +
		"disagg-profile-handler, so the router would serve every request from one flat pool; state a " +
		"disaggregating configuration in the scenario's router block (endpoint_picker_config; for example examples/llm-d/pd.yaml)")
}

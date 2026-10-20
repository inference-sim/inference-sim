package llmd

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/spf13/pflag"
	"sigs.k8s.io/yaml"

	"github.com/llm-d/llm-d-router/cmd/epp/runner"
	"github.com/llm-d/llm-d-router/pkg/epp/config"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// topologyConfig admits only the single-EPP, default-label form that BLIS
// can feed from its synthetic Pods. In particular, a peer header requires a
// coordinator that this embedded gateway does not run. It verifies static
// producer references; a filter used before peer selection still follows the
// real router's intentional fail-open semantics and is not reinterpreted here.
func topologyConfig(raw []byte) (bool, error) {
	var doc struct {
		Plugins []struct {
			Type       string         `json:"type"`
			Name       string         `json:"name"`
			Parameters map[string]any `json:"parameters"`
		} `json:"plugins"`
	}
	if err := yaml.Unmarshal(raw, &doc); err != nil {
		return false, nil // the router's own loader reports malformed config
	}
	extractors := map[string]bool{}
	type consumerRef struct{ plugin, producer string }
	var consumers []consumerRef
	for _, p := range doc.Plugins {
		switch p.Type {
		case "topology-extractor":
			extractors[cmp.Or(p.Name, p.Type)] = true
			for _, field := range []struct{ key, expected string }{
				{"hostname", "kubernetes.io/hostname"},
				{"rack", "topology.kubernetes.io/rack"},
				{"zone", "topology.kubernetes.io/zone"},
				{"region", "topology.kubernetes.io/region"},
			} {
				key, expected := field.key, field.expected
				if value, _ := p.Parameters[key].(string); value != "" && value != expected {
					return false, fmt.Errorf("topology-extractor %s label %q is not supported; synthetic Pods use %q", key, value, expected)
				}
			}
		case "topology-affinity-filter", "topology-affinity-scorer":
			producer, _ := p.Parameters["topologyProducerName"].(string)
			consumers = append(consumers, consumerRef{p.Type, cmp.Or(producer, "topology-extractor")})
			if header, _ := p.Parameters["peerTopologyHeader"].(string); header != "" {
				return false, fmt.Errorf("%s peerTopologyHeader requires a coordinator; BLIS embeds one EPP", p.Type)
			}
		}
	}
	for _, c := range consumers {
		if !extractors[c.producer] {
			return false, fmt.Errorf("%s needs topology-extractor %q; without it affinity silently fails open", c.plugin, c.producer)
		}
	}
	return len(extractors) > 0 || len(consumers) > 0, nil
}

// supported lists the router plugin types a simulation runs faithfully. Each one was audited in
// the fork: every clock read, background loop, blocking wait and random draw it makes goes
// through the hosting runtime, and every signal it consumes is produced by the simulated
// engines. A configuration that instantiates any other type is refused at startup with the
// reason from unsupported (or a generic one): the simulation never runs a plugin it cannot run
// as llm-d does.
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

// Unsupported gives the reason a plugin type cannot be simulated faithfully.
var unsupported = map[string]string{
	"burst-prefix-cache-producer":               "it blocks a request on a sealing timer the hosting runtime does not model",
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
	"p2p-source-producer":                       "peer KV pulls are not modeled by the simulated engines",
	"selective-kv-policy":                       "peer KV pulls are not modeled by the simulated engines",
	"dcgm-data-source":                          "the simulated engines publish no DCGM GPU telemetry",
	"dcgm-extractor":                            "the simulated engines publish no DCGM GPU telemetry",
	"k8s-notification-source":                   "it needs a Kubernetes controller manager",
	"k8s-peer-discovery":                        "it needs a Kubernetes API server",
	"file-discovery":                            "endpoints come from the simulated deployment, not from a discovery file",
	"multicluster-file-discovery":               "multi-cluster routing is not simulated",
	"disaggregatedset-rollout-screener":         "its revision split draws from process-wide randomness",
	"topology-stamp-handler":                    "coordinator and response-header topology propagation is not modeled",
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
func validateEmbedded(cfg *config.Config, handle fwkplugin.Handle) error {
	var problems []string
	for name, typ := range embeddedPlugins(cfg, handle) {
		if supported[typ] {
			continue
		}
		reason := unsupported[typ]
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

// embeddedPlugins returns every plugin the embedded router holds, by name, with its type: the
// handle's named plugins and every plugin the loaded configuration references (a flow-control
// policy or saturation detector named by a ref, a data source, ...), found by walking the
// configuration, so that a plugin reached by any path is validated.
func embeddedPlugins(cfg *config.Config, handle fwkplugin.Handle) map[string]string {
	out := map[string]string{}
	for name, p := range handle.GetAllPluginsWithNames() {
		out[name] = p.TypedName().Type
	}
	pluginType := reflect.TypeFor[fwkplugin.Plugin]()
	seen := map[uintptr]bool{}
	var walk func(v reflect.Value)
	walk = func(v reflect.Value) {
		if !v.IsValid() {
			return
		}
		if v.Kind() == reflect.Pointer || v.Kind() == reflect.Map || v.Kind() == reflect.Slice {
			if v.IsNil() {
				return
			}
			if v.Kind() != reflect.Slice {
				if seen[v.Pointer()] {
					return
				}
				seen[v.Pointer()] = true
			}
		}
		if v.Kind() == reflect.Interface {
			if v.IsNil() {
				return
			}
			v = v.Elem()
		}
		if v.CanInterface() && v.Type().Implements(pluginType) {
			if p, ok := v.Interface().(fwkplugin.Plugin); ok && (v.Kind() != reflect.Pointer || !v.IsNil()) {
				tn := p.TypedName()
				out[cmp.Or(tn.Name, tn.Type)] = tn.Type
				return // a plugin's internals are its own
			}
		}
		switch v.Kind() {
		case reflect.Pointer:
			walk(v.Elem())
		case reflect.Struct:
			for i := range v.NumField() {
				if v.Type().Field(i).IsExported() {
					walk(v.Field(i))
				}
			}
		case reflect.Slice, reflect.Array:
			for i := range v.Len() {
				walk(v.Index(i))
			}
		case reflect.Map:
			for it := v.MapRange(); it.Next(); {
				walk(it.Value())
			}
		}
	}
	walk(reflect.ValueOf(cfg))
	return out
}

// checkRawConfig refuses, before the router is built, a configuration the simulation cannot host
// as configured:
//   - a core-metrics-extractor that reads, for some pod, an engine's series the simulated engines
//     do not publish: the extractor picks each pod's mapping by its engine label
//     (engineLabelKey, default llm-d.ai/engine-type) or else defaultEngine (default vllm), and
//     the simulated engines publish vLLM's series. The router's default configuration has the
//     default extractor;
//   - a token-producer whose vLLM backend discovers endpoints (vllm.endpointDiscovery): it probes
//     pods on its own schedule, outside the hosting runtime;
//   - a metrics-data-source scraping another path than vLLM's /metrics, which would read nothing;
//   - a precise-prefix-cache-producer whose block size differs from the engines' (its keys would
//     never match the KV events), or whose index the simulation cannot host (checkPrecisePrefix).
func checkRawConfig(config []byte, podLabels []map[string]string, engineBlockSize int) error {
	var doc struct {
		Plugins []struct {
			Type       string         `json:"type"`
			Name       string         `json:"name"`
			Parameters map[string]any `json:"parameters"`
		} `json:"plugins"`
	}
	if len(config) > 0 {
		if err := yaml.Unmarshal(config, &doc); err != nil {
			return nil // the router's own loader reports a malformed configuration
		}
	}
	extractors := 0
	for _, p := range doc.Plugins {
		name := cmp.Or(p.Name, p.Type)
		switch p.Type {
		case "token-producer":
			if v, _ := p.Parameters["vllm"].(map[string]any); v != nil && enabled(v["endpointDiscovery"]) {
				return fmt.Errorf("plugin %q discovers render endpoints (vllm.endpointDiscovery), probing pods on its own "+
					"schedule outside the simulation; give its vLLM backend a fixed url, or use the estimate backend", name)
			}
		case "precise-prefix-cache-producer":
			if err := checkPrecisePrefix(name, p.Parameters, engineBlockSize); err != nil {
				return err
			}
		case "metrics-data-source":
			if path, _ := p.Parameters["path"].(string); path != "" && path != "/metrics" {
				return fmt.Errorf("plugin %q scrapes %q; the simulated vLLM servers serve their metrics at /metrics", name, path)
			}
		case "core-metrics-extractor":
			extractors++
			if err := checkExtractor(name, p.Parameters, podLabels); err != nil {
				return err
			}
		}
	}
	if extractors == 0 { // the default data layer's extractor
		return checkExtractor("core-metrics-extractor", nil, podLabels)
	}
	return nil
}

// checkPrecisePrefix refuses a precise-prefix-cache-producer the simulation cannot run as
// configured. The router's block keys must be cut at the engines' block size, or the KV events the
// engines publish would never match a request's keys. A Redis index reaches the network; the
// cost-aware index ranges a sync.Map, so its order is not reproducible.
func checkPrecisePrefix(name string, params map[string]any, engineBlockSize int) error {
	blockSize := 16 // kvblock.TokenProcessorConfig's default
	if tp, _ := params["tokenProcessorConfig"].(map[string]any); tp != nil {
		for _, key := range []string{"blockSize", "blockSizeTokens"} { // blockSize is the deprecated spelling
			if v, ok := tp[key].(float64); ok && v > 0 {
				blockSize = int(v)
			}
		}
	}
	if engineBlockSize > 0 && blockSize != engineBlockSize {
		return fmt.Errorf("plugin %q cuts block keys every %d tokens, but the simulated engines' KV block size is %d; "+
			"set tokenProcessorConfig.blockSizeTokens to %d", name, blockSize, engineBlockSize, engineBlockSize)
	}
	indexer, _ := params["indexerConfig"].(map[string]any)
	index, _ := indexer["kvBlockIndexConfig"].(map[string]any)
	if enabled(index["redisConfig"]) {
		return fmt.Errorf("plugin %q keeps its index in Redis, an external service", name)
	}
	if enabled(index["costAwareMemoryConfig"]) {
		return fmt.Errorf("plugin %q uses the cost-aware index, whose lookups range a sync.Map in no fixed order", name)
	}
	if events, _ := params["kvEventsConfig"].(map[string]any); events != nil {
		if engine, _ := events["engineType"].(string); engine != "" && engine != "vllm" {
			return fmt.Errorf("plugin %q reads %s KV events; the simulated engines publish vLLM's", name, engine)
		}
	}
	return nil
}

// enabled reports whether a configuration value switches a feature on: true, or a non-empty
// block of settings.
func enabled(v any) bool {
	switch v := v.(type) {
	case bool:
		return v
	case map[string]any:
		return len(v) > 0
	case nil:
		return false
	default:
		return true
	}
}

func checkExtractor(name string, params map[string]any, podLabels []map[string]string) error {
	labelKey, _ := params["engineLabelKey"].(string)
	labelKey = cmp.Or(labelKey, "llm-d.ai/engine-type")
	defaultEngine, _ := params["defaultEngine"].(string)
	engines := map[string]bool{cmp.Or(defaultEngine, "vllm"): true}
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
		var specs map[string]any
		if list, ok := params["engineConfigs"].([]any); ok {
			for _, ec := range list {
				if m, ok := ec.(map[string]any); ok && m["name"] == engine {
					specs = m
				}
			}
		}
		if specs == nil {
			if engine != "vllm" {
				return fmt.Errorf("plugin %q reads the %q engine's metrics (pod label %s or defaultEngine); the simulated engines publish vLLM's series (%s)",
					name, engine, labelKey, strings.Join(publishedSeries, ", "))
			}
			continue // the built-in vLLM mapping
		}
		var missing []string
		for _, key := range []string{"queuedRequestsSpec", "runningRequestsSpec", "kvUsageSpec", "loraSpec", "cacheInfoSpec", "cacheBlockSizeSpec", "cacheNumBlocksSpec"} {
			if s, _ := specs[key].(string); s != "" && !isPublished(s) {
				missing = append(missing, fmt.Sprintf("%s %q", key, s))
			}
		}
		if custom, ok := specs["customMetrics"].([]any); ok {
			for _, c := range custom {
				m, _ := c.(map[string]any)
				s, _ := m["metricSpec"].(string)
				if optional, _ := m["optional"].(bool); !optional && s != "" && !isPublished(s) {
					missing = append(missing, fmt.Sprintf("custom metric %q", s))
				}
			}
		}
		if len(missing) > 0 {
			return fmt.Errorf("plugin %q requires engine series the simulated engines do not publish: %s (published: %s)",
				name, strings.Join(missing, ", "), strings.Join(publishedSeries, ", "))
		}
	}
	return nil
}

// refuseUnsupportedFactories replaces the factory of every registered plugin type the simulation
// does not support with one that refuses it, with the reason. The router instantiates every plugin
// through its factory, whether the configuration names it under plugins:, reaches it only through
// a reference (a flow-control fairnessPolicyRef, ...), or adds it as a default, so an unsupported
// plugin is refused before its real constructor runs: before it starts a goroutine, a timer or any
// external work. Runs as runner.EmbedOptions.AfterRegister, under embedMu.
func refuseUnsupportedFactories() error {
	for typ := range fwkplugin.Registry {
		if supported[typ] {
			continue
		}
		reason := cmp.Or(unsupported[typ], "it has not been audited for the hosting runtime")
		fwkplugin.Registry[typ] = func(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
			return nil, fmt.Errorf("the router configuration cannot be simulated faithfully: plugin %q (type %s): %s", name, typ, reason)
		}
	}
	return nil
}

// IsSupported reports whether a simulation runs router plugins of type typ.
func IsSupported(typ string) bool { return supported[typ] }

// UnsupportedReason is why plugins of type typ are refused, or "" for a type with no recorded
// reason (refused as not audited) or a supported one.
func UnsupportedReason(typ string) string { return unsupported[typ] }

// validateTopology refuses a deployment whose pools the router configuration cannot serve as
// that deployment: a P/D deployment needs a disaggregating profile handler, or the router would
// treat the prefill and decode pods as one flat pool.
func validateTopology(e *runner.Embedded, host cluster.GatewayHost) error {
	var prefill, decode bool
	for _, inst := range host.Instances() {
		role := host.PoolRole(inst.ID())
		prefill = prefill || role.Has(cluster.PoolRolePrefill)
		decode = decode || role.Has(cluster.PoolRoleDecode)
	}
	if !prefill || !decode {
		return nil
	}
	for _, p := range e.Handle.GetAllPluginsWithNames() {
		if p.TypedName().Type == "disagg-profile-handler" {
			return nil
		}
	}
	return errors.New("the deployment has prefill and decode pools, but the router configuration has no " +
		"disagg-profile-handler, so the router would serve every request from one flat pool; state a " +
		"disaggregating configuration in the scenario's router block (endpoint_picker_config; for example examples/llm-d/pd.yaml)")
}

// inertEPPFlags are router flags that change how a deployed router is served or observed
// (transport, TLS, replicas, shutdown, telemetry), never what it decides. A simulation has none of
// that, so they are accepted and have no effect; a run that sets one says so on stderr.
var inertEPPFlags = map[string]bool{
	"ha-enable-leader-election": true, "ha-lease-name": true, "ha-lease-duration": true, "ha-renew-deadline": true,
	"ha-retry-period": true, "drain-timeout": true, "grpc-max-recv-msg-size": true, "grpc-max-send-msg-size": true,
	"grpc-port": true, "grpc-health-port": true, "metrics-port": true, "emit-endpoint-scores": true,
	"refresh-prometheus-metrics-interval": true, "tracing": true, "health-checking": true, "enable-pprof": true,
	"cert-path": true, "enable-cert-reload": true, "enable-grpc-stream-metrics": true, "fairness-id-metric-label-limit": true,
	"secure-serving": true, "tls-min-version": true, "metrics-endpoint-auth": true, "metrics-client-ca-file": true,
	"metrics-cert-dir": true, "tls-cipher-suites": true,
}

// legacyMetricFlags name the engine series the router reads when its configuration does not.
var legacyMetricFlags = []string{
	"total-queued-requests-metric", "total-running-requests-metric", "kv-cache-usage-percentage-metric",
	"lora-info-metric", "cache-info-metric",
}

// checkEPPFlags refuses router flags that name an engine series the simulated engines do not
// publish, and returns the flags that have no effect in a simulation.
func checkEPPFlags(args []string) (inert []string, err error) {
	opts := runserver.NewOptions()
	fs := pflag.NewFlagSet("epp", pflag.ContinueOnError)
	opts.AddFlags(fs)
	if err := fs.Parse(args); err != nil {
		return nil, err // ParseEPPFlags reports it
	}
	for _, name := range legacyMetricFlags {
		if f := fs.Lookup(name); f != nil && fs.Changed(name) && f.Value.String() != "" && !isPublished(f.Value.String()) {
			return nil, fmt.Errorf("EPP flag --%s=%s names an engine series the simulated engines do not publish (%s)",
				name, f.Value.String(), strings.Join(publishedSeries, ", "))
		}
	}
	fs.Visit(func(f *pflag.Flag) {
		if inertEPPFlags[f.Name] {
			inert = append(inert, "--"+f.Name)
		}
	})
	sort.Strings(inert)
	return inert, nil
}

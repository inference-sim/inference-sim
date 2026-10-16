package sim

import (
	"path/filepath"
	"strings"
	"testing"
)

func mustIngest(t *testing.T, path string, v EPPVersion) *EPPIngest {
	t.Helper()
	in, err := LoadEndpointPickerConfig(path, 64, v)
	if err != nil {
		t.Fatalf("%s (%s): %v", path, v, err)
	}
	return in
}

func plugin(in *EPPIngest, name string) EPPPluginRecord {
	for _, p := range in.Provenance.Plugins {
		if p.Name == name {
			return p
		}
	}
	return EPPPluginRecord{}
}

// Every fixture — the three deployed routers and six llm-d guide configs —
// ingests under each llm-d version that serves its apiVersion (v1 exists only
// after v0.10.0: apix/config/v1 is absent at the v0.10.0 tag).
func TestEPPConfig_AllFixturesIngest(t *testing.T) {
	cb, _ := filepath.Glob("testdata/epp/*.yaml")
	guides, _ := filepath.Glob("testdata/epp/llm-d-guides/*.yaml")
	if len(cb) != 3 || len(guides) != 6 {
		t.Fatalf("fixtures: %d deployed, %d guides", len(cb), len(guides))
	}
	for _, f := range cb {
		mustIngest(t, f, EPPVersionMain)
		mustIngest(t, f, EPPVersionV0_10_0)
	}
	for _, f := range guides {
		mustIngest(t, f, EPPVersionMain)
		if _, err := LoadEndpointPickerConfig(f, 64, EPPVersionV0_10_0); err == nil || !strings.Contains(err.Error(), "not served by llm-d-router v0.10.0") {
			t.Errorf("%s under v0.10.0: want the apiVersion refusal, got %v", f, err)
		}
	}
}

func TestEPPConfig_DeployedOptimizedBaseline(t *testing.T) {
	for _, v := range []EPPVersion{EPPVersionMain, EPPVersionV0_10_0} {
		in := mustIngest(t, "testdata/epp/optimized-baseline-cpu-tier.yaml", v)
		if in.Disaggregated || len(in.Scorers) != 2 || in.Scorers[0].Name != "token-load" || in.Scorers[1].Name != "prefix-cache" ||
			in.Scorers[1].Params.PrefixMatchInfoProducerName != "cpu-prefix-cache-producer" {
			t.Fatalf("%s: scorers %+v", v, in.Scorers)
		}
		f := in.Pipeline.Filters
		if len(f) != 2 || f[0].Type != LLMDConcurrencyDetector || f[1].Type != LLMDPrefixAffinityFilter {
			t.Fatalf("%s: filters %+v (order matters: guard first)", v, f)
		}
		g := f[0].Concurrency
		if g.MaxConcurrency != 256 || g.Headroom != 0 || g.Mode != ConcurrencyModeRequests || g.FailOpen != (v == EPPVersionMain) {
			t.Errorf("%s: guard %+v", v, g)
		}
		a := f[1].Affinity
		if a.AffinityThreshold != 0.5 || a.ExplorationProbability != 0 || a.MaxTTFTPenaltyMs != 4500 || a.PeakPrefillThroughput != 15000 {
			t.Errorf("%s: affinity %+v", v, a)
		}
		var cpu PrefixProducerConfig
		for _, p := range in.Pipeline.Producers {
			if p.Name == "cpu-prefix-cache-producer" {
				cpu = p
			}
		}
		// llm-d: an ordinary approx producer, GPU-sized under autoTune (makeserver).
		if cpu.CPUTier || !cpu.AutoTune || len(in.Provenance.Deviations) != 0 {
			t.Errorf("%s: cpu producer must be llm-d's LRU estimate by default: %+v, deviations %v", v, cpu, in.Provenance.Deviations)
		}
		if len(in.Pipeline.InFlightLoads) != 1 || in.Pipeline.InFlightLoads[0].PrefixMatchInfoProducerName != LLMDApproxPrefixProducer {
			t.Errorf("%s: inflight %+v", v, in.Pipeline.InFlightLoads)
		}
		fc := in.FlowControl
		if fc == nil || fc.MaxRequests != 4096 || fc.MaxBytes != 4<<30 || fc.RequestTTLUs != 300_000_000 || fc.Detector.MaxConcurrency != 256 {
			t.Errorf("%s: flow control %+v", v, fc)
		}
		if in.Provenance.Profile.Picker != "max-score-picker" || in.Provenance.ProfileHandler != "single-profile-handler" {
			t.Errorf("%s: injected defaults missing: %+v", v, in.Provenance.Profile)
		}
	}
}

func TestEPPConfig_DeployedPreciseGPU(t *testing.T) {
	for _, v := range []EPPVersion{EPPVersionMain, EPPVersionV0_10_0} {
		in := mustIngest(t, "testdata/epp/precise-gpu.yaml", v)
		p := in.Pipeline.Producers[0]
		if p.Type != LLMDPrecisePrefixProducer || p.BlockSizeTokens != 64 || p.Speculative || p.TierWeights["gpu"] != 1 {
			t.Errorf("%s: precise producer %+v", v, p)
		}
		// kvCacheBackendConfigs lists gpu only: an unlisted tier weighs 1.0 on v0.10.0, 0 on main.
		if want := map[EPPVersion]float64{EPPVersionMain: 0, EPPVersionV0_10_0: 1}[v]; weightOf(p, "cpu") != want {
			t.Errorf("%s: cpu tier weight %v, want %v", v, weightOf(p, "cpu"), want)
		}
		joined := strings.Join(in.Provenance.Inert, "\n")
		for _, want := range []string{`token-producer "token-producer"`, `endpoint-notification-source`, "dataLayer", "kvEventsConfig"} {
			if !strings.Contains(joined, want) {
				t.Errorf("%s: inert record missing %q:\n%s", v, want, joined)
			}
		}
	}
}

func TestEPPConfig_DeployedPD(t *testing.T) {
	in := mustIngest(t, "testdata/epp/pd-1p1d.yaml", EPPVersionMain)
	if !in.Disaggregated || in.PDDecider != "always" {
		t.Fatalf("P/D: disagg=%v decider=%q", in.Disaggregated, in.PDDecider)
	}
	want := []struct {
		name string
		w    float64
	}{{"prefix-cache", 5}, {"prefix-cache", 2}, {"active-requests", 1}}
	if len(in.PrefillScorers) != 3 {
		t.Fatalf("prefill scorers %+v", in.PrefillScorers)
	}
	for i, w := range want {
		if in.PrefillScorers[i].Name != w.name || in.PrefillScorers[i].Weight != w.w {
			t.Errorf("prefill scorer %d: %+v, want %s w%v", i, in.PrefillScorers[i], w.name, w.w)
		}
	}
	if in.PrefillScorers[0].Params.PrefixMatchInfoProducerName != "gpu-prefix-cache-producer" || in.PrefillScorers[1].Params.PrefixMatchInfoProducerName != "cpu-prefix-cache-producer" {
		t.Errorf("producer bindings %+v", in.PrefillScorers)
	}
	if len(in.DecodeScorers) != 1 || in.DecodeScorers[0].Name != "active-requests" || in.DecodeScorers[0].Weight != 3 {
		t.Errorf("decode scorers %+v", in.DecodeScorers)
	}
	for _, p := range in.Pipeline.Producers {
		if p.Name == "cpu-prefix-cache-producer" && (p.AutoTune || p.LRUCapacityPerServer != 30659) {
			t.Errorf("cpu producer %+v", p)
		}
	}
	if plugin(in, "queue-scorer").Role != "scorer" || in.FlowControl != nil {
		t.Errorf("unreferenced queue-scorer / no flow-control gate: %+v", in.Provenance.Plugins)
	}
}

func TestEPPConfig_LLMDGuides(t *testing.T) {
	ob := mustIngest(t, "testdata/epp/llm-d-guides/optimized-baseline.yaml", EPPVersionMain)
	// Unnamed defaults: the filter's defaults (0.80, 18000, 15928) and auto-bound producers.
	if a := ob.Pipeline.Filters[0].Affinity; a.AffinityThreshold != 0.8 || a.MaxTTFTPenaltyMs != 18000 || a.PeakPrefillThroughput != 15928 {
		t.Errorf("optimized-baseline affinity defaults %+v", a)
	}
	pr := mustIngest(t, "testdata/epp/llm-d-guides/precise-prefix-cache-routing.yaml", EPPVersionMain)
	if p := pr.Pipeline.Producers[0]; !p.Speculative || p.SpeculativeTTLUs != 2_000_000 || p.TierWeights["cpu"] != 0.8 {
		t.Errorf("precise guide producer %+v", p)
	}
	tier := mustIngest(t, "testdata/epp/llm-d-guides/tiered-prefix-cache-cpu.yaml", EPPVersionMain)
	if sc := tier.Scorers[4]; sc.Name != "no-hit-lru" || !sc.Params.ProducerBound || sc.Params.PrefixMatchInfoProducerName != "gpu-prefix-cache-producer" {
		t.Errorf("tiered guide no-hit-lru %+v", sc)
	}
	fc := mustIngest(t, "testdata/epp/llm-d-guides/flow-control.yaml", EPPVersionMain)
	if f := fc.FlowControl; f == nil || f.MaxRequests != 1000 || f.MaxBytes != 10<<30 || len(f.Bands) != 3 ||
		f.Bands[0] != (EPPPriorityBand{Priority: 100, MaxRequests: 500, MaxBytes: 1_000_000_000, Fairness: "round-robin"}) {
		t.Errorf("flow-control guide %+v", fc.FlowControl)
	}
	we := mustIngest(t, "testdata/epp/llm-d-guides/wide-ep.yaml", EPPVersionMain)
	if !strings.Contains(strings.Join(we.Provenance.Inert, "\n"), "disaggregatedset-rollout-screener") {
		t.Errorf("wide-ep screener not recorded inert: %v", we.Provenance.Inert)
	}
	if plugin(we, LLMDApproxPrefixProducer).AutoCreated != true {
		t.Errorf("wide-ep: prefix-cache-scorer without a producer must auto-create the default approx producer")
	}
}

const eppMinimal = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
%s
- type: queue-scorer
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: queue-scorer
%s
`

func eppDoc(plugins, refs string) []byte {
	return []byte(strings.Replace(strings.Replace(eppMinimal, "%s", plugins, 1), "%s", refs, 1))
}

// Refusals: unmodeled scheduling plugins and parameters are refused by name, never
// ignored; version-specific fields are refused under the release that lacks them.
func TestEPPConfig_Refusals(t *testing.T) {
	cases := []struct {
		name, plugins, refs string
		v                   EPPVersion
		want                string
	}{
		{"unknown scheduling plugin", "- type: session-affinity-scorer", "  - pluginRef: session-affinity-scorer", EPPVersionMain, `type "session-affinity-scorer" is not supported`},
		{"latency scorer", "- type: latency-scorer", "", EPPVersionMain, `"latency-scorer" is not supported`},
		{"strict parameters", "- type: token-load-scorer\n  parameters: {queueThresholdToken: 5}", "", EPPVersionMain, "unknown field"},
		{"params on a defaults-only scorer", "- type: active-request-scorer\n  parameters: {idleThreshold: 3}", "", EPPVersionMain, "parameters are not supported"},
		{"failOpen before main", "- type: concurrency-detector\n  parameters: {failOpen: false}", "", EPPVersionV0_10_0, "failOpen does not exist"},
		{"outputRatio after v0.10.0", "- type: inflight-load-producer\n  parameters: {outputRatio: 2}", "", EPPVersionMain, "outputRatio was removed"},
		{"latency predictor source", "- type: prefix-cache-affinity-filter\n  parameters: {ttftSource: latencyPredictor}", "", EPPVersionMain, "latencyPredictor is not supported"},
		{"negative threshold on main", "- type: prefix-cache-affinity-filter\n  parameters: {affinityThreshold: -0.1}", "", EPPVersionMain, "affinityThreshold must be in [0, 1]"},
		{"precise block-size mismatch", "- type: precise-prefix-cache-producer\n  parameters: {tokenProcessorConfig: {blockSizeTokens: 16}}", "", EPPVersionMain, "differs from the engine block size 64"},
		{"multi-endpoint picker", "- type: max-score-picker\n  parameters: {maxNumOfEndpoints: 2}", "", EPPVersionMain, "maxNumOfEndpoints 2 > 1"},
		{"zero weight", "", "    weight: 0", EPPVersionMain, "weight must be a finite positive number"},
		{"role filter outside P/D", "- type: prefill-filter", "  - pluginRef: prefill-filter", EPPVersionMain, "role filters are modeled as the P/D pool"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseEndpointPickerConfig(eppDoc(tc.plugins, tc.refs), 64, tc.v)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("want error containing %q, got %v", tc.want, err)
			}
		})
	}
}

// A disagg-profile-handler with prefix-based-pd-decider maps onto the llm-d decider;
// prefill-first is refused (BLIS routes decode first).
func TestEPPConfig_PrefixBasedDeciderAndStageOrder(t *testing.T) {
	doc := `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: prefix-based-pd-decider
  parameters: {nonCachedTokens: 16, promptTokens: 4}
- type: disagg-profile-handler
  parameters:
    deciders: {prefill: prefix-based-pd-decider}
%s
- type: prefill-filter
- type: decode-filter
- type: queue-scorer
schedulingProfiles:
- name: prefill
  plugins: [{pluginRef: prefill-filter}, {pluginRef: queue-scorer}]
- name: decode
  plugins: [{pluginRef: decode-filter}, {pluginRef: queue-scorer}]
`
	in, err := ParseEndpointPickerConfig([]byte(strings.Replace(doc, "%s", "", 1)), 64, EPPVersionMain)
	if err != nil {
		t.Fatal(err)
	}
	if in.PDDecider != "prefix-based" || in.PDPrefixBased == nil || in.PDPrefixBased.NonCachedTokens != 16 || in.PDPrefixBased.PromptTokens != 4 {
		t.Errorf("decider %q %+v", in.PDDecider, in.PDPrefixBased)
	}
	if plugin(in, LLMDApproxPrefixProducer).AutoCreated != true {
		t.Error("the decider's default producer binding must auto-create the approx producer")
	}
	prefillFirst := strings.Replace(doc, "    deciders: {prefill: prefix-based-pd-decider}\n%s", "    stageOrder: prefill-first", 1)
	if _, err := ParseEndpointPickerConfig([]byte(prefillFirst), 64, EPPVersionMain); err == nil || !strings.Contains(err.Error(), "prefill-first is not supported") {
		t.Errorf("prefill-first: %v", err)
	}
}

// Resource quantities follow resource.Quantity (Value rounds up).
func TestParseQuantity(t *testing.T) {
	for in, want := range map[interface{}]int64{"4Gi": 4 << 30, "1k": 1000, "10Gi": 10 << 30, 4096: 4096, "500": 500, "1.5k": 1500, "0.5": 1} {
		got, err := parseQuantity(in, "q")
		if err != nil || got != want {
			t.Errorf("%v: got %d (%v), want %d", in, got, err, want)
		}
	}
	if _, err := parseQuantity("-1", "q"); err == nil {
		t.Error("negative quantity must be refused")
	}
}

// The live CPU-tier view is an explicit BLIS opt-in, recorded as a deviation; it
// must name an approx producer of the config.
func TestEPPConfig_CPUTierViewIsAnOptIn(t *testing.T) {
	in, err := LoadEndpointPickerConfig("testdata/epp/optimized-baseline-cpu-tier.yaml", 64, EPPVersionV0_10_0, WithCPUTierView("cpu-prefix-cache-producer"))
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range in.Pipeline.Producers {
		if (p.Name == "cpu-prefix-cache-producer") != p.CPUTier {
			t.Errorf("%s: CPUTier=%v", p.Name, p.CPUTier)
		}
	}
	if len(in.Provenance.Deviations) != 1 || !strings.Contains(in.Provenance.Deviations[0], "cpu-prefix-cache-producer") {
		t.Errorf("deviation not recorded: %v", in.Provenance.Deviations)
	}
	if _, err := LoadEndpointPickerConfig("testdata/epp/optimized-baseline-cpu-tier.yaml", 64, EPPVersionMain, WithCPUTierView("token-load-scorer")); err == nil {
		t.Error("a non-producer name must be refused")
	}
}

// Inert plugins: parameters with no simulation meaning are accepted, any other is
// refused; metrics-data-source interval maps to the snapshot refresh interval and
// static-usage-limit-policy threshold to the uniform band ceiling.
func TestEPPConfig_InertParameters(t *testing.T) {
	ok := eppDoc("- type: metrics-data-source\n  parameters: {scheme: https, interval: 120ms}\n- type: token-producer\n  parameters: {modelName: m, vllm: {url: http://x}}", "")
	in, err := ParseEndpointPickerConfig(ok, 64, EPPVersionMain)
	if err != nil {
		t.Fatal(err)
	}
	if in.MetricsRefreshIntervalUs != 100_000 {
		t.Errorf("120ms rounds to 2 base ticks (100ms): got %d", in.MetricsRefreshIntervalUs)
	}
	for _, bad := range []string{
		"- type: endpoint-notification-source\n  parameters: {debounce: 5}",
		"- type: metrics-data-source\n  parameters: {stalenessThreshold: 1s}",
		"- type: fcfs-ordering-policy\n  parameters: {x: 1}",
	} {
		if _, err := ParseEndpointPickerConfig(eppDoc(bad, ""), 64, EPPVersionMain); err == nil || !strings.Contains(err.Error(), "is not supported by the simulator") {
			t.Errorf("%q: want a refusal, got %v", bad, err)
		}
	}
	fc := `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
featureGates: [flowControl]
plugins:
- type: static-usage-limit-policy
  parameters: {threshold: 0.7}
- type: concurrency-detector
- type: queue-scorer
schedulingProfiles:
- name: default
  plugins: [{pluginRef: queue-scorer}]
flowControl:
  saturationDetector: {pluginRef: concurrency-detector}
`
	in, err = ParseEndpointPickerConfig([]byte(fc), 64, EPPVersionMain)
	if err != nil {
		t.Fatal(err)
	}
	if in.FlowControl.UniformCeiling != 0.7 {
		t.Errorf("static threshold: ceiling %v, want 0.7", in.FlowControl.UniformCeiling)
	}
	// Renamed and unreferenced, the policy is not the default and has no effect.
	renamed := strings.Replace(fc, "- type: static-usage-limit-policy\n", "- type: static-usage-limit-policy\n  name: other\n", 1)
	if in, err = ParseEndpointPickerConfig([]byte(renamed), 64, EPPVersionMain); err != nil || in.FlowControl.UniformCeiling != 1.0 {
		t.Errorf("unreferenced policy: %v %v", err, in.FlowControl)
	}
}

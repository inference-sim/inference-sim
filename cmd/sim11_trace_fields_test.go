package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// sim11Trace is the fixture TestSIM11_TraceFieldsConsumedChangeTheResult toggles: 120 requests
// arriving every 700 µs, long enough that two instances queue, so a field that moves a request
// moves a figure.
func sim11Trace() (*workload.TraceHeader, []workload.TraceRecord) {
	h := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated"}
	var r []workload.TraceRecord
	for i := 0; i < 120; i++ {
		r = append(r, workload.TraceRecord{RequestID: i, ClientID: "c", TenantID: "t", SLOClass: "standard",
			SessionID: fmt.Sprintf("s%d", i), InputTokens: 300 + 37*i, OutputTokens: 20 + i, TextTokens: 300 + 37*i,
			ArrivalTimeUs: int64(i) * 700, SendTimeUs: int64(i) * 700, Status: "ok"})
	}
	return h, r
}

// sim11Pool is the router the pool-mode inputs are toggled under: pool mode, scored by the
// scraped queue and KV-cache metrics with flow control bounded (inv13Config), one InferencePool
// selecting app=vllm and two objectives (sim11Manifests).
func sim11Pool(h *workload.TraceHeader, r []workload.TraceRecord) {
	for i := range r {
		if i%2 == 0 {
			r[i].SLOClass = "critical"
		}
	}
	h.Router = &workload.TraceRouterConfig{EPPFlags: []string{"--pool-name=vllm", "--refresh-metrics-interval=20ms", "--config-text=" + inv13Config},
		Manifests: sim11Manifests("vllm"), PodLabels: map[string]string{"app": "vllm"}}
}

// sim11Manifests is an InferencePool selecting the pods labelled app=<app>, and two objectives.
func sim11Manifests(app string) string {
	return fmt.Sprintf(`apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata:
  name: vllm
spec:
  selector:
    matchLabels:
      app: %s
  targetPorts:
  - number: 8000
  endpointPickerRef:
    name: epp
    port:
      number: 9002
---
apiVersion: llm-d.ai/v1
kind: InferenceObjective
metadata:
  name: standard
spec:
  priority: 1
  poolRefs:
  - name: vllm
---
apiVersion: llm-d.ai/v1
kind: InferenceObjective
metadata:
  name: critical
spec:
  priority: 100
  poolRefs:
  - name: vllm
`, app)
}

// sim11Precise is the router a field the precise prefix producer alone reads is toggled under:
// llm-d's KV-events configuration (testdata/epp/precise.yaml), with prefix groups so that its
// scorer separates the endpoints.
func sim11Precise(t *testing.T, speculative bool) func(*workload.TraceHeader, []workload.TraceRecord) {
	raw, err := os.ReadFile("testdata/epp/precise.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if !speculative {
		// Without speculative indexing the producer knows only what the engines' KV events have
		// told it, so their delay reaches its scores.
		raw = []byte(strings.Replace(string(raw), "speculativeIndexing: true", "speculativeIndexing: false", 1))
	}
	flags := []string{"--endpoint-selector=llm-d.ai/inference-serving=true", "--endpoint-target-ports=8000", "--config-text=" + string(raw)}
	return func(h *workload.TraceHeader, r []workload.TraceRecord) {
		for i := range r {
			r[i].PrefixGroup, r[i].PrefixLength = fmt.Sprint("g", i%3), 256
		}
		h.Router = &workload.TraceRouterConfig{EPPFlags: flags}
	}
}

// TestSIM11_TraceFieldsConsumedChangeTheResult: every trace field the status table calls consumed
// changes some byte of the Result's simulated sections when it changes (SIM-11, #1308). Each
// consumed field is toggled from the fixture's value under the configuration that reads it, and
// the Result of that replay is compared with the Result of the configuration's own baseline. A
// consumed field this test does not toggle fails it, so the table and the test cannot drift.
//
// Not toggled here, each for a stated reason:
//   - Header.KVOffload, and every field of TraceKVOffloadConfig and TraceKVOffloadTier: the block
//     is replayed whole (build.ReconcileReplayKVOffload), and the round trip is pinned by
//     TestKVOffload_EndToEnd_RunReplayRoundTrip and TestINV13_RunReplayParity_Offload.
//   - Header.Mode: consumed only through replayRecordsDeployment (mode real lets
//     --kv-offload-config add a deployment), tested by TestReplayRecordsDeployment.
//   - Header.SessionContextGrowth, Record.ThinkTimeUs, Record.InputTokensReset: consumed by the
//     closed-loop and fixed-accumulate modes, pinned by
//     TestReplayCmd_ClosedLoopAccumulate_FaithfulReExport and TestReplayFixedAccumulate_*.
//   - Record.ImageTokens, AudioTokens, VideoTokens: consumed when an encode pool is deployed
//     (sim/llmd TestEncode_RouterPlacesMultimodalRequestsOnEncoders); no committed scenario has one.
//   - Record.Streaming: the router reads it through its response-header hooks; no configuration the
//     repository ships has a plugin whose decision depends on it (sim/llmd's gateway tests pin
//     the headers).
//   - TraceObservedKVMetrics.Source: a refusal gate, tested by TestValidateObservedKVReplayable.
func TestSIM11_TraceFieldsConsumedChangeTheResult(t *testing.T) {
	if testing.Short() {
		t.Skip("replays the fixture about 30 times (about 20 s)")
	}
	type mutate = func(*workload.TraceHeader, []workload.TraceRecord)
	precise := sim11Precise(t, true)
	eventsOnly := sim11Precise(t, false)
	// The configurations a field is read under, each applied to sim11Trace's fixture; "plain" is
	// the fixture itself.
	bases := map[string]mutate{
		"plain":       func(*workload.TraceHeader, []workload.TraceRecord) {},
		"precise":     precise,
		"events-only": eventsOnly,
		"pool":        sim11Pool,
		"lengths":     func(_ *workload.TraceHeader, r []workload.TraceRecord) { r[0].PrefixLength, r[1].PrefixLength = 64, 64 },
		"goodput": func(h *workload.TraceHeader, _ []workload.TraceRecord) {
			h.GoodputSLOTargets = map[string]workload.SLODimTargets{"critical": {TTFTMs: 5}, "standard": {TTFTMs: 5}}
		},
	}
	// Each consumed field: the configuration it is read under, and the toggle.
	toggles := map[string]struct {
		base string
		set  mutate
	}{
		"TraceHeader.GoodputSLOTargets": {"plain", func(h *workload.TraceHeader, _ []workload.TraceRecord) {
			h.GoodputSLOTargets = map[string]workload.SLODimTargets{"standard": {TTFTMs: 5}}
		}},
		"TraceHeader.Router": {"plain", func(h *workload.TraceHeader, _ []workload.TraceRecord) {
			h.Router = &workload.TraceRouterConfig{GatewayLatencyUs: 500}
		}},
		"TraceRouterConfig.EPPFlags":  {"plain", func(h *workload.TraceHeader, r []workload.TraceRecord) { precise(h, r) }},
		"TraceRouterConfig.Manifests": {"pool", func(h *workload.TraceHeader, _ []workload.TraceRecord) { h.Router.Manifests = sim11Manifests("other") }},
		"TraceRouterConfig.PodLabels": {"pool", func(h *workload.TraceHeader, _ []workload.TraceRecord) {
			h.Router.PodLabels = map[string]string{"app": "other"}
		}},
		"TraceRouterConfig.GatewayLatencyUs": {"precise", func(h *workload.TraceHeader, _ []workload.TraceRecord) { h.Router.GatewayLatencyUs = 500 }},
		"TraceRouterConfig.ScrapeJitterUs":   {"pool", func(h *workload.TraceHeader, _ []workload.TraceRecord) { h.Router.ScrapeJitterUs = 15_000 }},
		"TraceRouterConfig.KVEventDelayUs":   {"events-only", func(h *workload.TraceHeader, _ []workload.TraceRecord) { h.Router.KVEventDelayUs = 400_000 }},
		"TraceRouterConfig.Seed": {"precise", func(h *workload.TraceHeader, _ []workload.TraceRecord) {
			s := int64(1234)
			h.Router.Seed = &s
		}},
		"TraceRecord.TenantID":     {"plain", func(_ *workload.TraceHeader, r []workload.TraceRecord) { r[0].TenantID = "z" }},
		"TraceRecord.SLOClass":     {"goodput", func(_ *workload.TraceHeader, r []workload.TraceRecord) { r[0].SLOClass = "critical" }},
		"TraceRecord.SessionID":    {"plain", func(_ *workload.TraceHeader, r []workload.TraceRecord) { r[0].SessionID = "s1" }},
		"TraceRecord.RoundIndex":   {"plain", func(_ *workload.TraceHeader, r []workload.TraceRecord) { r[0].RoundIndex = 3 }},
		"TraceRecord.PrefixGroup":  {"lengths", func(_ *workload.TraceHeader, r []workload.TraceRecord) { r[0].PrefixGroup, r[1].PrefixGroup = "g", "g" }},
		"TraceRecord.PrefixLength": {"plain", func(_ *workload.TraceHeader, r []workload.TraceRecord) { r[0].PrefixGroup, r[0].PrefixLength = "g", 64 }},
		"TraceRecord.InputTokens":  {"plain", func(_ *workload.TraceHeader, r []workload.TraceRecord) { r[0].InputTokens = 900 }},
		"TraceRecord.OutputTokens": {"plain", func(_ *workload.TraceHeader, r []workload.TraceRecord) { r[0].OutputTokens = 90 }},
		"TraceRecord.Model":        {"plain", func(_ *workload.TraceHeader, r []workload.TraceRecord) { r[0].Model = "other" }},
		"TraceRecord.DeadlineUs":   {"plain", func(_ *workload.TraceHeader, r []workload.TraceRecord) { r[0].DeadlineUs = 100 }},
		"TraceRecord.SLOTargetUs":  {"plain", func(_ *workload.TraceHeader, r []workload.TraceRecord) { r[0].SLOTargetUs = 1000 }},
		// #1308: a server-reported prompt length is the one the simulation uses.
		"TraceRecord.ServerInputTokens": {"plain", func(_ *workload.TraceHeader, r []workload.TraceRecord) { r[0].ServerInputTokens = 900 }},
		"TraceRecord.ArrivalTimeUs": {"plain", func(_ *workload.TraceHeader, r []workload.TraceRecord) {
			r[119].ArrivalTimeUs, r[119].SendTimeUs = 900_000, 900_000
		}},
		"TraceRecord.SendTimeUs": {"plain", func(_ *workload.TraceHeader, r []workload.TraceRecord) { r[119].SendTimeUs = 900_000 }},
		"TraceRecord.Adapter":    {"plain", func(_ *workload.TraceHeader, r []workload.TraceRecord) { r[0].Adapter = "a" }},
	}
	notToggled := map[string]bool{
		"TraceHeader.KVOffload": true, "TraceHeader.Mode": true, "TraceHeader.SessionContextGrowth": true,
		"TraceRecord.ThinkTimeUs": true, "TraceRecord.InputTokensReset": true, "TraceRecord.ImageTokens": true,
		"TraceRecord.AudioTokens": true, "TraceRecord.VideoTokens": true, "TraceRecord.Streaming": true,
		"TraceObservedKVMetrics.Source": true,
	}
	var consumed []string
	for _, typ := range []string{"TraceHeader", "TraceRouterConfig", "TraceKVOffloadConfig", "TraceKVOffloadTier",
		"TraceServerConfig", "TraceNetworkConfig", "TraceObservedKVMetrics", "TraceRecord"} {
		table, _ := workload.TraceFieldStatus(typ)
		for name, e := range table {
			key := typ + "." + name
			if e.Status != workload.FieldConsumed || notToggled[key] ||
				typ == "TraceKVOffloadConfig" || typ == "TraceKVOffloadTier" {
				continue
			}
			consumed = append(consumed, key)
		}
	}
	sort.Strings(consumed)

	catalogDir, defaultsPath := setupKernelTestFixturesWithDefaults(t)
	replay := func(muts ...mutate) string {
		h, r := sim11Trace()
		for _, m := range muts {
			m(h, r)
		}
		dir := t.TempDir()
		hp, dp := filepath.Join(dir, "h.yaml"), filepath.Join(dir, "d.csv")
		if err := workload.ExportTraceV2(h, r, hp, dp); err != nil {
			t.Fatal(err)
		}
		return simulated(t, runReplayCaptureStdout(t, []string{"--trace-header", hp, "--trace-data", dp,
			"--catalog", catalogDir, "--scenario", kernelTestScenario, "--scenarios", kernelScenariosDir(t),
			"--registry", kernelRegistryDir(t), "--defaults-filepath", defaultsPath,
			"--num-instances", "2", "--horizon", "60000000"}))
	}
	baselines := map[string]string{}
	baseline := func(name string, m mutate) string {
		if v, ok := baselines[name]; ok {
			return v
		}
		baselines[name] = replay(m)
		return baselines[name]
	}
	for _, key := range consumed {
		tg, ok := toggles[key]
		if !ok {
			t.Errorf("%s is consumed and this test does not toggle it: add a toggle, or a reason it is not toggled", key)
			continue
		}
		if replay(bases[tg.base], tg.set) == baseline(tg.base, bases[tg.base]) {
			t.Errorf("%s is consumed, but changing it changed no byte of the Result's simulated sections", key)
		}
	}
	for key := range toggles {
		if !slices.Contains(consumed, key) {
			t.Errorf("%s is toggled here, but the status table does not call it consumed", key)
		}
	}
}

// TestReplayRecordsDeployment: the header's mode is consumed in one place. A trace of mode real
// records no deployment, so --kv-offload-config may add one; a generated or replayed trace records
// its deployment, and an agentic replay has no trace header at all.
func TestReplayRecordsDeployment(t *testing.T) {
	for _, c := range []struct {
		mode    string
		agentic bool
		want    bool
	}{{"generated", false, true}, {"replayed", false, true}, {"real", false, false}, {"generated", true, false}} {
		if got := replayRecordsDeployment(workload.TraceHeader{Mode: c.mode}, c.agentic); got != c.want {
			t.Errorf("mode %q, agentic %v: records deployment = %v, want %v", c.mode, c.agentic, got, c.want)
		}
	}
}

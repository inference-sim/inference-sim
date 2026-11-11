package vllmengine

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/load"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/blis-vllm/core/vllm"
	"github.com/inference-sim/blis-vllm/internal/substratedata"
)

// recorder is the optional method a host reads by type assertion on the engine.Engine.
type recorder interface{ RecordSelected(*load.Data) }

// selectedBy builds the pool with a private Data (Input.Data nil), so the open-time
// registration lands in a Data nobody reads, and returns what registering the engine with a
// fresh host Data puts there.
func selectedBy(t *testing.T, scenario string, pool int) ([]load.Selected, engine.Engine) {
	t.Helper()
	e, err := New(Input{Spec: engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, scenario), Pool: pool}})
	if err != nil {
		t.Fatal(err)
	}
	r, ok := e.(recorder)
	if !ok {
		t.Fatalf("%s pool %d built %T, which has no RecordSelected(*load.Data)", scenario, pool, e)
	}
	host := embeddedData(t)
	if n := len(host.Report().Selected); n != 0 {
		t.Fatalf("a fresh host Data holds %d entries before registration", n)
	}
	r.RecordSelected(nil)
	if n := len(host.Report().Selected); n != 0 {
		t.Fatalf("RecordSelected(nil) changed the host Data to %d entries", n)
	}
	r.RecordSelected(host)
	return host.Report().Selected, e
}

// openedWith is what the same pool puts in a Data the host passes at build time: the entries
// RecordSelected must reproduce for an engine built with a private Data.
func openedWith(t *testing.T, scenario string, pool int) []load.Selected {
	t.Helper()
	host := embeddedData(t)
	_, err := New(Input{Spec: engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, scenario), Pool: pool}, Data: host})
	if err != nil {
		t.Fatal(err)
	}
	return host.Report().Selected
}

// The engine New returns registers its pricing kernel with the host's Data on request: a
// plain engine, a data-parallel group (one shared pricer), each pool of a prefill/decode
// deployment (separate kernels), and several instances of one pool.
func TestRecordSelectedRegistersEachPricingKernel(t *testing.T) {
	Register()
	cases := []struct {
		name, scenario string
		pool           int
		group          bool
	}{
		{"plain engine", "d058-032.yaml", 0, false},
		{"dp=2 group", "d079-032.yaml", 0, true},
		{"prefill pool", "muse-h100-roce-1p1d.yaml", 0, false},
		{"decode pool", "muse-h100-roce-1p1d.yaml", 1, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, e := selectedBy(t, c.scenario, c.pool)
			if len(got) == 0 {
				t.Fatal("RecordSelected put no coefficient in the host's Data")
			}
			if _, isGroup := e.(*vllm.DPGroup); isGroup != c.group {
				t.Fatalf("built %T, want a DPGroup: %v", e, c.group)
			}
			if want := openedWith(t, c.scenario, c.pool); !reflect.DeepEqual(got, want) {
				t.Errorf("RecordSelected recorded %d entries, a build given the Data records %d", len(got), len(want))
			}
		})
	}
	// Several instances of one pool, each registered once into one host Data.
	host := embeddedData(t)
	for range 3 {
		e, err := New(Input{Spec: engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, "d058-032.yaml")}})
		if err != nil {
			t.Fatal(err)
		}
		e.(recorder).RecordSelected(host)
	}
	if got, want := host.Report().Selected, openedWith(t, "d058-032.yaml", 0); !reflect.DeepEqual(got, want) {
		t.Errorf("three instances recorded %d entries, one instance records %d", len(got), len(want))
	}
}

// narrowOverlay is a registry overlay directory that replaces one measured or calibrated
// coefficient entry by itself with a fit range of 1 to 4 tokens (same set, name, scope and method, so
// the entry is replaced in place and keeps its value), and the identity of that entry. The entry is
// the first one that pools 0 and 1 of the scenario both select, so a kernel of either pool prices a
// step against the narrow range.
func narrowOverlay(t *testing.T, scenario string) (string, load.Selected) {
	t.Helper()
	Register()
	pick := func(pool int) []load.Selected {
		host := embeddedData(t)
		if _, err := New(Input{Spec: engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, scenario), Pool: pool}, Data: host}); err != nil {
			t.Fatal(err)
		}
		return host.Report().Selected
	}
	same := func(a, b load.Selected) bool {
		return a.Set == b.Set && a.Entry == b.Entry && a.Scope.Key() == b.Scope.Key() && a.Method == b.Method
	}
	var chosen *load.Selected
	other := pick(1)
	for _, c := range pick(0) {
		if c.Kind != load.KindCoefficient || (c.Method != "measured" && c.Method != "calibrated") {
			continue
		}
		for _, o := range other {
			if same(c, o) {
				chosen = &c
				break
			}
		}
		if chosen != nil {
			break
		}
	}
	if chosen == nil {
		t.Fatalf("%s: no measured or calibrated coefficient is selected by both pools", scenario)
	}
	registry, err := substratedata.Registry()
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(registry, chosen.File))
	if err != nil {
		t.Fatal(err)
	}
	var doc yaml.Node
	if err := yaml.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	var set coefficient.Set
	if err := doc.Decode(&set); err != nil {
		t.Fatal(err)
	}
	items := mapValue(doc.Content[0], "coefficients")
	if len(items.Content) != len(set.Coefficients) {
		t.Fatalf("%s: %d nodes for %d entries", chosen.File, len(items.Content), len(set.Coefficients))
	}
	for i, e := range set.Coefficients {
		if e.Name != chosen.Entry || e.Scope.Key() != chosen.Scope.Key() || string(e.Method) != chosen.Method {
			continue
		}
		item := items.Content[i]
		fields := item.Content[1] // {name: {value, units, ...}}
		var narrow yaml.Node
		if err := yaml.Unmarshal([]byte("[{dim: tokens, low: 1, high: 4}]"), &narrow); err != nil {
			t.Fatal(err)
		}
		if old := mapValue(fields, "fit_range"); old != nil {
			*old = *narrow.Content[0]
		} else {
			fields.Content = append(fields.Content, &yaml.Node{Kind: yaml.ScalarNode, Value: "fit_range"}, narrow.Content[0])
		}
		out, err := yaml.Marshal(map[string]any{"kind": "CoefficientSet", "name": set.Name, "coefficients": []*yaml.Node{item}})
		if err != nil {
			t.Fatal(err)
		}
		dir := t.TempDir()
		if err := os.MkdirAll(filepath.Join(dir, "coefficients"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, chosen.File), out, 0o644); err != nil {
			t.Fatal(err)
		}
		return dir, *chosen
	}
	t.Fatalf("%s: the entry %s was not found in its file", chosen.File, chosen.Entry)
	return "", load.Selected{}
}

// mapValue is the value node of key in a YAML mapping, or nil.
func mapValue(m *yaml.Node, key string) *yaml.Node {
	for i := 0; i+1 < len(m.Content); i += 2 {
		if m.Content[i].Value == key {
			return m.Content[i+1]
		}
	}
	return nil
}

// buildWithOverlay builds a pool of the scenario on data that holds the narrow overlay, each engine
// on a Data of its own: the engine's open-time registration then lands in a Data nobody reports
// from, and only RecordSelected carries the kernel to the host's.
func buildWithOverlay(t *testing.T, overlay, scenario string, pool int) engine.Engine {
	t.Helper()
	data, err := kernelmodel.OpenData(load.Options{RegistryOverlays: []string{overlay}})
	if err != nil {
		t.Fatal(err)
	}
	e, err := New(Input{Spec: engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, scenario), Pool: pool}, Data: data})
	if err != nil {
		t.Fatal(err)
	}
	return e
}

// priceStep runs one request of promptLen tokens through the engine's first step, which the
// kernel prices.
func priceStep(t *testing.T, e engine.Engine, id string, promptLen int) {
	t.Helper()
	if err := e.Add(req(id, promptLen, 1), 0); err != nil {
		t.Fatal(err)
	}
	b, err := e.Schedule(0)
	if err != nil || b.Empty() {
		t.Fatalf("schedule: %v, empty %v", err, b.Empty())
	}
	e.StepTime(b)
}

// register is the host's one call per engine instance, right after open.
func register(t *testing.T, e engine.Engine, host *load.Data) {
	t.Helper()
	r, ok := e.(recorder)
	if !ok {
		t.Fatalf("%T has no RecordSelected(*load.Data)", e)
	}
	r.RecordSelected(host)
}

// extrapolated is whether the host's report marks the entry extrapolated.
func extrapolated(t *testing.T, host *load.Data, want load.Selected) bool {
	t.Helper()
	for _, s := range host.Report().Selected {
		if s.Set == want.Set && s.Entry == want.Entry && s.Scope.Key() == want.Scope.Key() && s.Method == want.Method {
			return s.Extrapolated
		}
	}
	t.Fatalf("the host's report holds no %s/%s: RecordSelected did not register the kernel", want.Set, want.Entry)
	return false
}

// Layer 10 of blis-substrate makes Kernel.RecordSelected register the kernel as an observer of
// the run's Data, so the report says at read time which entries a priced step read outside the
// fit range, and records accumulate across kernels. The host calls the engine's RecordSelected once
// per instance right after open. Here two engines (the two pools of a prefill/decode deployment,
// then three instances of one pool) price steps against an entry whose fit range is 1 to 4 tokens:
// only one kernel prices a step outside it, and the host's Result marks the entry extrapolated
// whichever order the engines are registered in. The engines that stayed inside the range do not
// clear the flag; with none registered, or with only the engines that stayed inside, it is not set.
func TestRecordSelectedFlagsExtrapolationFromOneOfTwoKernels(t *testing.T) {
	Register()
	t.Run("the two pools of a prefill and decode deployment", func(t *testing.T) {
		const scenario = "muse-h100-roce-1p1d.yaml"
		overlay, entry := narrowOverlay(t, scenario)
		prefill, decode := buildWithOverlay(t, overlay, scenario, 0), buildWithOverlay(t, overlay, scenario, 1)
		priceStep(t, prefill, "p", 64) // 64 tokens: outside 1 to 4
		priceStep(t, decode, "d", 2)   // 2 tokens: inside
		for name, order := range map[string][]engine.Engine{
			"prefill then decode": {prefill, decode},
			"decode then prefill": {decode, prefill},
		} {
			host := embeddedData(t)
			for _, e := range order {
				register(t, e, host)
			}
			if !extrapolated(t, host, entry) {
				t.Errorf("%s: the entry %s is not marked extrapolated, though the prefill kernel priced 64 tokens against 1 to 4", name, entry.Entry)
			}
		}
		inside := embeddedData(t)
		register(t, decode, inside)
		if extrapolated(t, inside, entry) {
			t.Errorf("the entry is marked extrapolated by the decode pool, which priced 2 tokens against 1 to 4")
		}
		// A later registration of the other kernel raises it, and registering the clean one again
		// clears nothing.
		register(t, prefill, inside)
		register(t, decode, inside)
		if !extrapolated(t, inside, entry) {
			t.Errorf("the flag the prefill kernel raised was lost when the decode kernel registered again")
		}
	})
	t.Run("several instances of one pool", func(t *testing.T) {
		const scenario = "muse-h100-roce-1p1d.yaml"
		overlay, entry := narrowOverlay(t, scenario)
		host := embeddedData(t)
		for i := range 3 {
			e := buildWithOverlay(t, overlay, scenario, 0)
			n := 2
			if i == 1 {
				n = 64
			}
			priceStep(t, e, "r", n)
			register(t, e, host)
		}
		if !extrapolated(t, host, entry) {
			t.Errorf("the entry is not marked extrapolated, though instance 1 priced 64 tokens against 1 to 4")
		}
	})
}

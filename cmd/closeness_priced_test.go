package cmd

import (
	"slices"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	specmodel "github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/units"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
	latencykernel "github.com/inference-sim/blis-substrate/kernel"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/blis-vllm/compat/vllmsettings"
	vllmrules "github.com/inference-sim/blis-vllm/rules"
)

// The engine reports which implementation each lowered operation runs (plan.Operation.Impl); the
// substrate's kernel turns that, with the coefficient entries it selected, into the Result's
// selected-entry report (closeness version 2). These tests drive that producer with an engine's
// plan that this repository states, a pool whose attention runs two builds, through the same
// substrate entry points a simulator uses (OpenInputs, New, RecordSelected, the data's report),
// and require what the report says about which build an entry priced. The engines at the pin do
// not report their lowered builds yet, so a double is the only way to run the counterexample.

var (
	buildA = plan.KernelImpl{Library: "flashinfer", Variant: "decode_full", CodeHash: "aaaaaaaaaaaa"}
	buildB = plan.KernelImpl{Library: "flashinfer", Variant: "decode_window", CodeHash: "bbbbbbbbbbbb"}
	laneBF = plan.KernelImpl{Library: "flashinfer", Variant: "trtllm-prefill-bf16", CodeHash: "0123456789ab"}
)

func asEntryImpl(i plan.KernelImpl) coefficient.KernelImpl {
	return coefficient.KernelImpl{Library: i.Library, Variant: i.Variant, CodeHash: i.CodeHash}
}

// poolInputs opens the pool of a committed scenario into the documents the cost model prices from.
func poolInputs(t *testing.T, scenario string) latencykernel.Inputs {
	t.Helper()
	repos := kernelmodel.Repos{Scenarios: testScenariosDir(t), Data: testData(), Rules: latencykernel.RulesFrom(vllmrules.Lookup),
		RulesVersions: vllmrules.Versions, RulesKind: "vllm", EngineSettings: vllmsettings.OfPool}
	in, err := latencykernel.OpenInputs(scenario, repos, 0)
	if err != nil {
		t.Fatalf("%s: %v", scenario, err)
	}
	return in
}

// withKeys returns in with each named measured entry (and the trtllm-gen lane's, which the registry
// transferred from an older sweep) stated as valid for impls (the registry's
// kernel_impls), and the names in drop removed from its sets.
func withKeys(in latencykernel.Inputs, keys map[string][]plan.KernelImpl, drop ...string) latencykernel.Inputs {
	var sets []*coefficient.Set
	for _, s := range in.Coefficients {
		c := *s
		c.Coefficients = nil
		for _, e := range s.Coefficients {
			if slices.Contains(drop, e.Name) {
				continue
			}
			if impls, ok := keys[e.Name]; ok && (e.Method == vocab.MethodMeasured || strings.Contains(e.Name, "trtllm_gen")) {
				e.KernelImpls = nil
				for _, i := range impls {
					e.KernelImpls = append(e.KernelImpls, asEntryImpl(i))
				}
				if e.Method != vocab.MethodMeasured {
					// A transferred or assumed entry names the build its value was measured on.
					e.MeasuredOn = []coefficient.KernelImpl{{Library: "flashinfer", Variant: "sweep-source", CodeHash: "ba9876543210"}}
				}
			}
			c.Coefficients = append(c.Coefficients, e)
		}
		sets = append(sets, &c)
	}
	in.Coefficients = sets
	return in
}

// reportOf builds the kernel of in, registers it with a fresh data as an engine does when it
// opens its pool, and returns the Result's selected-entry report as the generic document a Result
// serialises, with the coefficient entries by name.
func reportOf(t *testing.T, in latencykernel.Inputs) (map[string]any, map[string]map[string]any) {
	t.Helper()
	k, err := latencykernel.New(in)
	if err != nil {
		t.Fatalf("the pool does not build: %v", err)
	}
	d := testData()
	k.RecordSelected(d)
	se := d.Report().Result()
	raw, err := yaml.Marshal(se)
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	if err := yaml.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	byName := map[string]map[string]any{}
	coeffs, _ := doc["coefficients"].([]any)
	for _, c := range coeffs {
		e, _ := c.(map[string]any)
		byName[e["name"].(string)] = e
	}
	if p := pricedProblems(doc); len(p) != 0 {
		t.Fatalf("the report breaks closeness version 2's rules: %v", p)
	}
	return doc, byName
}

func implList(v any) []string {
	var out []string
	l, _ := v.([]any)
	for _, x := range l {
		m, _ := x.(map[string]any)
		out = append(out, implKey(m))
	}
	slices.Sort(out)
	return out
}

func keysOf(impls ...plan.KernelImpl) []string {
	var out []string
	for _, i := range impls {
		out = append(out, i.Library+"/"+i.Variant+"/"+i.CodeHash)
	}
	slices.Sort(out)
	return out
}

// gptOSSDecodeOps is gpt-oss's attention decode on two builds: the full-attention layers run A, the
// sliding-window layers run B, as an engine whose attention backend differs by layer kind lowers it.
func gptOSSDecodeOps(in latencykernel.Inputs) []plan.Operation {
	kinds := map[string]specmodel.LayerKind{}
	for _, lk := range in.Model.LayerKinds {
		kinds[lk.ID] = lk
	}
	var ops []plan.Operation
	for i, id := range in.Model.Stack.Expand() {
		impl := buildA
		for _, n := range kinds[id].Nodes {
			if n.Op == specmodel.OpAttention && n.AttentionKind == specmodel.AttentionSWA {
				impl = buildB
			}
		}
		ops = append(ops, plan.Operation{Layer: i, Primitive: plan.PrimAttentionDecode, Impl: impl})
	}
	return ops
}

// One primitive runs builds A and B in different operations: the full-attention layers on A, the
// sliding-window layers on B. An entry valid for A and B that only the sliding-window layers read
// (the sliding-window pair, which has its own) is reported with B alone as priced, though its
// kernel_impls lists both, and the part-wide pair, read by the full layers, with A alone: a sibling
// build an entry is valid for is not what it priced. An entry that lists no build is unkeyed, and
// says so; with the sliding-window pair gone, the part-wide pair is the fallback the window layers
// read, and prices both builds. Nothing here is skipped.
func TestResult_PricedIsTheBuildOfTheOperationsThatReadTheEntry(t *testing.T) {
	in := poolInputs(t, "gpt-oss-120b-h200-fp4-vllm-tp4.yaml")
	in.Operations = gptOSSDecodeOps(in)
	swa := []string{"attention_decode_rate_swa", "attention_decode_floor_swa"}
	wide := []string{"attention_decode_rate", "attention_decode_floor"}
	both := map[string][]plan.KernelImpl{}
	for _, n := range append(slices.Clone(swa), wide...) {
		both[n] = []plan.KernelImpl{buildA, buildB}
	}

	doc, got := reportOf(t, withKeys(in, both))
	if ran := implList(doc["implementations"].(map[string]any)[string(plan.PrimAttentionDecode)]); !slices.Equal(ran, keysOf(buildA, buildB)) {
		t.Fatalf("non-vacuity: the run's builds of the decode primitive are %v, want A and B", ran)
	}
	for _, n := range swa {
		e, ok := got[n]
		if !ok {
			t.Fatalf("%s was not selected", n)
		}
		if e["kernel_match"] != "exact" || !slices.Equal(implList(e["kernel_impls"]), keysOf(buildA, buildB)) || !slices.Equal(implList(e["priced"]), keysOf(buildB)) {
			t.Errorf("%s: match %v, valid for %v, priced %v; want exact, valid for A and B, priced B alone (only the window layers read it)",
				n, e["kernel_match"], implList(e["kernel_impls"]), implList(e["priced"]))
		}
	}
	for _, n := range wide {
		e := got[n]
		if !slices.Equal(implList(e["priced"]), keysOf(buildA)) {
			t.Errorf("%s: priced %v, want A alone: the window layers have a pair of their own", n, implList(e["priced"]))
		}
	}

	// An entry that lists no build is unkeyed and is not shown to match the build that ran.
	_, unkeyed := reportOf(t, withKeys(in, nil))
	if e := unkeyed["attention_decode_rate_swa"]; e["kernel_match"] != "unkeyed" || len(implList(e["kernel_impls"])) != 0 {
		t.Errorf("an entry listing no build: match %v, valid for %v; want unkeyed and none", e["kernel_match"], implList(e["kernel_impls"]))
	}

	// With no sliding-window pair the window layers fall back to the part-wide pair, which then
	// prices both builds.
	_, fallback := reportOf(t, withKeys(in, map[string][]plan.KernelImpl{"attention_decode_rate": {buildA, buildB}}, swa...))
	if e := fallback["attention_decode_rate"]; !slices.Equal(implList(e["priced"]), keysOf(buildA, buildB)) {
		t.Errorf("the part-wide rate with no window pair: priced %v, want both builds (it is the fallback the window layers read)", implList(e["priced"]))
	}
}

// The trtllm-gen prefill lane was measured on a bf16 cache. A float16 cache is 16 bits as well, but
// is not what the lane measured: it consumes the part-wide pair, and the report prices the lane for
// the bf16 cache only. The Result for an fp16 pool names the part-wide prefill entries as read, with
// no lane entry priced.
func TestResult_AnFP16CacheDoesNotPriceTheBF16Lane(t *testing.T) {
	in := poolInputs(t, "minimax-m3-b200-fp4-vllm-tp4.yaml")
	laneNames := []string{"attention_prefill_floor_trtllm_gen_kv_bf16", "attention_prefill_work_scale_trtllm_gen_kv_bf16"}
	keys := map[string][]plan.KernelImpl{}
	for _, n := range laneNames {
		keys[n] = []plan.KernelImpl{laneBF}
	}
	at := func(dtype string) map[string]map[string]any {
		keyed := withKeys(in, keys)
		kinds := map[string]specmodel.LayerKind{}
		for _, lk := range keyed.Model.LayerKinds {
			kinds[lk.ID] = lk
		}
		for i, id := range keyed.Model.Stack.Expand() {
			for _, n := range kinds[id].Nodes {
				if n.Op == specmodel.OpAttention && n.ReadsEngineKV() {
					keyed.Operations = append(keyed.Operations, plan.Operation{Layer: i, Primitive: plan.PrimAttentionPrefill,
						Format: units.QuantFormat{PayloadBits: 16}, Dtype: plan.Dtype(dtype), Impl: laneBF})
					break
				}
			}
		}
		_, got := reportOf(t, keyed)
		return got
	}
	bf := at("bf16")
	for _, n := range laneNames {
		if e, ok := bf[n]; !ok || !slices.Equal(implList(e["priced"]), keysOf(laneBF)) || e["kernel_match"] != "exact" {
			t.Errorf("a bf16 cache: %s selected %v, priced %v, match %v; want the lane priced on its kernel", n, ok, implList(e["priced"]), e["kernel_match"])
		}
	}
	fp := at("fp16")
	for _, n := range laneNames {
		if e, ok := fp[n]; ok && len(implList(e["priced"])) != 0 {
			t.Errorf("an fp16 cache: %s priced %v, but the lane was measured on a bf16 cache and is not read", n, implList(e["priced"]))
		}
	}
	var wideRead bool
	for n, e := range fp {
		if strings.HasPrefix(n, "attention_prefill_") && !slices.Contains(laneNames, n) && !strings.Contains(n, "trtllm_gen") &&
			slices.Equal(implList(e["priced"]), keysOf(laneBF)) {
			wideRead = true
		}
	}
	if !wideRead {
		t.Errorf("an fp16 cache: no part-wide prefill entry reports the kernel the prefill runs as priced; got %v", fp)
	}
}

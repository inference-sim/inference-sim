// Original to blis-sglang.

package sglangengine

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/kernel"

	"github.com/inference-sim/blis-sglang/config"
)

// fakeKernel is a cost model with fixed resident bytes and a fixed step time; it records the
// padding of the last priced batch.
type fakeKernel struct {
	kernel.Kernel
	lastPad *int
}

const gib = int64(1 << 30)

func (fakeKernel) FixedBytes() kernel.MemoryBreakdown     { return kernel.MemoryBreakdown{Weights: gib} }
func (fakeKernel) SequenceVariableBytes(tokens int) int64 { return int64(tokens) * 100 << 10 }
func (f fakeKernel) StepTime(b kernel.Batch) kernel.StepEstimate {
	*f.lastPad = b.PadTokens
	return kernel.StepEstimate{NoOverlap: 5 * time.Millisecond}
}

type fakeHost struct{ k fakeKernel }

func (h fakeHost) Open(engine.Spec) (Pool, error) {
	var doc struct {
		Cases []struct {
			Name  string            `json:"name"`
			Chip  config.Chip       `json:"chip"`
			Model config.ModelFacts `json:"model_facts"`
		} `json:"cases"`
	}
	b, err := os.ReadFile("../config/testdata/serverargs.json")
	if err != nil {
		return Pool{}, err
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		return Pool{}, err
	}
	for _, c := range doc.Cases {
		if c.Name == "qwen-h100-defaults" {
			return Pool{Kernel: h.k, Chip: c.Chip, Model: c.Model, PreModelLoadGiB: 78, ContextLen: 32768,
				EOSTokenIDs: []int64{151643, 151645}, VocabSize: 151936}, nil
		}
	}
	return Pool{}, os.ErrNotExist
}

func specOf(t *testing.T, block string) engine.Spec {
	t.Helper()
	var n yaml.Node
	if err := yaml.Unmarshal([]byte(block), &n); err != nil {
		t.Fatal(err)
	}
	return engine.Spec{Kind: Kind, Config: *n.Content[0]}
}

func newHost() fakeHost { return fakeHost{k: fakeKernel{lastPad: new(int)}} }

func TestTheFactoryBuildsAnEngineFromADocumentBlock(t *testing.T) {
	h := newHost()
	e, err := FactoryFor(h).New(specOf(t, "model_path: m\nmax_running_requests: 8\nchunked_prefill_size: 512\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(e.Fields().Fields) == 0 {
		t.Error("the engine reports no field list")
	}
	if err := e.Add(engine.Request{ID: "a", Prompt: make([]int32, 100), Output: []int32{5, 6, 7, 8}}, 0); err != nil {
		t.Fatal(err)
	}
	b, err := e.Schedule(0)
	if err != nil {
		t.Fatal(err)
	}
	if b.Empty() || b.Tokens() != 100 {
		t.Fatalf("first pass %+v, want the 100-token prompt in one chunk", b)
	}
	if d := e.StepTime(b); d != 5000 {
		t.Errorf("step time %d us, want the cost model's 5 ms", d)
	}
}

func TestTheFactoryRefusesAnotherReleaseAndWhatTheEngineDoesNotModel(t *testing.T) {
	h := newHost()
	s := specOf(t, "model_path: m\n")
	s.UpstreamVersion = "0.4.0"
	if _, err := FactoryFor(h).New(s); err == nil || !strings.Contains(err.Error(), "0.4.0") {
		t.Errorf("got %v, want a refusal naming the release", err)
	}
	for block, issue := range map[string]string{
		"model_path: m\ntp_size: 8\ndp_size: 8\nenable_dp_attention: true\n": "#70",
		"model_path: m\nenable_hierarchical_cache: true\n":                   "#37",
		"model_path: m\nstream_interval: 10\n":                               "#49",
	} {
		_, err := FactoryFor(h).New(specOf(t, block))
		if err == nil || !strings.Contains(err.Error(), issue) {
			t.Errorf("%q: got %v, want a refusal citing %s", block, err, issue)
		}
	}
}

func TestADecodePassIsPaddedToTheCaptureBucketThatHoldsIt(t *testing.T) {
	h := newHost()
	p := &pricer{k: h.k, buckets: []int{1, 2, 4, 8}}
	decode := engine.Batch{Reqs: []engine.Scheduled{{Tokens: 1}, {Tokens: 1}, {Tokens: 1}}}
	p.StepTime(decode)
	if *h.k.lastPad != 1 {
		t.Errorf("3 decodes padded by %d, want 1 (bucket 4)", *h.k.lastPad)
	}
	p.StepTime(engine.Batch{Reqs: []engine.Scheduled{{Tokens: 1}, {Tokens: 7}}})
	if *h.k.lastPad != 0 {
		t.Errorf("a pass with a prefill chunk is not a graph decode: padded by %d", *h.k.lastPad)
	}
	p.StepTime(engine.Batch{Reqs: make([]engine.Scheduled, 9)})
	if *h.k.lastPad != 0 {
		t.Errorf("a batch past the largest bucket runs eagerly")
	}
}

// Each setting the engine does not run is refused by refuseUnmodelled on its own, naming its
// issue. Through the factory the pool plan refuses attention-DP pools too, so a factory-level test
// cannot tell the refusal here from the plan's; this one calls it directly (a mutant that drops a
// case is caught by that case's row).
func TestEachUnmodelledSettingIsRefusedOnItsOwn(t *testing.T) {
	def := config.Defaults()
	if err := refuseUnmodelled(&def); err != nil {
		t.Fatalf("the defaults are refused: %v", err)
	}
	spec := "EAGLE"
	for _, tc := range []struct {
		name  string
		set   func(*config.ServerArgs)
		issue string
	}{
		{"speculative decoding", func(a *config.ServerArgs) { a.SpeculativeAlgorithm = &spec }, "#68"},
		{"DP attention", func(a *config.ServerArgs) { a.EnableDpAttention = true }, "#70"},
		{"a data-parallel size above one", func(a *config.ServerArgs) { a.DpSize = 2 }, "#70"},
		{"pipeline parallelism", func(a *config.ServerArgs) { a.PpSize = 2 }, "#72"},
		{"P/D disaggregation", func(a *config.ServerArgs) { a.DisaggregationMode = "prefill" }, "#64"},
		{"HiCache", func(a *config.ServerArgs) { a.EnableHierarchicalCache = true }, "#37"},
	} {
		a := config.Defaults()
		tc.set(&a)
		if err := refuseUnmodelled(&a); err == nil || !strings.Contains(err.Error(), tc.issue) {
			t.Errorf("%s: got %v, want a refusal citing %s", tc.name, err, tc.issue)
		}
	}
}

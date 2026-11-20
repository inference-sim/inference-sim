// Original to blis-sglang.

package sglangengine

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"

	"github.com/inference-sim/inference-sim/engines/sglang/config"
	"github.com/inference-sim/inference-sim/engines/sglang/internal/testgz"
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
	b, err := testgz.ReadFile("../config/testdata/serverargs.json")
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
		"model_path: m\nscheduler_recv_interval: 10\n":                       "#49",
	} {
		_, err := FactoryFor(h).New(specOf(t, block))
		if err == nil || !strings.Contains(err.Error(), issue) {
			t.Errorf("%q: got %v, want a refusal citing %s", block, err, issue)
		}
	}
}

// classifyingKernel prices a batch the way the substrate kernel's step model splits it: a request
// schedules a prefill chunk when it schedules more tokens than the batch's DecodeThreshold
// (kernel/step.go), and each such request costs a millisecond more than a decode pass's base 5.
type classifyingKernel struct{ fakeKernel }

func (c classifyingKernel) StepTime(b kernel.Batch) kernel.StepEstimate {
	prefills := 0
	for _, r := range b.Reqs {
		if r.Computed+r.Scheduled > 0 && r.Scheduled > b.DecodeThreshold {
			prefills++
		}
	}
	return kernel.StepEstimate{NoOverlap: time.Duration(5+prefills) * time.Millisecond}
}

// A pass whose requests each schedule one token is a decode pass: the kernel must price it through
// its decode path, not as sixteen prefill chunks (before the fix the batch carried no decode
// threshold, so the kernel read every one-token request as a prefill and priced 21 ms, not 5).
func TestADecodeOnlyPassIsPricedAsDecode(t *testing.T) {
	p := &pricer{k: classifyingKernel{newHost().k}}
	var reqs []engine.Scheduled
	for i := 0; i < 16; i++ {
		reqs = append(reqs, engine.Scheduled{Tokens: 1, Computed: 4096, PromptLen: 4096})
	}
	if got := p.StepTime(engine.Batch{Reqs: reqs}); got != 5000 {
		t.Errorf("16 one-token requests priced %d us, want 5000 (decode)", got)
	}
	chunk := append([]engine.Scheduled{{Tokens: 7, Computed: 0, PromptLen: 7}}, reqs[:3]...)
	if got := p.StepTime(engine.Batch{Reqs: chunk}); got != 6000 {
		t.Errorf("a 7-token chunk with 3 decodes priced %d us, want 6000 (one prefill)", got)
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

// The pool's env map is the process environment the scheduler's settings read, not a ServerArgs field:
// a variable the engine reads is set from it, SGLANG_ENABLE_JIT_DEEPGEMM at its default (true, the
// value the recipes export) changes nothing, and any value the engine cannot honour is refused by name.
func TestTheEngineReadsItsEnvironmentFromTheEnvMap(t *testing.T) {
	h := newHost()
	build := func(env string) error {
		_, err := FactoryFor(h).New(specOf(t, "model_path: m\nmax_running_requests: 8\nenv: {"+env+"}\n"))
		return err
	}
	for _, ok := range []string{"SGLANG_ENABLE_JIT_DEEPGEMM: '1'", "SGLANG_ENABLE_JIT_DEEPGEMM: true",
		"SGLANG_RADIX_FORCE_MISS: '0'", "SGLANG_RETRACT_DECODE_STEPS: '10', SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION: '2048'"} {
		if err := build(ok); err != nil {
			t.Errorf("env {%s}: %v", ok, err)
		}
	}
	for env, want := range map[string]string{
		"SGLANG_ENABLE_JIT_DEEPGEMM: '0'":  "SGLANG_ENABLE_JIT_DEEPGEMM",
		"SGLANG_RADIX_FORCE_MISS: '1'":     "SGLANG_RADIX_FORCE_MISS",
		"NCCL_NVLS_ENABLE: '1'":            "NCCL_NVLS_ENABLE",
		"SGLANG_RETRACT_DECODE_STEPS: 'x'": "SGLANG_RETRACT_DECODE_STEPS",
		// read by the engine's scheduler settings, whose own refusal names it
		"SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP: 'true'": "SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP",
	} {
		if err := build(env); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("env {%s}: got %v, want a refusal naming %s", env, err, want)
		}
	}
	if _, err := FactoryFor(h).New(specOf(t, "model_path: m\nenv: [SGLANG_RADIX_FORCE_MISS]\n")); err == nil || !strings.Contains(err.Error(), "map") {
		t.Errorf("an env that is not a map: got %v", err)
	}
}

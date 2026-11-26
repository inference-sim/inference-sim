// Original to blis-sglang.

package sglangengine

import (
	"encoding/json"
	"fmt"
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

// fakeHost opens the pool of the oracle case named by oracleCase, qwen-h100-defaults when it is empty.
type fakeHost struct {
	k          fakeKernel
	oracleCase string
	// arch, when set, replaces the opened model's architecture: the model a scenario names selects a
	// class by it (an architecture SGLang's config translation yields), and the allowlist reads it.
	arch string
}

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
	want := h.oracleCase
	if want == "" {
		want = "qwen-h100-defaults"
	}
	for _, c := range doc.Cases {
		if c.Name == want {
			if h.arch != "" {
				c.Model.Architecture = h.arch
			}
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

// refusingOpenHost records whether the host was asked to open a pool at all.
type refusingOpenHost struct{ opened bool }

func (h *refusingOpenHost) Open(engine.Spec) (Pool, error) {
	h.opened = true
	return Pool{}, fmt.Errorf("the host opened a pool")
}

// A release this build does not model is refused before the host opens anything: the refusal
// must not size (and record) a pool the run can never use.
func TestTheFactoryRefusesAnotherReleaseBeforeOpeningAPool(t *testing.T) {
	h := &refusingOpenHost{}
	s := specOf(t, "model_path: m\n")
	s.UpstreamVersion = "0.4.0"
	_, err := FactoryFor(h).New(s)
	if err == nil || !strings.Contains(err.Error(), "0.4.0") {
		t.Fatalf("got %v, want a refusal naming the release", err)
	}
	if h.opened {
		t.Error("the host opened a pool before the release was refused")
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
		"model_path: m\ntp_size: 8\ndp_size: 8\nenable_dp_attention: true\n": "#2356",
		"model_path: m\nenable_hierarchical_cache: true\n":                   "#2341",
		"model_path: m\nscheduler_recv_interval: 10\n":                       "#2344",
	} {
		_, err := FactoryFor(h).New(specOf(t, block))
		if err == nil || !strings.Contains(err.Error(), issue) {
			t.Errorf("%q: got %v, want a refusal citing %s", block, err, issue)
		}
	}
}

// The start-up rules of the priority settings are SGLang's, raised before the engine is built.
func TestTheFactoryRefusesWhatSGLangRefusesOfThePrioritySettings(t *testing.T) {
	h := newHost()
	for block, want := range map[string]string{
		"model_path: m\nschedule_policy: priority\n":                              "Unknown schedule_policy: policy='priority'",
		"model_path: m\nenable_priority_scheduling: true\nschedule_policy: lpm\n": "schedule_policy must be 'fcfs' or 'lof'",
		"model_path: m\nretraction_policy: priority\n":                            "--retraction-policy priority requires --enable-priority-scheduling",
	} {
		_, err := FactoryFor(h).New(specOf(t, block))
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%q: got %v, want SGLang's refusal %q", block, err, want)
		}
	}
	if _, err := FactoryFor(h).New(specOf(t, "model_path: m\nmax_running_requests: 8\nenable_priority_scheduling: true\nretraction_policy: priority\nmax_queued_requests: 4\n")); err != nil {
		t.Errorf("priority scheduling with priority retraction: %v", err)
	}
}

// The settings the plan's section 3A.4 lists for the vLLM engine that no SGLang field carries are
// refused by name, not read as another field's: they are not ServerArgs fields (fields.json holds none
// of them), and an NCCL or FlashInfer variable in the env map is one the engine does not model.
func TestASettingNoSGLangFieldCarriesIsRefusedByItsName(t *testing.T) {
	h := newHost()
	for _, name := range []string{"data_parallel_rpc_port", "moe_backend", "use_trtllm_attention", "mm_encoder_tp"} {
		_, err := FactoryFor(h).New(specOf(t, "model_path: m\n"+name+": 1\n"))
		if err == nil || !strings.Contains(err.Error(), name) {
			t.Errorf("%s: got %v, want a refusal naming it", name, err)
		}
	}
	for _, name := range []string{"NCCL_ALGO", "FLASHINFER_DISABLE_VERSION_CHECK"} {
		_, err := FactoryFor(h).New(specOf(t, "model_path: m\nenv: {"+name+": '1'}\n"))
		if err == nil || !strings.Contains(err.Error(), name) {
			t.Errorf("env %s: got %v, want a refusal naming it", name, err)
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
	spec, topk := "EAGLE", 2
	for _, tc := range []struct {
		name  string
		set   func(*config.ServerArgs)
		issue string
	}{
		{"tree drafting and verification", func(a *config.ServerArgs) { a.SpeculativeAlgorithm = &spec; a.SpeculativeEagleTopk = &topk }, "#2355"},
		{"DP attention", func(a *config.ServerArgs) { a.EnableDpAttention = true }, "#2356"},
		{"a data-parallel size above one", func(a *config.ServerArgs) { a.DpSize = 2 }, "#2356"},
		{"pipeline parallelism", func(a *config.ServerArgs) { a.PpSize = 2 }, "#2358"},
		{"P/D disaggregation", func(a *config.ServerArgs) { a.DisaggregationMode = "prefill" }, "#2351"},
		{"HiCache", func(a *config.ServerArgs) { a.EnableHierarchicalCache = true }, "#2341"},
	} {
		a := config.Defaults()
		tc.set(&a)
		if err := refuseUnmodelled(&a); err == nil || !strings.Contains(err.Error(), tc.issue) {
			t.Errorf("%s: got %v, want a refusal citing %s", tc.name, err, tc.issue)
		}
	}
	// A chain (top-k 1) of the EAGLE family is modelled (S30): the algorithm alone is not refused.
	for _, alg := range []string{"EAGLE", "EAGLE3", "NEXTN"} {
		a, one := config.Defaults(), 1
		a.SpeculativeAlgorithm, a.SpeculativeEagleTopk = &alg, &one
		if err := refuseUnmodelled(&a); err != nil {
			t.Errorf("%s at top-k 1: %v", alg, err)
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

// A verify pass carries num_draft_tokens per request (S30, inference-sim/inference-sim#2355). It is a decode: the kernel
// must price it through its decode path and pad it to a capture bucket, counted in tokens, whose
// batch size the runner captured at that width. A prefill chunk is not one, and neither is a pass whose
// request carries the width as a chunk of prompt.
func TestAVerifyPassIsPricedAsDecodeAtTheDraftWidth(t *testing.T) {
	const width = 4
	h := newHost()
	p := &pricer{k: h.k, buckets: []int{1, 2, 4, 8}, width: width}
	verify := func(n int) engine.Batch {
		b := engine.Batch{}
		for i := 0; i < n; i++ {
			b.Reqs = append(b.Reqs, engine.Scheduled{Tokens: width, Computed: 100, Drafts: width - 1})
		}
		return b
	}
	for _, tc := range []struct{ reqs, wantPad int }{{1, 0}, {3, 1 * width}, {5, 3 * width}, {8, 0}} {
		kb := p.shape(verify(tc.reqs))
		if kb.DecodeThreshold != width {
			t.Fatalf("decode threshold %d, want the draft width %d", kb.DecodeThreshold, width)
		}
		if kb.PadTokens != tc.wantPad {
			t.Errorf("%d verify requests padded by %d tokens, want %d (bucket rows x width)", tc.reqs, kb.PadTokens, tc.wantPad)
		}
	}
	if kb := p.shape(verify(9)); kb.PadTokens != 0 {
		t.Errorf("a batch past the largest bucket runs eagerly: padded by %d", kb.PadTokens)
	}
	// A pass that mixes in a prompt chunk is not a graph decode.
	mixed := verify(3)
	mixed.Reqs = append(mixed.Reqs, engine.Scheduled{Tokens: 37, PromptLen: 37})
	if kb := p.shape(mixed); kb.PadTokens != 0 {
		t.Errorf("a pass with a prefill chunk was padded by %d tokens", kb.PadTokens)
	}
	// A request that schedules the width without drafts is a prompt chunk, not a verify.
	chunk := engine.Batch{Reqs: []engine.Scheduled{{Tokens: width, PromptLen: width}, {Tokens: width, PromptLen: width}, {Tokens: width, PromptLen: width}}}
	if kb := p.shape(chunk); kb.PadTokens != 0 {
		t.Errorf("a prompt chunk of the draft width was padded as a graph decode: %d", kb.PadTokens)
	}
}

// The factory builds a speculating engine for each algorithm of the EAGLE family at top-k 1, on a model
// whose draft layers ship in its checkpoint, and the engine drives the draft/verify contract: a decode
// pass schedules num_draft_tokens tokens of which num_draft_tokens-1 are drafts, the outcome's accepted
// drafts plus the one the target samples are emitted, and NEXTN is EAGLE.
func TestTheFactoryBuildsAnEagleChain(t *testing.T) {
	const steps = 3
	for _, alg := range []string{"EAGLE", "EAGLE3", "NEXTN"} {
		for accepted := 0; accepted <= steps; accepted++ {
			t.Run(fmt.Sprintf("%s/accept%d", alg, accepted), func(t *testing.T) {
				h := newHost()
				h.oracleCase = "glm53-h100-tp8-mtp"
				e, err := FactoryFor(h).New(specOf(t, fmt.Sprintf("model_path: m\nspeculative_algorithm: %s\nspeculative_num_steps: %d\n"+
					"speculative_eagle_topk: 1\nspeculative_num_draft_tokens: %d\n", alg, steps, steps+1)))
				if err != nil {
					t.Fatal(err)
				}
				out := make([]int32, 40)
				for i := range out {
					out[i] = int32(500 + i)
				}
				if err := e.Add(engine.Request{ID: "a", Prompt: make([]int32, 64), Output: out, Tags: map[string]string{"ignore_eos": "true"}}, 0); err != nil {
					t.Fatal(err)
				}
				b, err := e.Schedule(0)
				if err != nil || b.Empty() {
					t.Fatalf("prefill: %+v %v", b, err)
				}
				if _, err := e.Complete(b.ID, engine.Outcome{}, 1); err != nil {
					t.Fatal(err)
				}
				b, err = e.Schedule(2)
				if err != nil || b.Empty() {
					t.Fatalf("verify pass: %+v %v", b, err)
				}
				if r := b.Reqs[0]; r.Tokens != steps+1 || r.Drafts != steps {
					t.Fatalf("verify pass schedules %d tokens with %d drafts, want %d and %d", r.Tokens, r.Drafts, steps+1, steps)
				}
				res, err := e.Complete(b.ID, engine.Outcome{Accepted: map[string]int{"a": accepted}}, 3)
				if err != nil {
					t.Fatal(err)
				}
				if got := res.Emitted["a"]; got != accepted+1 {
					t.Fatalf("emitted %d tokens, want the %d accepted drafts and the target's own", got, accepted)
				}
			})
		}
	}
}

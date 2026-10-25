package vllm

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"
)

// base is the engine block every test starts from.
const base = "block_size: 16\nmax_num_seqs: 4\nmax_num_batched_tokens: 64\nmax_model_len: 512\n" +
	"num_gpu_blocks: 64\nuse_v2_model_runner: false\nasync_scheduling: false\n"

// block is the base engine block with extra's top-level keys added or replacing the base's.
func block(t testing.TB, extra string) yaml.Node {
	t.Helper()
	keyOf := func(l string) string { k, _, _ := strings.Cut(l, ":"); return k }
	over := map[string]bool{}
	for _, l := range strings.Split(strings.TrimSpace(extra), "\n") {
		if l != "" {
			over[keyOf(l)] = true
		}
	}
	var lines []string
	for _, l := range strings.Split(strings.TrimSpace(base), "\n") {
		if !over[keyOf(l)] {
			lines = append(lines, l)
		}
	}
	src := strings.Join(lines, "\n") + "\n" + extra
	var n yaml.Node
	if err := yaml.Unmarshal([]byte(src), &n); err != nil {
		t.Fatal(err)
	}
	return *n.Content[0]
}

func spec(t testing.TB, extra string) engine.Spec {
	return engine.Spec{Kind: Kind, UpstreamVersion: Version, Config: block(t, extra)}
}

func fixed(engine.Spec) (Pricer, error) { return FixedPricer(100), nil }

func build(t testing.TB, extra string) func() (engine.Engine, error) {
	t.Helper()
	s := spec(t, extra)
	return func() (engine.Engine, error) { return Factory{Pricer: fixed}.New(s) }
}

func req(id string, prompt, output int) engine.Request {
	r := engine.Request{ID: id, Prompt: make([]int32, prompt), Output: make([]int32, output)}
	for i := range r.Prompt {
		r.Prompt[i] = int32(len(id)*1000 + i + 1)
	}
	for i := range r.Output {
		r.Output[i] = int32(9000 + i)
	}
	return r
}

func arrivals(rs ...engine.Request) []enginetest.Arrival {
	out := make([]enginetest.Arrival, len(rs))
	for i, r := range rs {
		out[i] = enginetest.Arrival{At: engine.Time(i * 10), Req: r}
	}
	return out
}

// The v0.32.0 engine keeps every law of the engine contract (blis-schemas enginetest): no
// request is lost or finished twice, batch IDs increase, batches complete in order, no more
// are in flight than max_concurrent_batches, a step always takes time, a batch emits only for
// its own requests and at most one token plus its accepted drafts each, and the same inputs
// give the same run. Every assigned output is emitted in full and stops the request.
func TestConformance(t *testing.T) {
	for _, c := range []struct {
		name     string
		extra    string
		opt      enginetest.Options
		preempts bool
	}{
		{"sync", "", enginetest.Options{}, false},
		// Nine blocks hold one 128-token sequence beside the null block, the least vLLM starts with.
		{"preemption under tight KV", "num_gpu_blocks: 9\nmax_model_len: 128\n", enginetest.Options{}, true},
		{"priority preemption", "num_gpu_blocks: 9\nmax_model_len: 128\nscheduling_policy: priority\n", enginetest.Options{}, true},
		{"eagle3 drafts accepted", "speculative: {method: eagle3, num_spec_tokens: 3}\n",
			enginetest.Options{Accepted: func(engine.Batch, engine.Scheduled) int { return 2 }}, false},
		{"long prefill cap", "long_prefill_token_threshold: 8\n", enginetest.Options{}, false},
		{"no prefix caching", "enable_prefix_caching: false\n", enginetest.Options{}, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			reqs := []engine.Request{req("a", 40, 6), req("b", 70, 3), req("c", 20, 9), req("d", 90, 1), req("e", 33, 5)}
			if c.preempts {
				// Four requests whose decodes outgrow nine blocks, so the scheduler preempts.
				reqs = []engine.Request{req("a", 40, 30), req("b", 40, 30), req("c", 40, 30), req("d", 40, 30)}
			}
			tr := enginetest.TestEngine(t, build(t, c.extra), arrivals(reqs...), c.opt)
			for id, f := range tr.Finished {
				if f.Reason != engine.FinishStop {
					t.Errorf("%s finished %+v, want stop", id, f)
				}
			}
			preempted, drafts, overlapped := 0, 0, false
			for i, s := range tr.Steps {
				preempted += len(s.Batch.Preempted)
				for _, r := range s.Batch.Reqs {
					drafts += r.Drafts
				}
				overlapped = overlapped || i > 0 && s.Batch.ID != tr.Steps[i-1].Batch.ID+1
			}
			// Each case reaches the behaviour it names, so passing is not vacuous.
			if c.preempts && preempted == 0 {
				t.Error("the tight-KV case never preempted")
			}
			if c.opt.Accepted != nil && drafts == 0 {
				t.Error("the drafting case verified no drafts")
			}
		})
	}
}

// The protocol errors a simulator relies on: completing an unknown batch, finishing an unknown
// request (a no-op) and an unknown transfer. The factory builds one batch in flight (see
// TestFactoryRefuses), so completing out of order is checked on an engine built with two slots
// directly, the queue the batch-queue work will open (https://github.com/inference-sim/blis-vllm/issues/21).
func TestProtocolErrors(t *testing.T) {
	if err := enginetest.CheckProtocolErrors(build(t, "max_num_batched_tokens: 32\n"),
		req("a", 40, 3), req("b", 40, 3)); err != nil {
		t.Fatal(err)
	}
	twoSlots := func() (engine.Engine, error) {
		cfg, err := decodeConfig(block(t, "max_num_batched_tokens: 32\n"))
		if err != nil {
			return nil, err
		}
		st, err := resolve(cfg)
		if err != nil {
			return nil, err
		}
		st.queueSize = 2
		return newEngine(st, FixedPricer(100))
	}
	if e, err := twoSlots(); err != nil || e.InFlightLimit() != 2 {
		t.Fatalf("two-slot engine: %v, %v", e, err)
	}
	if err := enginetest.CheckProtocolErrors(twoSlots, req("a", 40, 3), req("b", 40, 3)); err != nil {
		t.Fatal(err)
	}
}

// The engine is built through blis-schemas' registry from an opaque spec, as a simulator
// builds any engine; Register is idempotent.
func TestBuildsThroughTheRegistry(t *testing.T) {
	Register(Factory{Pricer: fixed})
	Register(Factory{Pricer: fixed})
	s := spec(t, "")
	enginetest.TestEngine(t, func() (engine.Engine, error) { return engine.New(s) },
		arrivals(req("a", 40, 6), req("b", 70, 3)), enginetest.Options{})
}

// InFlightLimit is vLLM's max_concurrent_batches (vllm/config/vllm.py:577-588@5df63a72), which
// is 1 for every engine the factory builds: a synchronous one on one pipeline stage. Larger
// queues are refused (TestFactoryRefuses) until the adapter reproduces vLLM's batch queue
// (https://github.com/inference-sim/blis-vllm/issues/21).
func TestInFlightLimitIsMaxConcurrentBatches(t *testing.T) {
	e, err := build(t, "")()
	if err != nil {
		t.Fatal(err)
	}
	if got := e.InFlightLimit(); got != 1 {
		t.Errorf("InFlightLimit %d, want 1", got)
	}
}

// The factory refuses what it does not model or what vLLM refuses, each with its reason, and
// never builds a v0.32.0 engine for another release (#2021 B6).
func TestFactoryRefuses(t *testing.T) {
	f := Factory{Pricer: fixed}
	for _, c := range []struct {
		name, version, extra, want string
	}{
		{"another release", "0.33.0", "", "this build models vLLM 0.32.0"},
		{"a compatibility release without its engine", "0.31.0", "", "compatibility engine"},
		{"an unknown key", Version, "max_num_seqz: 3\n", "max_num_seqz"},
		{"no KV blocks", Version, "num_gpu_blocks: 1\n", "num_gpu_blocks"},
		{"an unknown policy", Version, "scheduling_policy: sjf\n", "fcfs and priority"},
		{"an unmodelled draft method", Version, "speculative: {method: ngram, num_spec_tokens: 3}\n", "not built"},
		{"a drafter whose slots the deployment cannot state", Version, "speculative: {method: draft_model, num_spec_tokens: 3}\n", "not built"},
		{"dspark on the V1 runner", Version, "speculative: {method: dspark, num_spec_tokens: 3}\n", "V2 model runner"},
		{"async with drafts", Version, "async_scheduling: true\nspeculative: {method: eagle3, num_spec_tokens: 3}\n", "async scheduling with speculative"},
		{"V1 runner with PP and async", Version, "async_scheduling: true\npipeline_parallel_size: 2\n", "V1 model runner"},
		{"prefix_match_unit", Version, "prefix_match_unit: 8\n", "prefix_match_unit"},
		{"max_cpu_loras", Version, "max_cpu_loras: 4\n", "max_cpu_loras"},
		{"DP prefill throttling", Version, "prefill_schedule_interval: 4\n", "data-parallel"},
		{"active seqs above max_num_seqs", Version, "max_num_active_seqs: 9\n", "max_num_active_seqs"},
		// What vLLM v0.32.0 refuses at startup (review I1, S10, S11).
		{"a KV pool below one max_model_len sequence", Version, "num_gpu_blocks: 32\n", "cannot hold one max_model_len"},
		{"a budget below max_num_seqs", Version, "max_num_seqs: 8\nmax_num_batched_tokens: 7\n", "below max_num_seqs"},
		{"a negative max_num_scheduled_tokens", Version, "max_num_scheduled_tokens: -1\n", "max_num_scheduled_tokens -1 is negative"},
		{"a negative long-prefill threshold", Version, "long_prefill_token_threshold: -1\n", "long_prefill_token_threshold -1 is negative"},
		{"a negative max_num_active_seqs", Version, "max_num_active_seqs: -1\n", "max_num_active_seqs -1 is negative"},
		{"a negative max_loras", Version, "max_loras: -1\n", "max_loras -1 is negative"},
		{"a NaN watermark", Version, "watermark: .nan\n", "watermark NaN is outside"},
		// The batch queue, until the adapter reproduces it (review C1, issue 21).
		{"async scheduling", Version, "async_scheduling: true\n", "max_concurrent_batches 2"},
		{"PP on the V1 runner", Version, "pipeline_parallel_size: 2\n", "max_concurrent_batches 2"},
		{"PP2 async on the V2 runner", Version, "async_scheduling: true\nuse_v2_model_runner: true\npipeline_parallel_size: 2\n", "max_concurrent_batches 3"},
		{"mamba_cache_mode", Version, "mamba_cache_mode: align\n", "mamba_cache_mode is not modelled"},
		{"swa_bounded_replay", Version, "swa_bounded_replay: false\n", "swa_bounded_replay is not modelled"},
	} {
		s := spec(t, c.extra)
		s.UpstreamVersion = c.version
		_, err := f.New(s)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err %v, want one naming %q", c.name, err, c.want)
		}
	}
	// Settings vLLM resolves from facts the deployment cannot state must be stated.
	// max_num_seqs and max_num_batched_tokens: vllm serve sizes them from the GPU, and
	// SchedulerConfig's own values are test defaults (review I7). An unstated async_scheduling
	// resolves on, which opens the batch queue the adapter refuses.
	for _, missing := range []string{"block_size", "max_model_len", "use_v2_model_runner", "max_num_seqs",
		"max_num_batched_tokens", "async_scheduling"} {
		var kept []string
		for _, l := range strings.Split(strings.TrimSpace(base), "\n") {
			if !strings.HasPrefix(l, missing+":") {
				kept = append(kept, l)
			}
		}
		var n yaml.Node
		if err := yaml.Unmarshal([]byte(strings.Join(kept, "\n")), &n); err != nil {
			t.Fatal(err)
		}
		_, err := f.New(engine.Spec{Kind: Kind, UpstreamVersion: Version, Config: *n.Content[0]})
		if err == nil || !strings.Contains(err.Error(), missing) {
			t.Errorf("unstated %s: err %v, want one naming it", missing, err)
		}
	}
	if _, err := (Factory{}).New(spec(t, "")); err == nil || !strings.Contains(err.Error(), "Pricer") {
		t.Errorf("no pricer: err %v", err)
	}
}

// The smallest KV pool vLLM starts with, one max_model_len sequence beside the null block, is
// built, and a request that fills the window on it is admitted and finishes: the startup check
// refuses exactly the pools vLLM refuses, so no built engine leaves a request waiting for ever
// behind the idle poll (review I1, S12).
func TestTheSmallestPoolVLLMStartsServesAFullWindow(t *testing.T) {
	// max_model_len 128 in 16-token blocks needs 8 blocks, plus the null block.
	tr := enginetest.TestEngine(t, build(t, "num_gpu_blocks: 9\nmax_model_len: 128\n"),
		arrivals(req("full", 100, 28)), enginetest.Options{})
	if f := tr.Finished["full"]; f.Reason != engine.FinishStop || f.OutputTokens != 28 {
		t.Errorf("full-window request finished %+v, want stop after 28 tokens", f)
	}
	if _, err := build(t, "num_gpu_blocks: 8\nmax_model_len: 128\n")(); err == nil ||
		!strings.Contains(err.Error(), "cannot hold one max_model_len") {
		t.Errorf("one block fewer: err %v, want the startup refusal", err)
	}
}

// A compatibility release is built by the engine the caller wires in, and only for those
// releases.
func TestCompatSeam(t *testing.T) {
	var got string
	f := Factory{Pricer: fixed, Compat: func(s engine.Spec) (engine.Engine, error) {
		got = s.UpstreamVersion
		return nil, errors.New("compat built")
	}}
	for _, v := range CompatVersions {
		s := spec(t, "")
		s.UpstreamVersion = v
		if _, err := f.New(s); err == nil || err.Error() != "compat built" || got != v {
			t.Errorf("%s: err %v, compat saw %q", v, err, got)
		}
	}
}

// Input checks are the input processor's (vllm/v1/engine/input_processor.py:565-602@5df63a72):
// an empty prompt, a prompt that fills the window, an adapter while LoRA is off, and a
// duplicate ID are refused.
func TestAddRefuses(t *testing.T) {
	e, err := build(t, "")()
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Add(req("ok", 10, 2), 0); err != nil {
		t.Fatal(err)
	}
	lora := req("lora", 10, 2)
	lora.Adapter = "a1"
	for name, r := range map[string]engine.Request{
		"empty prompt": req("empty", 0, 2), "prompt fills the window": req("long", 512, 2),
		"no output": req("none", 10, 0), "duplicate": req("ok", 10, 2), "adapter with LoRA off": lora,
	} {
		if err := e.Add(r, 0); err == nil {
			t.Errorf("%s was accepted", name)
		}
	}
}

// A client abort finishes a request through Finish, frees its blocks, and its batch's later
// output is not emitted; finishing it again, or an unknown request, is a no-op.
func TestFinishAbortsAndFrees(t *testing.T) {
	e, err := build(t, "")()
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Add(req("a", 40, 20), 0); err != nil {
		t.Fatal(err)
	}
	b, err := e.Schedule(0)
	if err != nil || b.Empty() {
		t.Fatalf("schedule: %v %v", b, err)
	}
	if err := e.Finish("a", engine.FinishAbort, 5); err != nil {
		t.Fatal(err)
	}
	if err := e.Finish("a", engine.FinishAbort, 6); err != nil {
		t.Fatal(err)
	}
	res, err := e.Complete(b.ID, engine.Outcome{}, 100)
	if err != nil {
		t.Fatal(err)
	}
	if res.Emitted["a"] != 0 || len(res.Finished) != 0 {
		t.Errorf("an aborted request emitted %v, finished %v", res.Emitted, res.Finished)
	}
	if s := e.Snapshot(); s.KVBlocksUsed != 0 || s.Running != 0 || s.Waiting != 0 {
		t.Errorf("after the abort: %+v, want nothing held", s)
	}
}

// A request's prefix shared with an earlier one is a prefix-cache hit at admission, reported
// on the batch as CachedTokens and in the prefix-cache counters.
func TestPrefixHitIsReported(t *testing.T) {
	e, err := build(t, "")()
	if err != nil {
		t.Fatal(err)
	}
	a, b := req("x", 48, 1), req("x", 48, 1)
	a.ID, b.ID = "first", "second"
	run := func(r engine.Request) engine.Batch {
		if err := e.Add(r, 0); err != nil {
			t.Fatal(err)
		}
		bt, err := e.Schedule(0)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := e.Complete(bt.ID, engine.Outcome{}, 1); err != nil {
			t.Fatal(err)
		}
		return bt
	}
	run(a)
	second := run(b)
	if second.Reqs[0].CachedTokens != 32 {
		t.Errorf("the second request's cached tokens: %d, want 32 (two full blocks; the last is recomputed)", second.Reqs[0].CachedTokens)
	}
	m := e.Metrics(2)
	hits, err := m.Value("vllm:prefix_cache_hits")
	if err != nil || hits != 32 {
		t.Errorf("vllm:prefix_cache_hits = %v (%v), want 32", hits, err)
	}
}

// Metrics exposes the exact families as Prometheus names them: counters without _total in the
// family name, histograms with their buckets, and the scheduler gauges after the last step.
func TestMetricsFamilies(t *testing.T) {
	var last engine.Engine
	newE := func() (engine.Engine, error) {
		e, err := build(t, "")()
		last = e
		return e, err
	}
	tr := enginetest.TestEngine(t, newE, arrivals(req("a", 40, 6), req("b", 70, 3)), enginetest.Options{})
	m := last.Metrics(tr.End)
	for name, want := range map[string]float64{
		"vllm:num_requests_running": 0, "vllm:num_requests_waiting": 0, "vllm:generation_tokens": 9,
		"vllm:prompt_tokens": 110,
	} {
		got, err := m.Value(name)
		if err != nil || got != want {
			t.Errorf("%s = %v (%v), want %v", name, got, err, want)
		}
	}
	h, ok := m.Family("vllm:request_generation_tokens")
	if !ok || h.Kind != engine.Histogram {
		t.Fatalf("vllm:request_generation_tokens: %+v", h)
	}
	var count float64
	for _, s := range h.Samples {
		if s.Name == "vllm:request_generation_tokens_count" {
			count = s.Value
		}
	}
	if count != 2 {
		t.Errorf("request_generation_tokens_count %v, want 2", count)
	}
	succ, ok := m.Family("vllm:request_success")
	if !ok || succ.Kind != engine.Counter {
		t.Fatalf("vllm:request_success: %+v", succ)
	}
	for _, s := range succ.Samples {
		if s.Labels["finished_reason"] == "stop" && s.Value != 2 {
			t.Errorf("request_success{stop} %v, want 2", s.Value)
		}
	}
}

func ExampleFactory() {
	var n yaml.Node
	_ = yaml.Unmarshal([]byte("block_size: 16\nmax_num_seqs: 8\nmax_num_batched_tokens: 256\nmax_model_len: 4096\n"+
		"num_gpu_blocks: 1024\nuse_v2_model_runner: true\nasync_scheduling: false\n"), &n)
	e, err := Factory{Pricer: func(engine.Spec) (Pricer, error) { return FixedPricer(1000), nil }}.New(
		engine.Spec{Kind: Kind, UpstreamVersion: Version, Config: *n.Content[0]})
	if err != nil {
		panic(err)
	}
	fmt.Println(e.InFlightLimit())
	// Output: 1
}

// Acceptance is the outcome's (#2021 B3): a decode that verifies K drafts emits one sampled
// token plus the drafts the outcome accepted, never more than it verified, so with EAGLE3 K=3
// and two accepted each verify step, an output of 9 tokens takes the prefill's first token
// and then three per decode.
func TestAcceptanceIsTheOutcomes(t *testing.T) {
	for _, c := range []struct {
		accepted, wantMax int
	}{{0, 1}, {2, 3}, {7, 4}} {
		tr := enginetest.TestEngine(t, build(t, "speculative: {method: eagle3, num_spec_tokens: 3}\n"),
			arrivals(req("a", 20, 13)), enginetest.Options{
				Accepted: func(engine.Batch, engine.Scheduled) int { return c.accepted }})
		most := 0
		for _, s := range tr.Steps {
			most = max(most, s.Result.Emitted["a"])
		}
		if most != c.wantMax {
			t.Errorf("accepted %d: at most %d tokens a step, want %d", c.accepted, most, c.wantMax)
		}
		if f := tr.Finished["a"]; f.OutputTokens != 13 {
			t.Errorf("accepted %d: finished %+v, want all 13 tokens", c.accepted, f)
		}
	}
}

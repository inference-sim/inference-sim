package latencykernel

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/units"
	"github.com/inference-sim/blis-substrate/load"
)

// SUB-6, SUB-11: the kernel prices the engine settings the plan states, and nothing it would
// resolve itself. Each case below builds one catalog deployment from a test plan of plain values
// (fixtureLowered, which no engine resolves), changes one resolved setting of the plan, and
// checks that a price the kernel returns moves: a step time, a memory figure or a host cost.

// fixtureLowered is a lowered plan of plain test values: the settings a deployment would state
// to price a step, chosen by this test and resolved by no engine.
func fixtureLowered() plan.Lowered {
	return plan.Lowered{Plan: plan.PoolPlan{
		Batch:      plan.BatchLimits{Requests: 64, Tokens: 8192},
		MaxContext: 8192,
		PageTokens: 16,
		Cache:      plan.CacheFormat{ElementBytes: 2, StateElementBytes: 4},
		Graphs: plan.GraphPolicy{Decode: plan.LaunchGraph, Other: plan.LaunchPiecewise, Ceiling: 128,
			Buckets: []int{1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128}},
		InFlight:      2,
		DecodeContext: plan.DecodeContextPlan{Interleave: 1},
		Engram:        plan.EngramPlacement{Host: true, Shards: 1},
	}}
}

// wideCapture states a capture ladder up to 512 tokens.
func wideCapture(l *plan.Lowered) {
	l.Plan.Graphs.Ceiling = 512
	l.Plan.Graphs.Buckets = []int{1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 128, 256, 512}
}

// noDrafter is a Repos.DraftModel for a pool that names no drafter.
func noDrafter(*deployment.Deployment, int) (string, error) { return "", nil }

// lowerTo is a Repos.Lower that hands every pool fixtureLowered with the context bound, the page
// size and the memory share a test states, edited further by f.
func lowerTo(maxContext, page int, util float64, f ...func(*plan.Lowered)) func(Inputs) (*plan.Lowered, error) {
	return func(Inputs) (*plan.Lowered, error) {
		l := fixtureLowered()
		l.Plan.MaxContext, l.Plan.PageTokens, l.Plan.MemoryFraction = units.Tokens(maxContext), units.Tokens(page), util
		for _, e := range f {
			e(&l)
		}
		return &l, nil
	}
}

// planRules are the rules the cases price under: the windowed rules (a window cache holds its
// window's pages), with each stated cache paged on its own rather than in a shared pool, and
// nothing else of any engine.
type planRules struct{ windowedRules }

func (planRules) KVCacheGroups(rules.KVCacheGroupsInput) (rules.KVCacheGrouping, error) {
	return rules.KVCacheGrouping{}, errors.New("these test rules group no caches")
}

// planCase opens one catalog deployment and hands the kernel the lowered plan f edits from
// fixtureLowered.
func planCase(t *testing.T, d *load.Data, modelName, chip string, nodes int, parallel string,
	f func(*plan.Lowered)) Inputs {
	t.Helper()
	sets := "[cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory, cost-model-fabric, cost-model-offload-tiers]"
	packaging := ""
	if ch, err := d.Chip(chip); err == nil && ch.GPUsPerNode == 0 {
		packaging = "  gpus_per_node: 8\n"
	}
	body := fmt.Sprintf("kind: Scenario\nname: p\nmodel: %s\ncoefficients: %s\nengine_version: \"0.32.0\"\n"+
		"cluster:\n  hardware: %s\n  fabric: ib-400g\n  nodes: %d\n%s---\nkind: Deployment\nname: d\n"+
		"pools:\n  - role: colocated\n    nodes: %d\n    parallel: {%s}\n", modelName, sets, chip, nodes,
		packaging, nodes, parallel)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "s.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	low := fixtureLowered()
	if f != nil {
		f(&low)
	}
	r := Repos{Scenarios: dir, Data: d, Rules: func(string) rules.EngineRules { return planRules{} },
		DraftModel: noDrafter,
		Lower:      func(Inputs) (*plan.Lowered, error) { return &low, nil }}
	in, err := OpenInputs("s.yaml", r, 0)
	if err != nil {
		t.Fatalf("%s on %s (%s): %v", modelName, chip, parallel, err)
	}
	in.DraftFirstPassRows, in.DraftPassRequests = plan.FirstPassScheduled, plan.PassRequestsAll
	return in
}

// prices is every price a kernel returns for a fixed set of questions, as one comparable string:
// step times for decode, verify, prefill, chunked and mixed batches, the per-sequence and fixed
// memory, each pipeline stage's memory, the host costs and the drafts an adaptive verifier runs.
func prices(t *testing.T, in Inputs) string {
	t.Helper()
	k, err := New(in)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	var b strings.Builder
	step := func(width int, reqs ...kernel.ReqShape) {
		bt := kernel.Batch{Reqs: reqs, DecodeThreshold: width}
		e := k.StepTime(bt)
		fmt.Fprintf(&b, "%d/%d %v;", e.Overlap, e.NoOverlap, k.VerifiedDrafts(bt))
	}
	decodes := func(n, ctx, sched int) []kernel.ReqShape {
		out := make([]kernel.ReqShape, n)
		for i := range out {
			out[i] = kernel.ReqShape{Scheduled: sched, Computed: ctx, PromptLen: min(ctx, 1024)}
		}
		return out
	}
	for _, n := range []int{1, 32, 256, 1024} {
		for _, ctx := range []int{512, 4096, 32768} {
			step(1, decodes(n, ctx, 1)...)
			step(8, decodes(n, ctx, 1)...)
			step(8, decodes(n, ctx, 4)...)
		}
	}
	for _, s := range []int{1, 17, 512, 2048, 8192, 16384} {
		step(1, kernel.ReqShape{Scheduled: s, PromptLen: s})
		step(1, kernel.ReqShape{Scheduled: s, Computed: 4096, PromptLen: 4096 + s})
		step(1, kernel.ReqShape{Scheduled: s, PromptLen: s + 1024, CachedTokens: 1024})
		step(1, kernel.ReqShape{Scheduled: s, Computed: 32768, PromptLen: 32768 + s})
	}
	step(1, append([]kernel.ReqShape{{Scheduled: 1024, PromptLen: 2048}}, decodes(31, 3000, 1)...)...)
	for _, n := range []int{1, 1024, 8192, 65536} {
		fmt.Fprintf(&b, "%d %d %d;", k.SequenceVariableBytes(n), k.SequenceBoundedBytes(n),
			k.SequenceBoundedBytesAt(n, 64))
	}
	fmt.Fprintf(&b, "%+v %d %d %+v;", k.FixedBytes(), k.SequenceFixedBytes(), k.SequenceAdmissionBytes(),
		k.PipelineStageMemory())
	fmt.Fprintf(&b, "%d %+v %d %d %d;", k.AdmissionOverhead(1024), k.AdmissionDemand(1024),
		k.OutputTokenOverhead(), k.CompletionOverhead(), k.KVLayerPageBytes(16))
	return b.String()
}

// TestEachResolvedPlanSettingMovesAPrice: every resolved setting the plan states moves a price
// the kernel returns, so the kernel prices the setting as the engine resolved it. A kernel that
// resolved the setting itself, from a default or from the engine's own setting, would price both
// plans alike.
func TestEachResolvedPlanSettingMovesAPrice(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	type where struct {
		model, chip string
		nodes       int
		parallel    string
	}
	var (
		dense     = where{"llama-3.1-70b-instruct", "h100", 1, "tp: 4, pp: 1, dp: 1"}
		routed    = where{"qwen3-30b-a3b", "h200", 1, "tp: 1, pp: 1, dp: 8, enable_expert_parallel: true"}
		spMoE     = where{"qwen3-30b-a3b", "h200", 1, "tp: 4, pp: 1, dp: 2, enable_expert_parallel: true"}
		latent    = where{"deepseek-v3", "h200", 1, "tp: 8, pp: 1, dp: 1"}
		dcp       = where{"deepseek-v3", "h200", 1, "tp: 8, pp: 1, dp: 1, dcp: 2"}
		hybrid    = where{"nemotron-3.5-lightning-30b-a3b-bf16", "h100", 1, "tp: 2, pp: 1, dp: 1"}
		v41       = where{"deepseek-v4.1-flash", "b200", 1, "tp: 4, pp: 1, dp: 1"}
		v41ep     = where{"deepseek-v4.1-flash", "b200", 1, "tp: 1, pp: 1, dp: 4, enable_expert_parallel: true"}
		muse      = where{"muse-spark-open-816b-a42b", "gb200-nvl72", 1, "tp: 4, pp: 1, dp: 1"}
		ringed    = where{"gpt-oss-120b", "h200", 1, "tp: 4, pp: 1, dp: 1"}
		hopperFP8 = where{"minimax-m2.5", "h200", 1, "tp: 4, pp: 1, dp: 1"}
		v4pro     = where{"deepseek-v4-pro", "b200", 1, "tp: 8, pp: 1, dp: 1"}
		piped     = where{"qwen3.5-397b-a17b", "b200", 1, "tp: 4, pp: 2, dp: 1"}
	)
	for _, c := range []struct {
		field string
		at    where
		base  func(*plan.Lowered)
		edit  func(*plan.Lowered)
	}{
		{"BatchLimits.Requests", v41, func(l *plan.Lowered) { l.Plan.Drafts.Chain = 5; wideCapture(l) },
			func(l *plan.Lowered) { l.Plan.Batch.Requests = 2 }},
		{"BatchLimits.Tokens", dense, nil, func(l *plan.Lowered) { l.Plan.Batch.Tokens = 2048 }},
		{"PoolPlan.MaxContext", ringed, nil, func(l *plan.Lowered) { l.Plan.MaxContext = 2048 }},
		{"PoolPlan.PageTokens", dense, nil, func(l *plan.Lowered) { l.Plan.PageTokens = 256 }},
		{"PoolPlan.MemoryFraction", piped, func(l *plan.Lowered) { l.Plan.MemoryFraction = 0.95 },
			func(l *plan.Lowered) { l.Plan.MemoryFraction = 0.3 }},
		{"PoolPlan.WeightFormat", dense, nil, func(l *plan.Lowered) { l.Plan.WeightFormat = "fp8" }},
		{"CacheFormat.Record", v41, nil, func(l *plan.Lowered) { l.Plan.Cache.Record = "nvfp4_ds_mla" }},
		{"CacheFormat.ElementBytes", dense, nil, func(l *plan.Lowered) { l.Plan.Cache.ElementBytes = 1 }},
		{"CacheFormat.LatentCellBytes", latent, nil, func(l *plan.Lowered) { l.Plan.Cache.LatentCellBytes = 656 }},
		{"CacheFormat.StateElementBytes", hybrid, nil, func(l *plan.Lowered) { l.Plan.Cache.StateElementBytes = 2 }},
		{"CacheFormat.Recurrent", hybrid, nil, func(l *plan.Lowered) { l.Plan.Cache.Recurrent = plan.RecurrentEveryPage }},
		{"CacheFormat.TrackingSlots", hybrid, func(l *plan.Lowered) {
			l.Plan.Cache.Recurrent, l.Plan.Cache.TrackingSlots = plan.RecurrentTracked, 1
		}, func(l *plan.Lowered) { l.Plan.Cache.TrackingSlots = 2 }},
		{"MoEPlan.Dispatch", routed, nil, func(l *plan.Lowered) {
			l.Plan.MoE.Dispatch = plan.DispatchDeepEPLowLatency
		}},
		{"MoEPlan.Experts", hopperFP8, nil, func(l *plan.Lowered) {
			l.Plan.MoE.Experts = pinnedBuild(t, d, "gemm_floor_fp8_grouped_triton", "h200")
		}},
		{"MoEPlan.Fused", muse, nil, func(l *plan.Lowered) { l.Plan.MoE.Fused = true }},
		{"MoEPlan.Routing", routed, nil, func(l *plan.Lowered) { l.Plan.MoE.Routing = plan.RoutingSeparate }},
		{"MoEPlan.RedundantExperts", routed, nil, func(l *plan.Lowered) { l.Plan.MoE.RedundantExperts = 16 }},
		{"CollectivePlan.SequenceParallelMoE", spMoE, nil, func(l *plan.Lowered) { l.Plan.Collectives.SequenceParallelMoE = true }},
		{"CollectivePlan.SequenceParallelModel", v41, nil, func(l *plan.Lowered) { l.Plan.Collectives.SequenceParallelModel = true }},
		{"DecodeContextPlan.Combine", dcp, nil, func(l *plan.Lowered) { l.Plan.DecodeContext.Combine = plan.CombineAllToAll }},
		{"DecodeContextPlan.Interleave", dcp, nil, func(l *plan.Lowered) { l.Plan.DecodeContext.Interleave = 16 }},
		{"PoolPlan.Microbatch", v41, nil, func(l *plan.Lowered) { l.Plan.Microbatch = true }},
		{"EngramPlacement.Host", v41, nil, func(l *plan.Lowered) { l.Plan.Engram.Host = false }},
		{"EngramPlacement.Shards", v41ep, func(l *plan.Lowered) { l.Plan.Engram.Host = true },
			func(l *plan.Lowered) { l.Plan.Engram.Shards = 4 }},
		{"GraphPolicy.Decode", dense, nil, func(l *plan.Lowered) { l.Plan.Graphs.Decode = plan.LaunchPiecewise }},
		{"GraphPolicy.Other", dense, nil, func(l *plan.Lowered) { l.Plan.Graphs.Other = plan.LaunchGraph }},
		{"GraphPolicy.Ceiling", dense, nil, func(l *plan.Lowered) { l.Plan.Graphs.Ceiling = 32 }},
		{"GraphPolicy.Buckets", v41, func(l *plan.Lowered) { l.Plan.Drafts.Chain = 5; wideCapture(l) },
			func(l *plan.Lowered) { l.Plan.Graphs.Buckets = []int{1, 512} }},
		{"DraftPolicy.Chain", latent, func(l *plan.Lowered) { l.Plan.Drafts.Chain = 1 },
			func(l *plan.Lowered) { l.Plan.Drafts.Chain = 3 }},
		{"DraftPolicy.RetainedTokens", v4pro, func(l *plan.Lowered) { l.Plan.Drafts.Chain = 3; l.Plan.MaxContext = 65536 },
			func(l *plan.Lowered) { l.Plan.Drafts.RetainedTokens = 4096 }},
		{"DraftPolicy.OwnWorkspace", v41, func(l *plan.Lowered) { l.Plan.Drafts.Chain = 5 },
			func(l *plan.Lowered) { l.Plan.Drafts.OwnWorkspace = true }},
		{"PoolPlan.InFlight", ringed, func(l *plan.Lowered) { l.Plan.MaxContext = 131072; l.Plan.InFlight = 1 },
			func(l *plan.Lowered) { l.Plan.InFlight = 2 }},
	} {
		t.Run(c.field, func(t *testing.T) {
			at := c.at
			open := func(f func(*plan.Lowered)) Inputs {
				return planCase(t, d, at.model, at.chip, at.nodes, at.parallel, f)
			}
			base := prices(t, open(c.base))
			moved := prices(t, open(func(l *plan.Lowered) {
				if c.base != nil {
					c.base(l)
				}
				c.edit(l)
			}))
			if base == moved {
				t.Errorf("%s on %s at %s: changing it moves no price, so the kernel does not price the "+
					"value the plan states", c.field, at.model, at.parallel)
			}
		})
	}
}

// The compiled frontend's admission is priced at its own rate where the registry states one: the
// registry carries none today, so the case states it, as a copy of the interpreted frontend's rate
// at a fifth of it.
func TestTheCompiledFrontendIsPricedAtItsOwnRate(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	withRust := func(in Inputs) Inputs {
		out := make([]*coefficient.Set, len(in.Coefficients))
		for i, s := range in.Coefficients {
			c := *s
			c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
			for _, e := range s.Coefficients {
				if e.Name == "host_admission_per_token" && len(e.Scope.Model) == 0 {
					e.Name = "host_admission_per_token_rust_frontend"
					e.Value /= 5
					e.CI95 = nil
					c.Coefficients = append(c.Coefficients, e)
					break
				}
			}
			out[i] = &c
		}
		in.Coefficients = out
		return in
	}
	at := func(native bool) time.Duration {
		in := withRust(planCase(t, d, "llama-3.1-70b-instruct", "h100", 1, "tp: 4, pp: 1, dp: 1",
			func(l *plan.Lowered) { l.Plan.Host.NativeFrontend = native }))
		return mustKernel(t, in).AdmissionOverhead(4096)
	}
	if python, rust := at(false), at(true); rust >= python {
		t.Errorf("admission of 4096 tokens: %v through the compiled frontend, %v through the "+
			"interpreted one; the plan's NativeFrontend must select the compiled frontend's rate", rust, python)
	}
}

// An adaptive verifier runs the draft budget it chooses from the expected acceptance, so at low
// acceptance a decode step verifies fewer drafts, and prices less, than the same plan without it.
func TestAnAdaptiveVerifierThePlanStatesVerifiesFewerDrafts(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	rates, err := acceptance.MinimumVarianceRates(1.3, 5)
	if err != nil {
		t.Fatal(err)
	}
	at := func(adaptive bool) string {
		in := planCase(t, d, "deepseek-v4.1-flash", "b200", 1, "tp: 4, pp: 1, dp: 1", func(l *plan.Lowered) {
			l.Plan.Drafts.Chain, l.Plan.Drafts.OwnWorkspace, l.Plan.Drafts.Adaptive = 5, true, adaptive
		})
		in.Acceptance = &acceptance.Resolution{AcceptanceLength: 1.3, PerPositionRates: rates, RatesDerived: true}
		return prices(t, in)
	}
	if at(false) == at(true) {
		t.Error("adaptive verification moves no price at an acceptance length of 1.3, so the kernel " +
			"does not price the verifier the plan states")
	}
}

// The kernel refuses a plan that leaves out a setting it needs, or states one it cannot price,
// naming the field, and applies no default in its place (SUB-6).
func TestNewRefusesAPlanItCannotPriceByName(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	open := func(f func(*plan.Lowered)) Inputs {
		return planCase(t, d, "llama-3.1-70b-instruct", "h100", 1, "tp: 4, pp: 1, dp: 1", f)
	}
	if _, err := New(open(nil)); err != nil {
		t.Fatalf("the fixture plan is refused: %v", err)
	}
	for _, c := range []struct {
		name, want string
		edit       func(*plan.Lowered)
	}{
		{"no request limit", "Plan.Batch.Requests", func(l *plan.Lowered) { l.Plan.Batch.Requests = 0 }},
		{"no token limit", "Plan.Batch.Tokens", func(l *plan.Lowered) { l.Plan.Batch.Tokens = 0 }},
		{"no page size", "Plan.PageTokens", func(l *plan.Lowered) { l.Plan.PageTokens = 0 }},
		{"no KV element width", "Plan.Cache.ElementBytes", func(l *plan.Lowered) { l.Plan.Cache.ElementBytes = 0 }},
		{"no state element width", "Plan.Cache.StateElementBytes", func(l *plan.Lowered) { l.Plan.Cache.StateElementBytes = 0 }},
		{"a weight format no dtype names", "Plan.WeightFormat", func(l *plan.Lowered) { l.Plan.WeightFormat = "awq" }},
		{"captured graphs with no ceiling", "Plan.Graphs.Ceiling", func(l *plan.Lowered) { l.Plan.Graphs.Ceiling = 0 }},
		{"a launch word the plan does not know", "Graphs.Decode", func(l *plan.Lowered) { l.Plan.Graphs.Decode = "full" }},
		{"a drafter named and not given", "whose graph was not given", func(l *plan.Lowered) { l.Drafter = "minimax-m3-eagle3" }},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := New(open(c.edit)); err == nil || !strings.Contains(err.Error(), c.want) {
				t.Errorf("err %v, want a refusal naming %q", err, c.want)
			}
		})
	}
	// Eager launches need no capture ceiling.
	if _, err := New(open(func(l *plan.Lowered) {
		l.Plan.Graphs = plan.GraphPolicy{Decode: plan.LaunchEager, Other: plan.LaunchEager}
	})); err != nil {
		t.Errorf("an eager plan with no ceiling is refused: %v", err)
	}
}

// A stated cache the graph lists no record for, under the record the deployment runs, is sized
// at its base description, and the kernel says so in Provenance only when that record is not the
// one the engine stores an unstated cache in (plan.Lowered.DefaultCacheRecord): an engine whose
// own default is the record assumes nothing the base description does not, and "" is the model's
// own format.
func TestTheCacheRecordAssumptionFollowsTheEnginesDefaultRecord(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	assumed := func(record, def string) bool {
		t.Helper()
		k, err := New(planCase(t, d, "deepseek-v4.1-flash", "b200", 1, "tp: 4, pp: 1, dp: 1",
			func(l *plan.Lowered) { l.Plan.Cache.Record, l.DefaultCacheRecord = record, def }))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		for _, o := range k.Provenance() {
			if o.Name == "cache_record" {
				return true
			}
		}
		return false
	}
	for _, c := range []struct {
		record, def string
		want        bool
	}{
		{"nvfp4_ds_mla", "", true},
		{"nvfp4_ds_mla", "nvfp4_ds_mla", false},
		{"nvfp4_ds_mla", "fp8_ds_mla", true},
		{"", "", false},
		{"", "nvfp4_ds_mla", true},
	} {
		if got := assumed(c.record, c.def); got != c.want {
			t.Errorf("record %q, engine default %q: cache_record assumed %v, want %v",
				c.record, c.def, got, c.want)
		}
	}
}

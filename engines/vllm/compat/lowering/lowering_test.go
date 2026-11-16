// Original to blis-vllm.

package lowering

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	blisschemas "github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/units"
	"github.com/inference-sim/blis-substrate/load"

	"github.com/inference-sim/blis-vllm/compat/vllmsettings"
	vllmrules "github.com/inference-sim/blis-vllm/rules"
)

// SUB-6: the lowering resolves every setting the cost model prices, so the kernel receives a
// value for each and applies no default. Each case lowers a catalog deployment that leaves one
// setting unstated, or states it, and checks the value the plan carries: the one vLLM v0.31.0
// resolves it to, which is what the kernel resolved before the resolution moved here. A lowering
// that left the value to the cost model would hand it zero, and fails the case.

var data = mustData()

func mustData() *load.Data {
	d, err := load.Embedded()
	if err != nil {
		panic(err)
	}
	return d
}

// where is a deployment: a catalog model on a catalog chip, at a parallel layout, with the
// release's rules.
type where struct {
	model, chip, parallel string
}

// lowered lowers the deployment with the settings s states.
func lowered(t *testing.T, at where, s vllmsettings.Settings, graph ...*model.Graph) *plan.Lowered {
	t.Helper()
	l, err := tryLower(t, at, s, graph...)
	if err != nil {
		t.Fatalf("%s on %s (%s): %v", at.model, at.chip, at.parallel, err)
	}
	return l
}

func tryLower(t *testing.T, at where, s vllmsettings.Settings, graph ...*model.Graph) (*plan.Lowered, error) {
	t.Helper()
	chip, err := data.Chip(at.chip)
	if err != nil {
		t.Fatal(err)
	}
	return tryLowerOn(t, at, chip, s, graph...)
}

// tryLowerOn is tryLower on a chip the test states.
func tryLowerOn(t *testing.T, at where, chip *hardware.Chip, s vllmsettings.Settings, graph ...*model.Graph) (*plan.Lowered, error) {
	t.Helper()
	nodes, perNode := 1, ""
	if ch, err := data.Chip(at.chip); err == nil && ch.GPUsPerNode == 0 {
		perNode = "  gpus_per_node: 8\n"
	}
	body := fmt.Sprintf("kind: Scenario\nname: l\nmodel: %s\ncoefficients: [cost-model-primitives]\n"+
		"engine_version: \"0.32.0\"\ncluster:\n  hardware: %s\n  fabric: ib-400g\n  nodes: %d\n%s---\n"+
		"kind: Deployment\nname: d\npools:\n  - role: colocated\n    nodes: %d\n    parallel: {%s}\n",
		at.model, at.chip, nodes, perNode, nodes, at.parallel)
	path := filepath.Join(t.TempDir(), "s.yaml")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	sc, dep, err := blisschemas.LoadBundle(path)
	if err != nil {
		t.Fatal(err)
	}
	g, err := data.Model(at.model)
	if err != nil {
		t.Fatal(err)
	}
	if len(graph) > 0 && graph[0] != nil {
		g = graph[0]
	}
	return Lower(Input{Scenario: sc, Deployment: dep, Model: g, Draft: draftOf(graph), Chip: chip,
		Rules: vllmrules.Lookup("0.32.0"), Settings: s})
}

// noted reports the value of the note the lowering recorded for a setting, and whether it did.
func noted(l *plan.Lowered, name string) (string, bool) {
	for _, n := range l.Notes {
		if n.Name == name {
			return n.Value, true
		}
	}
	return "", false
}

// overridden reports whether the lowering recorded an override of a setting.
func overridden(l *plan.Lowered, field string) (plan.Override, bool) {
	for _, o := range l.Overrides {
		if o.Field == field {
			return o, true
		}
	}
	return plan.Override{}, false
}

func ptr[T any](v T) *T { return &v }

var (
	dense   = where{"llama-3.1-70b-instruct", "h100", "tp: 4, pp: 1, dp: 1"}
	denseA  = where{"llama-3.1-70b-instruct", "a100-80", "tp: 4, pp: 1, dp: 1"}
	denseB  = where{"llama-3.1-70b-instruct", "b200", "tp: 4, pp: 1, dp: 1"}
	moe     = where{"qwen3-30b-a3b", "h200", "tp: 1, pp: 1, dp: 8, enable_expert_parallel: true"}
	moeOne  = where{"qwen3-30b-a3b", "h200", "tp: 8, pp: 1, dp: 1, enable_expert_parallel: true"}
	dsa     = where{"glm-5", "h200", "tp: 8, pp: 1, dp: 1"}
	dsaDCP  = where{"glm-5", "h200", "tp: 8, pp: 1, dp: 1, dcp: 2"}
	v41     = where{"deepseek-v4.1-flash", "b200", "tp: 4, pp: 1, dp: 1"}
	v41h    = where{"deepseek-v4.1-flash", "h200", "tp: 4, pp: 1, dp: 1"}
	v41dp   = where{"deepseek-v4.1-flash", "b200", "tp: 1, pp: 1, dp: 4, enable_expert_parallel: true"}
	hybrid  = where{"nemotron-3.5-lightning-30b-a3b-bf16", "h100", "tp: 2, pp: 1, dp: 1"}
	latent  = where{"deepseek-v3", "h200", "tp: 8, pp: 1, dp: 1"}
	wideTP  = where{"llama-3.1-70b-instruct", "h100", "tp: 16, pp: 1, dp: 1"}
	v41sp   = where{"deepseek-v4.1-flash", "b200", "tp: 2, pp: 1, dp: 2, enable_expert_parallel: true"}
	v4pro   = where{"deepseek-v4-pro", "b200", "tp: 8, pp: 1, dp: 1"}
	minimax = where{"minimax-m3", "h200", "tp: 4, pp: 1, dp: 1"}
)

// TestTheLoweringResolvesEverySettingTheKernelPrices: one case per resolved value of the plan.
func TestTheLoweringResolvesEverySettingTheKernelPrices(t *testing.T) {
	for _, c := range []struct {
		name  string
		at    where
		set   vllmsettings.Settings
		check func(*testing.T, *plan.Lowered)
	}{
		{"an unstated block size is vLLM's default of 16", dense, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.PageTokens != 16 {
					t.Errorf("PageTokens %d, want 16", l.Plan.PageTokens)
				}
				if v, ok := noted(l, "block_size"); !ok || v != "16" {
					t.Errorf("block_size note %q (%t), want 16", v, ok)
				}
			}},
		{"an unstated block size on a DSA model is the indexer's 64", dsa, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.PageTokens != 64 {
					t.Errorf("PageTokens %d, want 64", l.Plan.PageTokens)
				}
			}},
		{"an unstated block size on DeepSeek-V4.1 is 128 on SM100", v41, vllmsettings.Settings{CacheDType: "fp8"},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.PageTokens != 128 {
					t.Errorf("PageTokens %d, want 128", l.Plan.PageTokens)
				}
			}},
		{"an unstated block size on DeepSeek-V4.1 is 64 on SM90", v41h, vllmsettings.Settings{CacheDType: "fp8"},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.PageTokens != 64 {
					t.Errorf("PageTokens %d, want 64", l.Plan.PageTokens)
				}
			}},
		{"a stated block size is kept", dense, vllmsettings.Settings{BlockSize: 32},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.PageTokens != 32 {
					t.Errorf("PageTokens %d, want the stated 32", l.Plan.PageTokens)
				}
				if _, ok := noted(l, "block_size"); ok {
					t.Error("a stated block size is noted as assumed")
				}
			}},
		{"an unstated graph mode is full for decode and piecewise otherwise", dense, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Graphs.Decode != plan.LaunchGraph || l.Plan.Graphs.Other != plan.LaunchPiecewise {
					t.Errorf("Graphs %q/%q, want graph/piecewise", l.Plan.Graphs.Decode, l.Plan.Graphs.Other)
				}
			}},
		{"a stated graph mode is kept", dense, vllmsettings.Settings{CUDAGraphMode: "full_decode_only"},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Graphs.Decode != plan.LaunchGraph || l.Plan.Graphs.Other != plan.LaunchEager {
					t.Errorf("Graphs %q/%q, want graph/eager", l.Plan.Graphs.Decode, l.Plan.Graphs.Other)
				}
			}},
		{"DeepEP high-throughput over data parallelism runs no graph", moe,
			vllmsettings.Settings{All2AllBackend: "deepep_high_throughput"},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Graphs.Decode != plan.LaunchEager || l.Plan.Graphs.Other != plan.LaunchEager || l.Plan.Graphs.Ceiling != 0 {
					t.Errorf("Graphs %+v, want no graph", l.Plan.Graphs)
				}
				if o, ok := overridden(l, "cudagraph_mode"); !ok || o.Resolved != "NONE" {
					t.Errorf("cudagraph_mode override %+v (%t), want NONE", o, ok)
				}
			}},
		{"an unstated request cap is 1024 on a part of 70 GiB or more", dense, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Batch.Requests != 1024 {
					t.Errorf("Batch.Requests %d, want 1024", l.Plan.Batch.Requests)
				}
			}},
		{"an unstated request cap is 256 on an A100", denseA, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Batch.Requests != 256 {
					t.Errorf("Batch.Requests %d, want 256", l.Plan.Batch.Requests)
				}
			}},
		{"an unstated token budget is 8192 on a part of 70 to 160 GiB", dense, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Batch.Tokens != 8192 {
					t.Errorf("Batch.Tokens %d, want 8192", l.Plan.Batch.Tokens)
				}
			}},
		{"an unstated token budget is 16384 on a part of 160 GiB or more", denseB, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Batch.Tokens != 16384 {
					t.Errorf("Batch.Tokens %d, want 16384", l.Plan.Batch.Tokens)
				}
			}},
		{"an unstated token budget is 2048 on an A100", denseA, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Batch.Tokens != 2048 {
					t.Errorf("Batch.Tokens %d, want 2048", l.Plan.Batch.Tokens)
				}
			}},
		{"an unstated token budget is 256 under a batched data-parallel MoE", moe,
			vllmsettings.Settings{All2AllBackend: "deepep_low_latency"},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Batch.Tokens != 256 {
					t.Errorf("Batch.Tokens %d, want 256", l.Plan.Batch.Tokens)
				}
			}},
		{"stated limits are kept", dense, vllmsettings.Settings{MaxNumSeqs: 48, MaxNumBatchedTokens: 4096},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Batch.Requests != 48 || l.Plan.Batch.Tokens != 4096 {
					t.Errorf("Batch %+v, want the stated 48 and 4096", l.Plan.Batch)
				}
			}},
		{"an unstated capture ceiling is min(2 x requests, 512) off SM100", dense, vllmsettings.Settings{MaxNumSeqs: 8},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Graphs.Ceiling != 16 {
					t.Errorf("Graphs.Ceiling %d, want 16", l.Plan.Graphs.Ceiling)
				}
			}},
		{"an unstated capture ceiling is 1024 on SM100", denseB, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Graphs.Ceiling != 1024 {
					t.Errorf("Graphs.Ceiling %d, want 1024", l.Plan.Graphs.Ceiling)
				}
				if len(l.Plan.Graphs.Buckets) == 0 || l.Plan.Graphs.Buckets[len(l.Plan.Graphs.Buckets)-1] != 1024 {
					t.Errorf("Graphs.Buckets %v, want the ladder up to 1024", l.Plan.Graphs.Buckets)
				}
			}},
		{"an unstated context bound is the model's window", v4pro, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				g, _ := data.Model(v4pro.model)
				if want := g.Global.DefaultWindow(); want <= 0 || l.Plan.MaxContext != units.Tokens(want) {
					t.Errorf("MaxContext %d, want the window %d", l.Plan.MaxContext, want)
				}
			}},
		{"a stated context bound is kept", dense, vllmsettings.Settings{MaxModelLen: 4096},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.MaxContext != 4096 {
					t.Errorf("MaxContext %d, want 4096", l.Plan.MaxContext)
				}
			}},
		{"a hybrid model with prefix caching keeps page-aligned states", hybrid, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Cache.Recurrent != plan.RecurrentPageAligned {
					t.Errorf("Cache.Recurrent %q, want page_aligned", l.Plan.Cache.Recurrent)
				}
			}},
		{"a hybrid model without prefix caching keeps the latest state", hybrid,
			vllmsettings.Settings{EnablePrefixCaching: ptr(false)},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Cache.Recurrent != plan.RecurrentLatest {
					t.Errorf("Cache.Recurrent %q, want latest", l.Plan.Cache.Recurrent)
				}
			}},
		{"the recurrent state is fp32 unless stated", hybrid, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Cache.StateElementBytes != 4 {
					t.Errorf("Cache.StateElementBytes %v, want 4", l.Plan.Cache.StateElementBytes)
				}
			}},
		{"an auto cache is the model's compute width", dense, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Cache.ElementBytes != 2 {
					t.Errorf("Cache.ElementBytes %v, want 2", l.Plan.Cache.ElementBytes)
				}
			}},
		{"an fp8 cache is one byte", dense, vllmsettings.Settings{CacheDType: "fp8"},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Cache.ElementBytes != 1 || l.Plan.Cache.Record != "fp8" {
					t.Errorf("Cache %+v, want 1 byte under the fp8 record", l.Plan.Cache)
				}
			}},
		{"DeepSeek-V4.1's attention backend serves an fp8 cache as its packed record", v41, vllmsettings.Settings{CacheDType: "fp8"},
			func(t *testing.T, l *plan.Lowered) {
				if !strings.HasSuffix(l.Plan.Cache.Record, "_ds_mla") {
					t.Errorf("Cache.Record %q, want a packed _ds_mla record", l.Plan.Cache.Record)
				}
			}},
		{"a DSA model's fp8 cache is the 656-byte packed cell", dsa, vllmsettings.Settings{CacheDType: "fp8"},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Cache.LatentCellBytes != 656 {
					t.Errorf("Cache.LatentCellBytes %v, want 656", l.Plan.Cache.LatentCellBytes)
				}
			}},
		{"fp8 quantization serves a bf16 checkpoint in fp8", dense, vllmsettings.Settings{Quantization: "fp8"},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.WeightFormat != "fp8" {
					t.Errorf("WeightFormat %q, want fp8", l.Plan.WeightFormat)
				}
			}},
		{"an unstated quantization serves the checkpoint as stored", dense, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				g, _ := data.Model(dense.model)
				if l.Plan.WeightFormat != string(g.Global.WeightDType) {
					t.Errorf("WeightFormat %q, want the checkpoint's %q", l.Plan.WeightFormat, g.Global.WeightDType)
				}
			}},
		{"an unstated decode-context combine is the all-gather and reduce-scatter", dsaDCP,
			vllmsettings.Settings{CacheDType: "fp8"},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.DecodeContext.Combine != plan.CombineGatherScatter || l.Plan.DecodeContext.Interleave != 1 {
					t.Errorf("DecodeContext %+v, want the gather-scatter combine at interleave 1", l.Plan.DecodeContext)
				}
				if v, ok := noted(l, "dcp_comm_backend"); !ok || v != "ag_rs" {
					t.Errorf("dcp_comm_backend note %q (%t), want ag_rs", v, ok)
				}
			}},
		{"asynchronous scheduling is on unless stated off", dense, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.InFlight != 2 {
					t.Errorf("InFlight %d, want 2", l.Plan.InFlight)
				}
			}},
		{"asynchronous scheduling stated off", dense, vllmsettings.Settings{AsyncScheduling: ptr(false)},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.InFlight != 1 {
					t.Errorf("InFlight %d, want 1", l.Plan.InFlight)
				}
			}},
		{"DeepEP low-latency is the dispatch kernel it names", moe,
			vllmsettings.Settings{All2AllBackend: "deepep_low_latency"},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.MoE.Dispatch != plan.DispatchDeepEPLowLatency {
					t.Errorf("MoE.Dispatch %+v, want DeepEP low-latency", l.Plan.MoE.Dispatch)
				}
			}},
		{"the gather-and-scatter dispatch runs no dispatch kernel", moe,
			vllmsettings.Settings{All2AllBackend: "allgather_reducescatter"},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.MoE.Dispatch != (plan.KernelImpl{}) {
					t.Errorf("MoE.Dispatch %+v, want none", l.Plan.MoE.Dispatch)
				}
			}},
		{"a stated dispatch backend with no dispatch to run is overridden", moeOne,
			vllmsettings.Settings{All2AllBackend: "deepep_low_latency"},
			func(t *testing.T, l *plan.Lowered) {
				if o, ok := overridden(l, "all2all_backend"); !ok || o.Resolved != "none" {
					t.Errorf("all2all_backend override %+v (%t), want none", o, ok)
				}
			}},
		{"expert load balancing places its redundant experts", moe,
			vllmsettings.Settings{EPLB: &vllmsettings.EPLB{Enabled: true, NumRedundantExperts: 16}},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.MoE.RedundantExperts != 16 {
					t.Errorf("MoE.RedundantExperts %d, want 16", l.Plan.MoE.RedundantExperts)
				}
			}},
		{"disabled load balancing places none", moe,
			vllmsettings.Settings{EPLB: &vllmsettings.EPLB{Enabled: false, NumRedundantExperts: 16}},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.MoE.RedundantExperts != 0 {
					t.Errorf("MoE.RedundantExperts %d, want 0", l.Plan.MoE.RedundantExperts)
				}
			}},
		{"the mega-MoE backend fuses the dispatch and the experts", v41dp,
			vllmsettings.Settings{CacheDType: "fp8", MoEBackend: "deep_gemm_mega_moe"},
			func(t *testing.T, l *plan.Lowered) {
				if !l.Plan.MoE.Fused {
					t.Error("MoE.Fused false, want true")
				}
			}},
		{"DeepSeek-V4.1 splits its residual stream across the tensor-parallel ranks", v41sp,
			vllmsettings.Settings{CacheDType: "fp8"},
			func(t *testing.T, l *plan.Lowered) {
				if !l.Plan.Collectives.SequenceParallelModel {
					t.Error("Collectives.SequenceParallelModel false, want true")
				}
			}},
		{"Engram tables live where the release's default puts them", v41, vllmsettings.Settings{CacheDType: "fp8"},
			func(t *testing.T, l *plan.Lowered) {
				if !l.Plan.Engram.Host || l.Plan.Engram.Shards != 4 {
					t.Errorf("Engram %+v, want host memory split four ways", l.Plan.Engram)
				}
				if _, ok := noted(l, "engram_cpu_offload"); !ok {
					t.Error("no engram_cpu_offload note for an unstated placement")
				}
			}},
		{"stated Engram tables on the GPU", v41,
			vllmsettings.Settings{CacheDType: "fp8", EngramConfig: &vllmsettings.EngramConfig{CPUOffload: ptr(false)}},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Engram.Host {
					t.Error("Engram.Host true, want the stated GPU placement")
				}
			}},
		{"a drafter's chain and its adaptive verifier", v41,
			vllmsettings.Settings{CacheDType: "fp8", Speculative: &vllmsettings.Speculative{Method: "dspark",
				NumSpecTokens: 5, EnableAdaptiveVerification: true}},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Drafts.Chain != 5 || !l.Plan.Drafts.Adaptive || !l.Plan.Drafts.OwnWorkspace {
					t.Errorf("Drafts %+v, want chain 5, adaptive, own workspace", l.Plan.Drafts)
				}
			}},
		{"a single-module MTP drafter retains no tokens", minimax,
			vllmsettings.Settings{Speculative: &vllmsettings.Speculative{Method: "mtp", NumSpecTokens: 3}},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Drafts.Chain != 3 || l.Plan.Drafts.RetainedTokens != 0 {
					t.Errorf("Drafts %+v, want chain 3 and nothing retained", l.Plan.Drafts)
				}
			}},
		{"the Rust frontend is the compiled one", dense, vllmsettings.Settings{Frontend: vllmsettings.FrontendRust},
			func(t *testing.T, l *plan.Lowered) {
				if !l.Plan.Host.NativeFrontend {
					t.Error("Host.NativeFrontend false, want true")
				}
			}},
		{"dual-batch overlap micro-batches", dense, vllmsettings.Settings{DBO: &vllmsettings.DBO{Enabled: true}},
			func(t *testing.T, l *plan.Lowered) {
				if !l.Plan.Microbatch {
					t.Error("Microbatch false, want true")
				}
			}},
		{"the memory share is the one stated", dense, vllmsettings.Settings{GPUMemoryUtilization: 0.85},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.MemoryFraction != 0.85 {
					t.Errorf("MemoryFraction %v, want 0.85", l.Plan.MemoryFraction)
				}
			}},
		{"cascade attention is off unless stated on", dense, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				if l.Reported.CascadeAttention {
					t.Error("Reported.CascadeAttention true, want false")
				}
			}},
		{"the custom all-reduce runs within a node", dense, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				if !l.Reported.SMAllReduce {
					t.Error("Reported.SMAllReduce false, want true")
				}
			}},
		{"the custom all-reduce is declined across nodes", wideTP, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				if l.Reported.SMAllReduce {
					t.Error("Reported.SMAllReduce true, want false across nodes")
				}
				if _, ok := overridden(l, "allreduce_backend"); !ok {
					t.Error("no allreduce_backend override")
				}
			}},
		{"the record an unstated cache runs under is the release's default", v41, vllmsettings.Settings{CacheDType: "fp8"},
			func(t *testing.T, l *plan.Lowered) {
				if !strings.HasSuffix(l.DefaultCacheRecord, "_ds_mla") {
					t.Errorf("DefaultCacheRecord %q, want DeepSeek-V4.1's packed record", l.DefaultCacheRecord)
				}
			}},
		{"a dense model's unstated cache runs under its own format", dense, vllmsettings.Settings{},
			func(t *testing.T, l *plan.Lowered) {
				if l.DefaultCacheRecord != "" && l.DefaultCacheRecord != "auto" {
					t.Errorf("DefaultCacheRecord %q, want the model's own", l.DefaultCacheRecord)
				}
			}},
		{"a latent model's MTP drafter keeps its chain", latent,
			vllmsettings.Settings{Speculative: &vllmsettings.Speculative{Method: "deepseek_mtp", NumSpecTokens: 1}},
			func(t *testing.T, l *plan.Lowered) {
				if l.Plan.Drafts.Chain != 1 || l.Plan.Drafts.RetainedTokens != 0 {
					t.Errorf("Drafts %+v, want chain 1 and nothing retained", l.Plan.Drafts)
				}
			}},
	} {
		t.Run(c.name, func(t *testing.T) { c.check(t, lowered(t, c.at, c.set)) })
	}
}

// The lowering refuses what vLLM v0.31.0 refuses at startup, as the cost model did.
func TestTheLoweringRefusesWhatTheEngineRefuses(t *testing.T) {
	for _, c := range []struct {
		name, want string
		at         where
		set        vllmsettings.Settings
	}{
		{"an unknown graph mode", "cudagraph_mode", dense, vllmsettings.Settings{CUDAGraphMode: "sometimes"}},
		{"a block size the DSA indexer cannot run", "block_size 48", dsa, vllmsettings.Settings{BlockSize: 48}},
		{"a cache dtype the release does not have", "cache_dtype", dense, vllmsettings.Settings{CacheDType: "fp7"}},
		{"a quantization the cost model does not price", "quantization", dense, vllmsettings.Settings{Quantization: "nf3"}},
		{"DeepSeek-V4.1 with a bf16 cache", "cache_dtype", v41, vllmsettings.Settings{CacheDType: "bfloat16"}},
		{"a recurrent cache mode the cost model does not price", "mamba_cache_mode", hybrid,
			vllmsettings.Settings{MambaCacheMode: "sometimes"}},
		{"a replicated query projection on a latent model", "dcp_q_replicate", dsaDCP,
			vllmsettings.Settings{CacheDType: "fp8", DCPQReplicate: ptr(true)}},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := tryLower(t, c.at, c.set); err == nil || !strings.Contains(err.Error(), c.want) {
				t.Errorf("err %v, want a refusal naming %q", err, c.want)
			}
		})
	}
}

// A multi-module MTP drafter reads num_speculative_tokens - 1 tokens ahead, which a request's KV
// keeps (extra_retained_tokens, vllm/v1/core/kv_cache_utils.py:2738-2744). No catalog graph states
// one, so the case gives MiniMax-M3's speculator three modules.
func TestAMultiModuleMTPDrafterRetainsTheDraftedTokensLessOne(t *testing.T) {
	g, err := data.Model(minimax.model)
	if err != nil {
		t.Fatal(err)
	}
	three := *g
	sp := *g.Speculator
	sp.Stack.Repeat = 3
	three.Speculator = &sp
	l := lowered(t, minimax, vllmsettings.Settings{Speculative: &vllmsettings.Speculative{Method: "mtp",
		NumSpecTokens: 3}}, &three)
	if l.Plan.Drafts.RetainedTokens != 2 {
		t.Errorf("Drafts.RetainedTokens %d, want 2", l.Plan.Drafts.RetainedTokens)
	}
}

// draftOf is the drafter graph a test passes after the target's, or nil.
func draftOf(graph []*model.Graph) *model.Graph {
	if len(graph) > 1 {
		return graph[1]
	}
	return nil
}

// tryLowerWithDraft lowers the deployment with the target and a named drafter.
func tryLowerWithDraft(t *testing.T, at where, target, draft *model.Graph, s vllmsettings.Settings) (*plan.Lowered, error) {
	t.Helper()
	return tryLower(t, at, s, target, draft)
}

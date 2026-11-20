// Original to blis-vllm.

package lower

import (
	"encoding/json"
	"os"
	"slices"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"

	"github.com/inference-sim/blis-vllm/internal/upstream"
)

// muse_fusions.json is vLLM v0.33.0's own decisions for each case, printed by
// oracle/musefusion/oracle.py, which runs the release's gating code on a CPU with only the GPU,
// FlashInfer's workspaces and the process group replaced.
type museOracle struct {
	Commit  string `json:"commit"`
	HeadDim int    `json:"head_dim"`
	Tokens  []int  `json:"tokens"`
	Cases   []struct {
		Case struct {
			Cap        int    `json:"cap"`
			TP         int    `json:"tp"`
			PP         int    `json:"pp"`
			DP         int    `json:"dp"`
			PCP        int    `json:"pcp"`
			DCP        int    `json:"dcp"`
			EP         bool   `json:"ep"`
			KV         string `json:"kv_cache_dtype"`
			SpansNodes bool   `json:"spans_nodes"`
			Multicast  bool   `json:"multicast"`
		} `json:"case"`
		QK          bool  `json:"qk_norm_rope_kv_insert"`
		AllReduce   bool  `json:"moe_finalize_all_reduce"`
		AllReduceTo int   `json:"moe_finalize_all_reduce_max_tokens"`
		Boundary    bool  `json:"column_gather_boundary"`
		ShardedAt   []int `json:"unpooler_sharded_at"`
		DeferredAt  []int `json:"gather_deferred_at"`
	} `json:"cases"`
}

func TestMuseSparkFusionsMatchTheRelease(t *testing.T) {
	b, err := os.ReadFile("testdata/muse_fusions.json")
	if err != nil {
		t.Fatal(err)
	}
	var o museOracle
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	if o.Commit != upstream.Commit {
		t.Fatalf("muse_fusions.json was recorded at %s, the port models %s", o.Commit, upstream.Commit)
	}
	if len(o.Cases) == 0 || len(o.Tokens) == 0 {
		t.Fatal("muse_fusions.json has no cases")
	}
	for _, c := range o.Cases {
		in := c.Case
		l := MuseSparkLayout{CapabilityMajor: in.Cap, TP: in.TP, PP: in.PP, DP: in.DP, PCP: in.PCP, DCP: in.DCP,
			ExpertParallel: in.EP, KVCacheDType: in.KV, TPSpansNodes: in.SpansNodes, Multicast: in.Multicast}
		f, err := MuseSparkFusionsFor(o.HeadDim, l)
		if err != nil {
			t.Errorf("%+v: %v", in, err)
			continue
		}
		if f.QKNormRopeKVInsert != c.QK {
			t.Errorf("%+v: QK-norm, RoPE and KV insert fused %v, vLLM %v", in, f.QKNormRopeKVInsert, c.QK)
		}
		if (f.MoEFinalizeAllReduceMaxTokens > 0) != c.AllReduce || f.MoEFinalizeAllReduceMaxTokens != c.AllReduceTo {
			t.Errorf("%+v: fused finalize all-reduce up to %d tokens, vLLM %v up to %d", in,
				f.MoEFinalizeAllReduceMaxTokens, c.AllReduce, c.AllReduceTo)
		}
		if f.ColumnGatherBoundary != c.Boundary {
			t.Errorf("%+v: fused gather boundary %v, vLLM %v", in, f.ColumnGatherBoundary, c.Boundary)
		}
		for _, n := range o.Tokens {
			if got, want := f.UnpoolerSharded(n), slices.Contains(c.ShardedAt, n); got != want {
				t.Errorf("%+v, %d tokens: unpooler sharded %v, vLLM %v", in, n, got, want)
			}
			if got, want := f.GatherDeferred(n), slices.Contains(c.DeferredAt, n); got != want {
				t.Errorf("%+v, %d tokens: gather deferred %v, vLLM %v", in, n, got, want)
			}
		}
	}
}

func museGraph(t *testing.T) *model.Graph {
	t.Helper()
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	g, err := d.Model("muse-spark-open-816b-a42b-nvfp4")
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// GB200 NVL72 at TP8 with expert parallelism: four GPUs to a tray, so the TP group spans two
// hosts, inside one NVLink domain with multicast.
var gb200TEP8 = MuseSparkLayout{CapabilityMajor: 10, TP: 8, PP: 1, DP: 1, ExpertParallel: true,
	KVCacheDType: "auto", TPSpansNodes: true, Multicast: true}

func stepChange(t *testing.T, g *model.Graph, l MuseSparkLayout, tokens int) int {
	t.Helper()
	n, err := MuseSparkStepLaunchChange(g, l, tokens)
	if err != nil {
		t.Fatal(err)
	}
	return n
}

// The test #161 asks for: the same deployment launches fewer kernels at PP1 than at PP2, where the
// gather is not fused across layers, and fewer than on an SM90 part, where none of the Blackwell
// fusions is built.
func TestMuseSparkBlackwellPP1LaunchesFewerKernels(t *testing.T) {
	g := museGraph(t)
	pp2 := gb200TEP8
	pp2.PP = 2
	h200 := gb200TEP8
	h200.CapabilityMajor, h200.Multicast = 9, false
	for _, tokens := range []int{1, 32, 128} {
		pp1 := stepChange(t, g, gb200TEP8, tokens)
		// Below 32 tokens a TP group across hosts does not shard the unpooler
		// (min_sharded_unpooler_tokens), so there is no gather for PP1 to fuse.
		if got := stepChange(t, g, pp2, tokens); tokens >= 32 && pp1 >= got {
			t.Errorf("%d tokens: PP1 changes the step's launches by %d, PP2 by %d; PP1 should launch fewer", tokens, pp1, got)
		} else if tokens < 32 && pp1 != got {
			t.Errorf("%d tokens: PP1 changes the step's launches by %d, PP2 by %d; with no gather they should agree", tokens, pp1, got)
		}
		if got := stepChange(t, g, h200, tokens); pp1 >= got {
			t.Errorf("%d tokens: Blackwell changes the step's launches by %d, SM90 by %d; Blackwell should launch fewer",
				tokens, pp1, got)
		}
		if pp1 >= 0 {
			t.Errorf("%d tokens: GB200 TP8+EP PP1 launches %d kernels more than the graph counts; its fusions remove launches",
				tokens, pp1)
		}
	}
}

// Past its token bound the fused finalize all-reduce gives way to a separate all-reduce, so a
// step just above the bound launches more kernels than one at it.
func TestMuseSparkFusedAllReduceStopsAtItsBound(t *testing.T) {
	g := museGraph(t)
	f, err := MuseSparkFusionsFor(64, gb200TEP8)
	if err != nil {
		t.Fatal(err)
	}
	bound := f.MoEFinalizeAllReduceMaxTokens
	if bound <= 0 {
		t.Fatal("GB200 TP8+EP builds no fused finalize all-reduce")
	}
	if at, above := stepChange(t, g, gb200TEP8, bound), stepChange(t, g, gb200TEP8, bound+1); above <= at {
		t.Errorf("at %d tokens the step's launches change by %d, at %d by %d; the separate all-reduce should add launches",
			bound, at, bound+1, above)
	}
	oneHost := gb200TEP8
	oneHost.TPSpansNodes = false
	h, err := MuseSparkFusionsFor(64, oneHost)
	if err != nil {
		t.Fatal(err)
	}
	if h.MoEFinalizeAllReduceMaxTokens <= bound {
		t.Errorf("a TP group on one host fuses up to %d tokens, one across hosts up to %d; the single host should reach further",
			h.MoEFinalizeAllReduceMaxTokens, bound)
	}
}

// An unfused, sharded unpooler gathers its columns before the boundary, so at PP2 the step makes
// more launches than the graph counts; the fused boundary never adds one.
func TestMuseSparkUnfusedGatherAddsLaunches(t *testing.T) {
	g := museGraph(t)
	pp2 := gb200TEP8
	pp2.PP, pp2.ExpertParallel = 2, false
	if n := stepChange(t, g, pp2, 64); n <= 0 {
		t.Errorf("GB200 TP8 PP2 at 64 tokens changes the step's launches by %d; its unfused gathers should add launches", n)
	}
	f, err := MuseSparkFusionsFor(64, gb200TEP8)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range MuseSparkLayerLaunches(f, 64, true) {
		if n.Role == "unpooler gather" && n.Engine != 0 {
			t.Errorf("the fused boundary leaves %d launches for the unpooler's gather", n.Engine)
		}
	}
}

func TestMuseSparkFusionsRefuseWhatVLLMDoesNotBuild(t *testing.T) {
	hopperFP4 := gb200TEP8
	hopperFP4.CapabilityMajor, hopperFP4.Multicast, hopperFP4.KVCacheDType = 9, false, "nvfp4"
	if _, err := MuseSparkFusionsFor(64, hopperFP4); err == nil {
		t.Error("an NVFP4 KV cache on SM90 resolved; vLLM finds no attention backend for it")
	}
	other := gb200TEP8
	other.CapabilityMajor = 12
	if _, err := MuseSparkFusionsFor(64, other); err == nil {
		t.Error("compute capability 12 resolved; its backend priority list is not ported")
	}
	if _, err := MuseSparkStepLaunchChange(&model.Graph{}, gb200TEP8, 1); err == nil {
		t.Error("a graph of another architecture resolved Muse Spark's fusions")
	}
}

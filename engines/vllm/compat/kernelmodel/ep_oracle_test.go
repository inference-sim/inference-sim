// Original to blis-vllm.

package kernelmodel

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"

	"github.com/inference-sim/blis-vllm/rules/v0_32"
)

// moeCase is one case of core/lower/testdata/vllm_lower.json's moe_parallel section
// (oracle/lower/extract.py moe_parallel_cases): vLLM v0.32.0's own FusedMoEParallelConfig.make,
// ParallelConfig.use_sequence_parallel_moe and determine_expert_map, run for every rank of a
// tp x pcp x dp layout with enable_expert_parallel on or off.
type moeCase struct {
	TP                   int            `json:"tp"`
	DP                   int            `json:"dp"`
	PCP                  int            `json:"pcp"`
	EnableExpertParallel bool           `json:"enable_expert_parallel"`
	All2AllBackend       string         `json:"all2all_backend"`
	EPSize               int            `json:"ep_size"`
	UseEP                bool           `json:"use_ep"`
	SequenceParallelMoE  bool           `json:"sequence_parallel_moe"`
	UseAll2AllKernels    bool           `json:"use_all2all_kernels"`
	ExpertsMost          map[string]int `json:"experts_most"`
	ExpertsFewest        map[string]int `json:"experts_fewest"`
}

func loadMoEOracle(t *testing.T) map[string]moeCase {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "core", "lower", "testdata", "vllm_lower.json"))
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		MoE []moeCase `json:"moe_parallel"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	out := map[string]moeCase{}
	for _, c := range f.MoE {
		out[moeKey(c.TP, c.DP, c.PCP, c.EnableExpertParallel, c.All2AllBackend)] = c
	}
	if len(out) < 384 {
		t.Fatalf("%d MoE cases in the oracle file, want the full sweep", len(out))
	}
	return out
}

func moeKey(tp, dp, pcp int, ep bool, backend string) string {
	return fmt.Sprintf("tp%d dp%d pcp%d ep=%v %s", tp, dp, pcp, ep, backend)
}

// An engine setting enable_expert_parallel true is read three ways, and each is vLLM's own.
//
// What it states: the expert-parallel group is tp x pcp x dp ranks wide and shards experts whole
// (FusedMoEParallelConfig.make, vllm/model_executor/layers/fused_moe/config.py:1108-1236@5df63a72);
// off, the width is 1 and every expert is sliced across the ranks. The MoE input is then
// sequence-parallel only when the backend is one of those that supports it and tp and dp both
// exceed 1 (ParallelConfig.use_sequence_parallel_moe, vllm/config/parallel.py:765-782), which
// replaces the layer's all-reduce with a reduce-scatter and all-gather; and all2all kernels are
// built only when expert parallelism has dp, pcp or a sequence-parallel input to dispatch across
// (use_all2all_kernels, config.py:1025-1029), so a stated backend with none of those takes no
// effect. An unstated backend is allgather_reducescatter (vllm/config/parallel.py:219).
//
// Part 1 compares the rules pack and the deployment schema with the oracle for all 384 layouts
// (tp 1 to 8 x dp 1 to 9 x pcp 1 to 2 x four backends x on and off). Part 2 builds the substrate's kernel
// for the stated shapes of the dashboard's deployments (MiniMax-M2.5 on a B200 node at tp2 dp2
// with expert parallelism, MiniMax-M3 and DeepSeek-V4 at tp8 with dp 1 to 8) by swapping tp, dp,
// enable_expert_parallel and all2all_backend into committed fixtures, and compares what the
// kernel resolved: the expert-parallel width, whether the MoE input is sequence-parallel, and the
// all2all_backend override it records when the stated backend cannot take effect.
//
// Limit: blis-vllm emits no plan.Layout.Expert yet and core/lower has no layout code, so these go
// through the kernel (Resolved()) and the pack, not through a lowering step of this repository.
func TestExpertLayoutIsVLLMs(t *testing.T) {
	oracle := loadMoEOracle(t)
	pack := v0_32.Pack()

	t.Run("pack and schema", func(t *testing.T) {
		if got := pack.EffectiveAll2AllBackend(""); got != "allgather_reducescatter" {
			t.Errorf("an unstated all2all_backend runs as %q, vLLM's default is allgather_reducescatter", got)
		}
		for _, c := range oracle {
			par := deployment.Parallelism{TP: c.TP, DP: c.DP, PCP: c.PCP, EnableExpertParallel: c.EnableExpertParallel}
			if got := par.ExpertParallelWidth(); got != c.EPSize {
				t.Errorf("tp%d dp%d pcp%d ep=%v: expert-parallel width %d, vLLM's ep_size %d", c.TP, c.DP, c.PCP, c.EnableExpertParallel, got, c.EPSize)
			}
			if got := pack.SequenceParallelMoE(c.All2AllBackend, c.EnableExpertParallel, c.TP, c.DP); got != c.SequenceParallelMoE {
				t.Errorf("%s: SequenceParallelMoE = %v, vLLM's use_sequence_parallel_moe %v",
					moeKey(c.TP, c.DP, c.PCP, c.EnableExpertParallel, c.All2AllBackend), got, c.SequenceParallelMoE)
			}
		}
	})

	type shape struct{ tp, dp int }
	for _, f := range []struct {
		fixture  string
		shapes   []shape
		backends []string
	}{
		// MiniMax-M2.5 on one B200 node: the dashboard's tp2 ep dp2, and the neighbouring splits
		// of the same eight GPUs.
		{"aisimulate/minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2.yaml",
			[]shape{{2, 1}, {4, 1}, {8, 1}, {2, 2}, {1, 4}, {4, 2}, {2, 4}, {1, 8}}, []string{"", "deepep_low_latency", "deepep_high_throughput"}},
		// MiniMax-M3 and DeepSeek-V4-Pro on B200 at tp8 with dp 1, 2, 4 and 8 (DeepSeek-V4's
		// deepep_low_latency buffers do not fit the card: the kernel refuses it, which is a
		// memory rule, not a layout one).
		{"direct/minimax-m3-b200-fp4-vllm-tp8-ep8-dp2.yaml",
			[]shape{{8, 1}, {8, 2}, {8, 4}, {8, 8}, {4, 1}, {4, 2}}, []string{"", "deepep_low_latency", "deepep_high_throughput"}},
		{"direct/deepseek-v4-pro-b200-fp4-vllm-tp8-ep8-dp2-cfg1.yaml",
			[]shape{{8, 1}, {8, 2}, {8, 4}, {8, 8}, {4, 2}}, []string{"", "deepep_high_throughput"}},
	} {
		for _, s := range f.shapes {
			for _, ep := range []bool{false, true} {
				for _, be := range f.backends {
					name := fmt.Sprintf("%s/%s", filepath.Base(f.fixture), moeKey(s.tp, s.dp, 1, ep, be))
					t.Run(name, func(t *testing.T) {
						t.Parallel()
						m, err := openLayout(t, f.fixture, s.tp, s.dp, ep, be)
						if err != nil {
							t.Fatal(err)
						}
						stated := be
						if stated == "" {
							stated = "allgather_reducescatter"
						}
						want := oracle[moeKey(s.tp, s.dp, 1, ep, stated)]
						r := m.Kernel().Resolved()
						if r.ExpertParallelWidth != want.EPSize {
							t.Errorf("ExpertParallelWidth = %d, vLLM's ep_size %d", r.ExpertParallelWidth, want.EPSize)
						}
						if r.SequenceParallelMoE != want.SequenceParallelMoE {
							t.Errorf("SequenceParallelMoE = %v, vLLM's use_sequence_parallel_moe %v", r.SequenceParallelMoE, want.SequenceParallelMoE)
						}
						// A stated backend takes no effect when expert parallelism builds no all2all
						// kernels: the kernel records that as an override of all2all_backend.
						override := false
						for _, o := range r.Overrides {
							override = override || o.Field == "all2all_backend"
						}
						if wantOverride := be != "" && ep && !want.UseAll2AllKernels; override != wantOverride {
							t.Errorf("all2all_backend override = %v, want %v (vLLM's use_all2all_kernels %v)", override, wantOverride, want.UseAll2AllKernels)
						}
					})
				}
			}
		}
	}

	// The heaviest rank's experts. The kernel prices the rank with the most (its expertsPerRank
	// is private; the pinned substrate has no accessor, so a one-line ExpertsPerRank() on the
	// kernel would make this direct). Its weights are a fixed part plus experts-per-rank times a
	// per-expert part, so over three widths at tp8 the differences of the weights are in the
	// ratio of the differences of the experts per rank: MiniMax-M2.5's 256 experts hold 32, 16
	// and, at the 72 ranks of dp9, 4 on the heaviest rank and 3 on the lightest
	// (determine_expert_map, vllm/model_executor/layers/fused_moe/expert_map_manager.py:22-98).
	// Pricing the lightest rank would give 16/13 for the second ratio, not 16/12.
	t.Run("experts on the heaviest rank", func(t *testing.T) {
		r := repos()
		r.Scenarios = filepath.Join("..", "..", "kerneltests", "testdata")
		var weights [3]float64
		var most, fewest [3]int
		for i, c := range []struct {
			fixture string
			width   int
		}{{"minimax-m25-h200-ep8.yaml", 8}, {"minimax-m25-h200-ep16.yaml", 16}, {"minimax-m25-h200-ep72.yaml", 72}} {
			m, err := Open(c.fixture, r)
			if err != nil {
				t.Fatal(err)
			}
			if got := m.Kernel().Resolved().ExpertParallelWidth; got != c.width {
				t.Fatalf("%s: expert-parallel width %d, want %d", c.fixture, got, c.width)
			}
			weights[i] = float64(m.Kernel().FixedBytes().Weights)
			o := oracle[moeKey(8, c.width/8, 1, true, "allgather_reducescatter")]
			most[i], fewest[i] = o.ExpertsMost["256"], o.ExpertsFewest["256"]
		}
		if most != [3]int{32, 16, 4} || fewest != [3]int{32, 16, 3} {
			t.Fatalf("vLLM's experts per rank at widths 8, 16, 72: most %v, fewest %v", most, fewest)
		}
		got := (weights[0] - weights[1]) / (weights[1] - weights[2])
		want := float64(most[0]-most[1]) / float64(most[1]-most[2])
		if math.Abs(got-want) > 1e-6*want {
			t.Errorf("weights differ in ratio %.6f across ep8, ep16, ep72; the heaviest rank's experts (32, 16, 4) give %.6f (the lightest rank's would give %.6f)",
				got, want, float64(fewest[0]-fewest[1])/float64(fewest[1]-fewest[2]))
		}
	})
}

// openLayout opens a committed fixture with its tensor-parallel width, data-parallel width,
// enable_expert_parallel and all2all_backend replaced, on as many nodes of its cluster as the
// layout needs (an InfiniBand fabric is added for more than one).
func openLayout(t *testing.T, fixture string, tp, dp int, ep bool, backend string) (*Model, error) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "kerneltests", "testdata", fixture))
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	set := func(key, val string) {
		re := regexp.MustCompile(`(?m)^(\s*)` + key + `:.*$`)
		if !re.MatchString(body) {
			t.Fatalf("%s states no %s", fixture, key)
		}
		body = re.ReplaceAllString(body, "${1}"+key+": "+val)
	}
	nodes := max(1, (tp*dp+7)/8)
	set("tp", fmt.Sprint(tp))
	set("dp", fmt.Sprint(dp))
	set("enable_expert_parallel", fmt.Sprint(ep))
	set("nodes", fmt.Sprint(nodes))
	if nodes > 1 && !regexp.MustCompile(`(?m)^\s*fabric:`).MatchString(body) {
		body = strings.Replace(body, "  gpus_per_node:", "  fabric: ib-400g\n  gpus_per_node:", 1)
	}
	if backend != "" {
		body = strings.Replace(body, "    engine:\n", "    engine:\n      all2all_backend: "+backend+"\n", 1)
	}
	r := repos()
	r.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return Open("v.yaml", r)
}

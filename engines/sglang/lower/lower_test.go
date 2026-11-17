// Original to blis-sglang.

package lower

import (
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"testing"

	"github.com/inference-sim/blis-sglang/config"
)

// lowerAnswers is lower/testdata/sglang_lower.json: SGLang's own lowering functions run on each
// accepted case of oracle/serverargs/cases.json (oracle/lower/extract.py), with the case's inputs.
type lowerAnswers struct {
	Cases []struct {
		Name       string            `json:"name"`
		Args       map[string]any    `json:"args"`
		Chip       config.Chip       `json:"chip"`
		ModelFacts config.ModelFacts `json:"model_facts"`
		Alignment  int               `json:"alignment"`
		Gather     struct {
			MLPTP   bool `json:"mlp_tp"`
			AttnTP  bool `json:"attn_tp"`
			MLPSync bool `json:"mlp_sync"`
		} `json:"gather"`
		MoE struct {
			A2A    string `json:"a2a_backend"`
			Runner string `json:"runner_backend"`
		} `json:"moe"`
		Attention struct {
			Prefill string `json:"prefill"`
			Decode  string `json:"decode"`
		} `json:"attention"`
		// CaptureRefused is SGLang's own refusal to lower the case: it asserts there is a decode
		// batch size to capture.
		CaptureRefused *struct {
			PoolSize int    `json:"pool_size"`
			ReqWidth int    `json:"req_width"`
			Message  string `json:"message"`
		} `json:"capture_refused"`
		Capture []struct {
			PoolSize  int   `json:"pool_size"`
			ReqWidth  int   `json:"req_width"`
			CaptureBs []int `json:"capture_bs"`
			Padded    []int `json:"padded"`
		} `json:"capture"`
	} `json:"cases"`
}

func readLower(t *testing.T) lowerAnswers {
	t.Helper()
	b, err := os.ReadFile("testdata/sglang_lower.json")
	if err != nil {
		t.Fatal(err)
	}
	var a lowerAnswers
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if len(a.Cases) == 0 {
		t.Fatal("testdata/sglang_lower.json holds no case")
	}
	return a
}

// TestLowerAgainstSGLang: for every accepted case, the decode step lowers as SGLang's own
// functions lower it: the gather rules, the batch-size alignment, the attention and MoE
// backends, and for every request-pool size and captured width, the capture batch sizes and the
// bucket each batch size pads to.
func TestLowerAgainstSGLang(t *testing.T) {
	for _, c := range readLower(t).Cases {
		t.Run(c.Name, func(t *testing.T) {
			args := map[string]any{"model_path": "<model>"}
			for k, v := range c.Args {
				args[k] = v
			}
			r, err := config.Resolve(config.Input{Args: args, Chip: c.Chip, Model: c.ModelFacts})
			if err != nil {
				t.Fatalf("config.Resolve: %v", err)
			}
			a := &r.Args
			mlp, err := RequireMLPTPGather(a)
			if err != nil {
				t.Fatal(err)
			}
			attn, _ := RequireAttnTPGather(a)
			sync, _ := RequireMLPSync(a)
			if mlp != c.Gather.MLPTP || attn != c.Gather.AttnTP || sync != c.Gather.MLPSync {
				t.Errorf("gather mlp_tp=%v attn_tp=%v mlp_sync=%v, SGLang's %+v", mlp, attn, sync, c.Gather)
			}
			if al, _ := Alignment(a); al != c.Alignment {
				t.Errorf("alignment %d, SGLang's %d", al, c.Alignment)
			}
			if p, d := AttentionBackends(a); p != c.Attention.Prefill || d != c.Attention.Decode {
				t.Errorf("attention (%s, %s), SGLang's (%s, %s)", p, d, c.Attention.Prefill, c.Attention.Decode)
			}
			moe, err := MoEOf(a)
			if err != nil || moe.A2ABackend != c.MoE.A2A || moe.RunnerBackend != c.MoE.Runner {
				t.Errorf("MoE %+v (%v), SGLang's %+v", moe, err, c.MoE)
			}
			if r := c.CaptureRefused; r != nil {
				// SGLang cannot lower this case (a decode CUDA graph of size 0): so must the Go lowering.
				if bs, err := CaptureBatchSizes(a, r.PoolSize, r.ReqWidth); err == nil {
					t.Errorf("SGLang refuses to capture (%s); the Go lowering captured %v", r.Message, bs)
				}
				return
			}
			for _, cp := range c.Capture {
				bs, err := CaptureBatchSizes(a, cp.PoolSize, cp.ReqWidth)
				if err != nil {
					t.Fatalf("pool %d width %d: %v", cp.PoolSize, cp.ReqWidth, err)
				}
				if !reflect.DeepEqual(bs, cp.CaptureBs) {
					t.Errorf("pool %d width %d: capture %v, SGLang's %v", cp.PoolSize, cp.ReqWidth, bs, cp.CaptureBs)
					continue
				}
				for i, want := range cp.Padded {
					if got, ok := PadToBucket(i+1, bs); !ok || got != want {
						t.Errorf("pool %d width %d: batch %d pads to %d (graph %v), SGLang's %d", cp.PoolSize, cp.ReqWidth, i+1, got, ok, want)
						break
					}
				}
				// One past the largest bucket runs eagerly at its own size.
				over := bs[len(bs)-1] + 1
				if got, ok := PadToBucket(over, bs); ok || got != over {
					t.Errorf("batch %d beyond the largest bucket: got %d (graph %v), want eager at %d", over, got, ok, over)
				}
			}
		})
	}
}

// TestLoweringRefusesWhatItCannotKnowByName: elastic EP, the MORI all-to-all backend and torch
// compile are refused by their SGLang names; the closest accepted settings lower (SIM-9).
func TestLoweringRefusesWhatItCannotKnowByName(t *testing.T) {
	c := readLower(t).Cases[0]
	r, err := config.Resolve(config.Input{Args: map[string]any{"model_path": "<model>"}, Chip: c.Chip, Model: c.ModelFacts})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Alignment(&r.Args); err != nil {
		t.Fatalf("the accepted partner does not lower: %v", err)
	}
	backend := "nixl"
	for name, set := range map[string]func(*config.ServerArgs){
		"elastic_ep_backend":   func(a *config.ServerArgs) { a.ElasticEpBackend = &backend },
		"moe_a2a_backend=mori": func(a *config.ServerArgs) { a.MoeA2aBackend = "mori" },
		"enable_torch_compile": func(a *config.ServerArgs) { a.EnableTorchCompile = true },
	} {
		a := r.Args
		set(&a)
		_, err := Alignment(&a)
		var nl *NotLowered
		if !errors.As(err, &nl) || nl.Name != name || nl.Issue != issueLower {
			t.Errorf("%s: got %v, want a refusal naming it and citing %s", name, err, issueLower)
		}
	}
}

package kernel

import (
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// Pad tokens ride beside the requests: they add to the tokens the layers run, not to the
// requests, so a padded step has no more requests than an unpadded one.
func TestPadTokens(t *testing.T) {
	b := Batch{Reqs: []ReqShape{{Scheduled: 6}, {Scheduled: 6}}, PadTokens: 20}
	if b.Tokens() != 12 || b.PaddedTokens() != 32 || len(b.Reqs) != 2 || !b.UniformDecode(6) {
		t.Errorf("tokens %d padded %d", b.Tokens(), b.PaddedTokens())
	}
	// A rank with nothing scheduled runs a dummy batch of pad alone.
	dummy := Batch{PadTokens: 64}
	if dummy.Tokens() != 0 || dummy.PaddedTokens() != 64 || dummy.UniformDecode(1) {
		t.Errorf("dummy batch: tokens %d padded %d", dummy.Tokens(), dummy.PaddedTokens())
	}
}

func TestBatchValidate(t *testing.T) {
	for _, c := range []struct {
		name string
		b    Batch
		bad  string
	}{
		{"empty", Batch{}, ""},
		{"padded", Batch{PadTokens: 64, Reqs: []ReqShape{{Scheduled: 1}}}, ""},
		{"negative pad", Batch{PadTokens: -1}, "PadTokens -1"},
		{"negative budget", Batch{SMBudget: -2}, "SMBudget -2"},
		{"negative threshold", Batch{DecodeThreshold: -1}, "DecodeThreshold -1"},
		{"negative request count", Batch{Reqs: []ReqShape{{Scheduled: 1}, {Computed: -3}}}, "Reqs[1].Computed -3"},
	} {
		t.Run(c.name, func(t *testing.T) {
			err := c.b.Validate()
			if (err == nil) != (c.bad == "") || (err != nil && !strings.Contains(err.Error(), c.bad)) {
				t.Errorf("Validate() = %v, want %q", err, c.bad)
			}
		})
	}
}

// A batch survives a marshal and unmarshal with its pad, and an unpadded one stays
// unpadded.
func TestBatchRoundTrips(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	for i := 0; i < 500; i++ {
		b := Batch{DecodeThreshold: r.Intn(8), SMBudget: r.Intn(144)}
		if i%2 == 0 {
			b.PadTokens = r.Intn(8192)
		}
		for n := r.Intn(4); n > 0; n-- {
			b.Reqs = append(b.Reqs, ReqShape{Scheduled: 1 + r.Intn(6), Computed: r.Intn(1 << 16)})
		}
		out, err := json.Marshal(b)
		if err != nil {
			t.Fatal(err)
		}
		var back Batch
		if err := json.Unmarshal(out, &back); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(back, b) || back.Validate() != nil {
			t.Fatalf("%+v round-tripped through %s to %+v", b, out, back)
		}
	}
}

// TestPadTokensMatchSource pins the engine lines PadTokens' documentation rests on, in a
// v0.31.0 checkout at VLLM_SOURCE. Without one it skips and says so.
func TestPadTokensMatchSource(t *testing.T) {
	root := os.Getenv("VLLM_SOURCE")
	skip := t.Skipf
	if os.Getenv("VLLM_SOURCE_REQUIRED") != "" {
		skip = t.Fatalf
	}
	if root == "" {
		skip("VLLM_SOURCE is unset: PadTokens' engine basis was NOT checked against vLLM v0.31.0")
	}
	if _, err := os.Stat(filepath.Join(root, "vllm", "v1", "worker")); err != nil {
		skip("VLLM_SOURCE=%s has no vllm/v1/worker: PadTokens' engine basis was NOT checked", root)
	}
	read := func(parts ...string) string {
		b, err := os.ReadFile(filepath.Join(append([]string{root}, parts...)...))
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	for file, wants := range map[string][]string{
		"dp_utils.py": {
			"def _post_process_dp_padding(tensor: torch.Tensor, should_dp_pad: bool) -> torch.Tensor:",
			"max_num_tokens = int(num_tokens_across_dp.max().item())",
		},
		"gpu_model_runner.py": {
			"num_tokens_unpadded = num_tokens_padded if is_profile else 0",
			"num_tokens=num_tokens_padded,\n                num_tokens_across_dp=num_tokens_across_dp,",
			"logits_indices = query_start_loc[1:] - 1",
		},
	} {
		src := read("vllm", "v1", "worker", file)
		for _, w := range wants {
			if !strings.Contains(src, w) {
				t.Errorf("%s no longer contains %q; re-derive PadTokens' documentation", file, w)
			}
		}
	}
}

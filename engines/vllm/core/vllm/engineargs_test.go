// Original to blis-vllm.

package vllm

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-vllm/internal/upstream"
)

// engineArgsFile is oracle/engineargs-cases.json with oracle/run_engineargs.py's answers:
// the pinned vLLM's own EngineArgs(...).create_engine_config(), run on each engine-block shape
// (tier 1, the dynamic half, blis-vllm#27).
type engineArgsCase struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
	// Expect is, for a case vLLM refuses, a substring of the message the factory refuses it with.
	Expect string `json:"expect"`
}

type engineArgsTrace struct {
	Provenance struct {
		Tier       int    `json:"tier"`
		VLLMCommit string `json:"vllm_commit"`
	} `json:"provenance"`
	Cases []struct {
		Name     string `json:"name"`
		Accepted bool   `json:"accepted"`
		Error    string `json:"error"`
		Resolved struct {
			MaxNumSeqs                int     `json:"max_num_seqs"`
			MaxNumBatchedTokens       int     `json:"max_num_batched_tokens"`
			MaxModelLen               int     `json:"max_model_len"`
			SchedulingPolicy          string  `json:"scheduling_policy"`
			EnableChunkedPrefill      bool    `json:"enable_chunked_prefill"`
			LongPrefillTokenThreshold int     `json:"long_prefill_token_threshold"`
			EnablePrefixCaching       bool    `json:"enable_prefix_caching"`
			BlockSize                 int     `json:"block_size"`
			Watermark                 float64 `json:"watermark"`
			MaxLoRAs                  int     `json:"max_loras"`
		} `json:"resolved"`
	} `json:"cases"`
}

// vllmDefaults are the values vLLM resolves when the engine block leaves a setting unstated
// and the factory cannot, because the answer belongs to the model and the GPU (the factory
// refuses such a block, and says why). They are vLLM's CPU-platform answers for the oracle's
// model, so they are compared only as "the factory refuses rather than guesses".
var unstatedRefused = map[string]string{
	"default-max-num-seqs-unstated":           "max_num_seqs and max_num_batched_tokens must be stated",
	"default-max-num-batched-tokens-unstated": "max_num_seqs and max_num_batched_tokens must be stated",
	"default-all-scheduler-unstated":          "max_num_seqs and max_num_batched_tokens must be stated",
	"default-block-size-unstated":             "block_size must be stated",
	"default-max-model-len-unstated":          "max_model_len must be stated",
}

// notModelledRefused are the shapes vLLM accepts and the factory refuses by design as not
// modelled, each with its issue: they are compared as "the factory refuses, and says why".
var notModelledRefused = map[string]string{
	// v0.33.0 runs async scheduling with an ngram drafter on the V2 runner
	// (vllm/config/vllm.py:1651-1719@4ba787ec), which is not diff-tested; the V1 path is
	// TestAsyncSchedulingWithNgramFollowsTheModelRunner's.
	"not-modelled-async-with-ngram-stated": "issues/186",
	// The same, with async_scheduling unstated: on V2 it resolves to on (:1685-1719).
	"accepted-ngram":                    "issues/186",
	"default-async-unstated-with-ngram": "issues/186",
}

// engineBlockOf maps an EngineArgs keyword to the engine block's own key and value.
func engineBlockOf(args map[string]any) string {
	var b strings.Builder
	// The V2 runner is v0.33.0's default wherever Triton is installed, as on every CUDA image and
	// on the oracle (vllm/config/vllm.py:731-767@4ba787ec), so the block states it.
	b.WriteString("pipeline_parallel_size: 1\nnum_gpu_blocks: 400\nuse_v2_model_runner: true\n")
	for k, v := range args {
		switch k {
		case "speculative_config":
			sc := v.(map[string]any)
			b.WriteString("speculative_config: {method: " + sc["method"].(string) + ", num_speculative_tokens: " + itoaAny(sc["num_speculative_tokens"]) + "}\n")
		case "enable_lora":
			// the engine block has max_loras alone
		default:
			out, _ := yaml.Marshal(map[string]any{k: v})
			b.Write(out)
		}
	}
	return b.String()
}

func itoaAny(v any) string {
	out, _ := yaml.Marshal(v)
	return strings.TrimSpace(string(out))
}

// TestEngineArgsAgainstVLLM compares the factory with vLLM's own configuration classes on each
// engine-block shape in oracle/engineargs-cases.json: it accepts what vLLM accepts, refuses what
// vLLM refuses (with a message that names the rule, the case's expect), and where both resolve a
// setting the values are equal. The five shapes that leave a setting unstated, which vLLM
// resolves from the model and the GPU, are the exception: the factory makes the caller state
// them, so they are checked as refusals, not as values, and the two disagree on acceptance there
// by design.
func TestEngineArgsAgainstVLLM(t *testing.T) {
	var cases struct {
		Cases []engineArgsCase `json:"cases"`
	}
	for path, v := range map[string]any{"../../oracle/engineargs-cases.json": &cases, "testdata/engineargs.json": nil} {
		if v == nil {
			continue
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(raw, v); err != nil {
			t.Fatal(err)
		}
	}
	var tr engineArgsTrace
	raw, err := os.ReadFile("testdata/engineargs.json")
	if err != nil {
		t.Fatalf("no trace (%v): generate it with oracle/run_engineargs.py", err)
	}
	if err := json.Unmarshal(raw, &tr); err != nil {
		t.Fatal(err)
	}
	if tr.Provenance.Tier != 1 || tr.Provenance.VLLMCommit != upstream.Commit {
		t.Fatalf("the trace is tier %d at %q, not tier 1 at the pinned release", tr.Provenance.Tier, tr.Provenance.VLLMCommit)
	}
	if len(tr.Cases) != len(cases.Cases) {
		t.Fatalf("%d traced cases for %d inputs: regenerate the trace", len(tr.Cases), len(cases.Cases))
	}
	accepted, refused := 0, 0
	for i, c := range cases.Cases {
		want := tr.Cases[i]
		if want.Name != c.Name {
			t.Fatalf("case %d is %q in the trace and %q in the inputs", i, want.Name, c.Name)
		}
		t.Run(c.Name, func(t *testing.T) {
			var node yaml.Node
			if err := yaml.Unmarshal([]byte(engineBlockOf(c.Args)), &node); err != nil {
				t.Fatal(err)
			}
			cfg, err := decodeConfig(*node.Content[0])
			if err != nil {
				t.Fatal(err)
			}
			got, err := resolve(cfg)
			if reason, ok := notModelledRefused[c.Name]; ok {
				if !want.Accepted {
					t.Fatalf("vLLM refused %s, which the factory refuses as not modelled: drop it from notModelledRefused", c.Name)
				}
				if err == nil || !strings.Contains(err.Error(), reason) {
					t.Errorf("the factory should refuse %s as not modelled (%q); got %v", c.Name, reason, err)
				}
				return
			}
			if reason, ok := unstatedRefused[c.Name]; ok {
				if !want.Accepted {
					t.Fatalf("vLLM refused %s, which the factory expects it to resolve", c.Name)
				}
				if err == nil || !strings.Contains(err.Error(), reason) {
					t.Errorf("the factory should refuse an unstated setting that vLLM resolves from the model and GPU (%q); got %v", reason, err)
				}
				return
			}
			if want.Accepted != (err == nil) {
				t.Fatalf("vLLM accepted=%v (%s); the factory: %v", want.Accepted, want.Error, err)
			}
			if err != nil {
				// Agreeing that a block is refused is not agreeing why: the factory's own message
				// must name the rule the case was written for.
				if c.Expect == "" {
					t.Errorf("a refused case needs an expect, a substring of the factory's message; the factory said: %v", err)
				} else if !strings.Contains(err.Error(), c.Expect) {
					t.Errorf("the factory refused with %q, which does not contain %q", err.Error(), c.Expect)
				}
				refused++
				return
			}
			if c.Expect != "" {
				t.Errorf("the case expects the refusal %q, and the factory accepted it", c.Expect)
			}
			accepted++
			r, sc := want.Resolved, got.sched
			checks := []struct {
				what     string
				got, vll any
			}{
				{"max_num_seqs", sc.MaxNumSeqs, r.MaxNumSeqs},
				{"max_num_batched_tokens", sc.MaxNumBatchedTokens, r.MaxNumBatchedTokens},
				{"max_model_len", sc.MaxModelLen, r.MaxModelLen},
				{"scheduling_policy", string(sc.Policy), r.SchedulingPolicy},
				{"enable_chunked_prefill", sc.EnableChunkedPrefill, r.EnableChunkedPrefill},
				{"long_prefill_token_threshold", sc.LongPrefillTokenThreshold, r.LongPrefillTokenThreshold},
				{"enable_prefix_caching", sc.EnablePrefixCaching, r.EnablePrefixCaching},
				{"block_size", sc.BlockSize, r.BlockSize},
				{"watermark", sc.Watermark, r.Watermark},
				{"max_loras", sc.MaxLoRAs, r.MaxLoRAs},
			}
			for _, k := range checks {
				if k.got != k.vll {
					t.Errorf("%s: the factory resolves %v, vLLM %v", k.what, k.got, k.vll)
				}
			}
		})
	}
	// 39 shapes: 15 accepted by both, 16 refused by both, 5 that leave a setting unstated which
	// vLLM resolves from the model and GPU and the factory refuses by design (unstatedRefused), and
	// 3 that vLLM accepts and the factory refuses as not modelled (notModelledRefused). So the two
	// agree on acceptance in 31 of 39, not in all of them.
	if accepted != 15 || refused != 16 || len(unstatedRefused) != 5 || len(notModelledRefused) != 3 {
		t.Errorf("%d accepted, %d refused, %d unstated-by-design and %d not-modelled cases compared; want 15, 16, 5 and 3: the file changed",
			accepted, refused, len(unstatedRefused), len(notModelledRefused))
	}
}

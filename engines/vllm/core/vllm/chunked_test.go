package vllm

import (
	"reflect"
	"strings"
	"testing"
)

// chunkedBase is an engine block whose token budget equals max_model_len, which vLLM requires
// of a block with chunked prefill off (vllm/config/scheduler.py:314-327@5df63a72), and which
// matches the oracle scenario chunked-prefill-off: a 64-token budget, four sequences.
const chunkedBase = "max_num_batched_tokens: 64\nmax_model_len: 64\nnum_gpu_blocks: 120\nblock_size: 4\n" +
	"enable_prefix_caching: false\n"

// The stated enable_chunked_prefill reaches the scheduler's configuration, and unstated is
// vLLM's default, true (vllm/config/scheduler.py:134@5df63a72).
func TestEnableChunkedPrefillReachesTheScheduler(t *testing.T) {
	for _, c := range []struct {
		name, extra string
		want        bool
	}{
		{"unstated", "", true},
		{"true", "enable_chunked_prefill: true\n", true},
		{"false", "enable_chunked_prefill: false\n", false},
	} {
		s, err := resolve(mustConfig(t, merge(chunkedBase, c.extra)))
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if s.sched.EnableChunkedPrefill != c.want {
			t.Errorf("%s: EnableChunkedPrefill %t, want %t", c.name, s.sched.EnableChunkedPrefill, c.want)
		}
	}
}

// With it off, the engine's first step for the scenario chunked-prefill-off is vLLM's: c0's 40
// tokens, and c1, whose 40 do not fit the 24 left, waits (core/sched/testdata/golden/
// chunked-prefill-off.json, step 0 {c0: 40}); with it on c1 takes the 24 that fit.
func TestChunkedPrefillOffMakesTheFirstStepVLLMs(t *testing.T) {
	first := func(extra string) map[string]int {
		e, err := build(t, merge(chunkedBase, extra))()
		if err != nil {
			t.Fatal(err)
		}
		for _, id := range []string{"c0", "c1", "c2"} {
			n := 40
			if id == "c2" {
				n = 30
			}
			if err := e.Add(req(id, n, 3), 0); err != nil {
				t.Fatal(err)
			}
		}
		b, err := e.Schedule(0)
		if err != nil {
			t.Fatal(err)
		}
		got := map[string]int{}
		for _, s := range b.Reqs {
			got[s.ID] = s.Tokens
		}
		return got
	}
	if got, want := first("enable_chunked_prefill: false\n"), map[string]int{"c0": 40}; !reflect.DeepEqual(got, want) {
		t.Errorf("off: first step %v, want %v", got, want)
	}
	if got, want := first(""), map[string]int{"c0": 40, "c1": 24}; !reflect.DeepEqual(got, want) {
		t.Errorf("default: first step %v, want %v", got, want)
	}
}

// Each refusal has its smallest trigger and the closest configuration that builds (SIM-9).
func TestChunkedPrefillOffRefusals(t *testing.T) {
	off := merge(chunkedBase, "enable_chunked_prefill: false\n")
	for _, c := range []struct {
		name, extra, want string
		// ok is the same block with the trigger removed or on, which must build.
		ok string
	}{
		// vLLM's own refusals.
		{"budget below max_model_len", "max_num_batched_tokens: 63\n",
			"max_num_batched_tokens (63) is smaller than max_model_len (64)", "max_num_batched_tokens: 64\n"},
		{"a Mamba model with prefix caching", "mamba_model: true\nenable_prefix_caching: true\n",
			"Chunked prefill is required for mamba cache mode 'align'", "mamba_model: true\nenable_prefix_caching: false\n"},
		// What the engine does not model with it off.
		{"speculative decoding", "speculative: {method: eagle3, num_spec_tokens: 3}\n",
			"enable_chunked_prefill false with speculative decoding", ""},
		{"LoRA", "max_loras: 2\n", "enable_chunked_prefill false with max_loras", "max_loras: 0\n"},
		{"a KV transfer connector", producer, "enable_chunked_prefill false with a KV transfer connector", ""},
		{"a KV offload tier", cpuTier, "enable_chunked_prefill false with a KV offload tier", ""},
		{"data parallelism", "data_parallel_size: 2\n", "enable_chunked_prefill false with data parallelism", "data_parallel_size: 1\n"},
		{"the adaptive long-prefill cap", "long_prefill_token_threshold: 16\nlong_prefill_token_threshold_adaptive: true\n",
			"with long_prefill_token_threshold_adaptive", "long_prefill_token_threshold: 16\nlong_prefill_token_threshold_adaptive: false\n"},
		{"max_num_scheduled_tokens", "max_num_scheduled_tokens: 60\n", "with max_num_scheduled_tokens", "max_num_scheduled_tokens: 0\n"},
		{"a watermark", "watermark: 0.05\n", "with watermark", "watermark: 0\n"},
		{"max_num_active_seqs", "max_num_active_seqs: 2\n", "with max_num_active_seqs", "max_num_active_seqs: 0\n"},
		{"hybrid KV cache groups", "kv_layers: [{count: 2, spec: full, num_kv_heads: 1, head_size: 8}, " +
			"{count: 2, spec: sliding_window, sliding_window: 16, num_kv_heads: 1, head_size: 8}]\n",
			"with hybrid KV cache groups", "kv_layers: [{count: 4, spec: full, num_kv_heads: 1, head_size: 8}]\n"},
		{"pipeline parallelism", "pipeline_parallel_size: 2\nasync_scheduling: true\nuse_v2_model_runner: true\n",
			"with pipeline parallelism", "pipeline_parallel_size: 1\nasync_scheduling: true\nuse_v2_model_runner: true\n"},
	} {
		t.Run(c.name, func(t *testing.T) {
			err := buildErr(t, merge(off, c.extra))
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("err %v, want one naming %q", err, c.want)
			}
			if !strings.Contains(err.Error(), "enable_chunked_prefill") {
				t.Errorf("the refusal does not name enable_chunked_prefill: %v", err)
			}
			// The same trigger with chunked prefill on is not refused for chunked prefill.
			if err := buildErr(t, merge(chunkedBase, "enable_chunked_prefill: true\n", c.extra)); err != nil &&
				strings.Contains(err.Error(), "enable_chunked_prefill") {
				t.Errorf("with chunked prefill on: %v", err)
			}
			if c.ok != "" {
				if err := buildErr(t, merge(off, c.ok)); err != nil {
					t.Errorf("the closest configuration, %q, is refused: %v", c.ok, err)
				}
			}
		})
	}
}

// What the oracle scenarios run with it off builds: prefix caching, a long-prefill cap, priority
// scheduling, KV events, async scheduling at one pipeline stage and a uniform KV layer group.
func TestChunkedPrefillOffBuildsWhatTheOracleCovers(t *testing.T) {
	off := merge(chunkedBase, "enable_chunked_prefill: false\n")
	for name, extra := range map[string]string{
		"prefix caching":      "enable_prefix_caching: true\n",
		"long-prefill cap":    "long_prefill_token_threshold: 16\nscheduling_policy: priority\n",
		"KV events":           "enable_prefix_caching: true\nkv_events: {enable_kv_cache_events: true}\n",
		"async scheduling":    "async_scheduling: true\n",
		"async on the V2 run": "async_scheduling: true\nuse_v2_model_runner: true\n",
		"uniform KV layers": "kv_layers: [{count: 4, spec: full, num_kv_heads: 1, head_size: 8}, " +
			"{count: 2, spec: indexer, num_kv_heads: 1, head_size: 8}]\n",
	} {
		if err := buildErr(t, merge(off, extra)); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
}

// The batch defaults read the resolved value: with chunked prefill off an unstated token budget
// is raised to max_model_len (vllm/engine/arg_utils.py:3136-3141@5df63a72), which the engine
// block's stated enable_chunked_prefill decides.
func TestBatchDefaultsReadTheResolvedChunkedPrefill(t *testing.T) {
	const g = int64(1) << 30
	for _, c := range []struct {
		name   string
		stated *bool
		tokens int
	}{
		{"unstated is on", nil, 8192},
		{"on", ptrTrue(), 8192},
		{"off", ptrFalse(), 32768},
	} {
		r := ResolveBatch(BatchInput{DeviceMemoryBytes: 141 * g, DeviceName: "h200", MaxModelLen: 32768,
			ChunkedPrefill: ResolveChunkedPrefill(c.stated)})
		if r.MaxNumBatchedTokens != c.tokens {
			t.Errorf("%s: max_num_batched_tokens %d, want %d", c.name, r.MaxNumBatchedTokens, c.tokens)
		}
	}
}

func ptrTrue() *bool  { b := true; return &b }
func ptrFalse() *bool { b := false; return &b }

// buildErr is the error of building the engine block extra describes.
func buildErr(t *testing.T, extra string) error {
	t.Helper()
	_, err := Factory{Pricer: fixed}.New(spec(t, extra))
	return err
}

// merge joins engine-block fragments of one top-level key a line, a later fragment's key
// replacing an earlier one's.
func merge(parts ...string) string {
	var order []string
	lines := map[string]string{}
	for _, p := range parts {
		for _, l := range strings.Split(strings.TrimSpace(p), "\n") {
			if l == "" {
				continue
			}
			k, _, _ := strings.Cut(l, ":")
			if _, ok := lines[k]; !ok {
				order = append(order, k)
			}
			lines[k] = l
		}
	}
	var b strings.Builder
	for _, k := range order {
		b.WriteString(lines[k] + "\n")
	}
	return b.String()
}

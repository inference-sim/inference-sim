package vllmengine

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/fields"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/blis-vllm/compat/vllmsettings"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// VLM-3: a field marked not modelled is refused when it is set away from its default, and the
// refusal names the field and cites the open issue fields.json cites. This walks every
// not-modelled configuration entry of fields.json and sets it in a v0.32.0 deployment:
//
//   - an entry that is a key of an engine block (engineKeys) must be refused by engine.New,
//     naming the field and citing its issue (refused), unless the latency kernel reads and
//     prices the key (pricedByTheKernel), which is the explicit allow-list below;
//   - any other entry is not a key of the engine block, so a deployment cannot state it, and the
//     strict decoder refuses the name (TestEveryEngineKeyIsAccountedFor shows no key was missed).
//
// The environment variables of fields.json are not walked: the engine block's env refuses every
// key but VLLM_USE_V2_MODEL_RUNNER (core/vllm unreadEnv), including the Rust front end's and the
// launcher's, which no simulated step reads.

// engineKeys maps each key of an engine block, by path, to the fields.json names it states. A key
// the engine block has and fields.json does not name (allreduce_backend, vLLM's resolved override)
// maps to none.
var engineKeys = map[string][]string{
	"all2all_backend":                          {"all2all_backend"},
	"disable_custom_all_reduce":                {"disable_custom_all_reduce"},
	"allreduce_backend":                        nil,
	"cudagraph_mode":                           {"compilation_config.cudagraph_mode"},
	"max_cudagraph_capture_size":               {"max_cudagraph_capture_size", "compilation_config.max_cudagraph_capture_size"},
	"cudagraph_capture_sizes":                  {"cudagraph_capture_sizes", "compilation_config.cudagraph_capture_sizes"},
	"decoder_replay_cudagraph_capture_sizes":   {"compilation_config.decoder_replay_cudagraph_capture_sizes"},
	"custom_ops":                               {"compilation_config.custom_ops"},
	"use_trtllm_attention":                     {"attention_config.use_trtllm_attention"},
	"mla_prefill_backend":                      {"attention_config.mla_prefill_backend"},
	"indexer_kv_dtype":                         {"attention_config.indexer_kv_dtype"},
	"linear_backend":                           {"kernel_config.linear_backend", "linear_backend"},
	"env":                                      nil,
	"async_scheduling":                         {"async_scheduling"},
	"frontend":                                 nil,
	"disable_cascade_attn":                     {"disable_cascade_attn"},
	"enable_prefix_caching":                    {"enable_prefix_caching"},
	"swa_bounded_replay":                       {"swa_bounded_replay"},
	"block_size":                               {"block_size"},
	"max_num_batched_tokens":                   {"max_num_batched_tokens"},
	"max_num_seqs":                             {"max_num_seqs"},
	"long_prefill_token_threshold":             {"long_prefill_token_threshold"},
	"long_prefill_token_threshold_adaptive":    {"long_prefill_token_threshold_adaptive"},
	"max_num_scheduled_tokens":                 {"max_num_scheduled_tokens"},
	"max_num_active_seqs":                      {"max_num_active_seqs"},
	"watermark":                                {"watermark"},
	"dp_sync_interval":                         {"dp_sync_interval"},
	"prefill_schedule_interval":                {"prefill_schedule_interval"},
	"stream_interval":                          {"stream_interval"},
	"disable_hybrid_kv_cache_manager":          {"disable_hybrid_kv_cache_manager"},
	"max_model_len":                            {"max_model_len"},
	"kv_cache_memory_bytes":                    {"kv_cache_memory_bytes"},
	"prefix_match_unit":                        {"prefix_match_unit"},
	"prefix_cache_retention_interval":          {"prefix_cache_retention_interval"},
	"kv_events.enable_kv_cache_events":         {"kv_events_config.enable_kv_cache_events"},
	"kv_events.publisher":                      {"kv_events_config.publisher"},
	"kv_events.topic":                          {"kv_events_config.topic"},
	"max_loras":                                {"max_loras"},
	"max_cpu_loras":                            {"max_cpu_loras"},
	"quantization":                             {"quantization"},
	"cache_dtype":                              {"kv_cache_dtype"},
	"mamba_cache_dtype":                        {"mamba_cache_dtype"},
	"mamba_ssm_cache_dtype":                    {"mamba_ssm_cache_dtype"},
	"moe_backend":                              {"moe_backend", "kernel_config.moe_backend"},
	"mamba_cache_mode":                         {"mamba_cache_mode"},
	"gpu_memory_utilization":                   {"gpu_memory_utilization"},
	"scheduling_policy":                        {"scheduling_policy"},
	"dbo.enable_dbo":                           {"enable_dbo"},
	"dbo.dbo_decode_token_threshold":           {"dbo_decode_token_threshold"},
	"dbo.dbo_prefill_token_threshold":          {"dbo_prefill_token_threshold"},
	"eplb.enable_eplb":                         {"enable_eplb"},
	"eplb.num_redundant_experts":               {"eplb_config.num_redundant_experts"},
	"eplb.window_size":                         {"eplb_config.window_size"},
	"eplb.step_interval":                       {"eplb_config.step_interval"},
	"speculative.method":                       {"speculative_config.method"},
	"speculative.num_spec_tokens":              {"speculative_config.num_speculative_tokens"},
	"speculative.draft_model":                  {"speculative_config.model"},
	"speculative.kv_cache_dtype":               {"speculative_config.kv_cache_dtype"},
	"speculative.draft_sample_method":          {"speculative_config.draft_sample_method"},
	"speculative.rejection_sample_method":      {"speculative_config.rejection_sample_method"},
	"speculative.enable_adaptive_verification": {"speculative_config.enable_adaptive_verification"},
	"engram_config.cpu_offload":                {"engram_config.cpu_offload"},
	"engram_config.dp_shared_memory":           {"engram_config.dp_shared_memory"},
	"engram_config.use_thp":                    {"engram_config.use_thp"},
	"dcp_comm_backend":                         {"dcp_comm_backend"},
	"dcp_q_replicate":                          {"dcp_q_replicate"},
	"cp_kv_cache_interleave_size":              {"cp_kv_cache_interleave_size"},
}

// refused sets each refused field in the engine block, as YAML under pools[0].engine.
var refused = map[string]string{
	"kv_cache_memory_bytes":                                     "kv_cache_memory_bytes: 4294967296\n",
	"attention_config.use_trtllm_attention":                     "use_trtllm_attention: true\n",
	"attention_config.mla_prefill_backend":                      "mla_prefill_backend: FLASHINFER\n",
	"kernel_config.linear_backend":                              "linear_backend: flashinfer\n",
	"linear_backend":                                            "linear_backend: flashinfer\n",
	"compilation_config.custom_ops":                             "custom_ops: ['+rms_norm']\n",
	"compilation_config.decoder_replay_cudagraph_capture_sizes": "decoder_replay_cudagraph_capture_sizes: [1, 2]\n",
	"attention_config.indexer_kv_dtype":                         "indexer_kv_dtype: fp8\n",
	"speculative_config.kv_cache_dtype":                         "speculative: {method: eagle3, num_spec_tokens: 3, kv_cache_dtype: fp8}\n",
	"speculative_config.rejection_sample_method":                "speculative: {method: eagle3, num_spec_tokens: 3, rejection_sample_method: synthetic}\n",
	"kv_events_config.publisher":                                "kv_events: {enable_kv_cache_events: true, publisher: \"null\"}\n",
	"kv_events_config.topic":                                    "kv_events: {enable_kv_cache_events: true, topic: kv}\n",
	"mamba_ssm_cache_dtype":                                     "mamba_ssm_cache_dtype: float32\n",
	"dbo_decode_token_threshold":                                "dbo: {enable_dbo: true, dbo_decode_token_threshold: 64}\n",
	"dbo_prefill_token_threshold":                               "dbo: {enable_dbo: true, dbo_prefill_token_threshold: 1024}\n",
	"eplb_config.window_size":                                   "eplb: {enable_eplb: true, window_size: 500}\n",
	"eplb_config.step_interval":                                 "eplb: {enable_eplb: true, step_interval: 100}\n",
	"prefix_match_unit":                                         "prefix_match_unit: 8\n",
	"prefix_cache_retention_interval":                           "prefix_cache_retention_interval: 256\n",
	"max_cpu_loras":                                             "max_loras: 2\nmax_cpu_loras: 4\n",
	"mamba_cache_mode":                                          "mamba_cache_mode: align\n",
	"swa_bounded_replay":                                        "swa_bounded_replay: false\n",
}

// atDefault is the partner of each refused case (SIM-9's refusal pair): the same setting stated at
// the value vLLM runs when it is not stated, which must build, so a user who hands BLIS the flags
// they gave vLLM is not refused for a flag that changes nothing. The defaults are those of
// unreadSettings (core/vllm/config.go), each cited there. A key whose default is "unstated" has no
// such value, and its entry says so.
var atDefault = map[string]string{
	"attention_config.indexer_kv_dtype":                         "indexer_kv_dtype: auto\n",
	"kernel_config.linear_backend":                              "linear_backend: auto\n",
	"linear_backend":                                            "linear_backend: auto\n",
	"compilation_config.custom_ops":                             "custom_ops: []\n",
	"compilation_config.decoder_replay_cudagraph_capture_sizes": "decoder_replay_cudagraph_capture_sizes: []\n",
	"speculative_config.rejection_sample_method":                "speculative: {method: eagle3, num_spec_tokens: 3, rejection_sample_method: block}\n",
	"kv_events_config.publisher":                                "kv_events: {enable_kv_cache_events: true, publisher: zmq}\n",
	"kv_events_config.topic":                                    "kv_events: {enable_kv_cache_events: true, topic: \"\"}\n",
	"mamba_ssm_cache_dtype":                                     "mamba_ssm_cache_dtype: auto\n",
	"dbo_decode_token_threshold":                                "dbo: {enable_dbo: true, dbo_decode_token_threshold: 32}\n",
	"dbo_prefill_token_threshold":                               "dbo: {enable_dbo: true, dbo_prefill_token_threshold: 512}\n",
	"eplb_config.window_size":                                   "eplb: {enable_eplb: true, window_size: 1000}\n",
	"eplb_config.step_interval":                                 "eplb: {enable_eplb: true, step_interval: 3000}\n",
	"prefix_cache_retention_interval":                           "prefix_cache_retention_interval: 0\n",
	"max_cpu_loras":                                             "max_loras: 2\nmax_cpu_loras: 2\n",
	"mamba_cache_mode":                                          "mamba_cache_mode: none\n",
	"swa_bounded_replay":                                        "swa_bounded_replay: true\n",
}

// noDefaultValue lists the refused keys that vLLM leaves unstated (None), so no stated value is the
// default and every stated one is a choice. Each cites the line that makes it None.
var noDefaultValue = map[string]string{
	"kv_cache_memory_bytes":                 "vllm/config/cache.py:257@5df63a72",
	"attention_config.use_trtllm_attention": "vllm/config/attention.py:74@5df63a72",
	"attention_config.mla_prefill_backend":  "vllm/config/attention.py:81@5df63a72",
	"speculative_config.kv_cache_dtype":     "vllm/config/speculative.py:428@5df63a72",
	"prefix_match_unit":                     "vllm/config/cache.py:90@5df63a72",
}

// pricedByTheKernel is the allow-list: keys the latency kernel reads and prices from blis-substrate's
// enginesettings.Settings, so the engine must accept them. Their status is not_modelled because the
// kernel prices them with vLLM v0.31.0's policy, not v0.32.0's; the issue is the open one fields.json
// cites for each. VLM-3 and design 11.2 say a field the engine reads and prices when stated is a
// status to correct, to modelled with evidence or to refused when stated; neither is possible while
// the policy is v0.31.0's, and refusing would refuse every committed deployment that states them.
// The test checks each really reaches the kernel's settings (vllmsettings.Of), so an entry cannot
// outlive the kernel reading the key.
var pricedByTheKernel = map[string]string{
	"all2all_backend":                               "all2all_backend: allgather_reducescatter\n#75",
	"disable_custom_all_reduce":                     "disable_custom_all_reduce: true\n#78",
	"compilation_config.cudagraph_mode":             "cudagraph_mode: FULL\n#78",
	"max_cudagraph_capture_size":                    "max_cudagraph_capture_size: 512\n#78",
	"compilation_config.max_cudagraph_capture_size": "max_cudagraph_capture_size: 512\n#78",
	"cudagraph_capture_sizes":                       "cudagraph_capture_sizes: [1, 2, 4]\n#78",
	"compilation_config.cudagraph_capture_sizes":    "cudagraph_capture_sizes: [1, 2, 4]\n#78",
	"disable_cascade_attn":                          "disable_cascade_attn: false\n#78",
	"quantization":                                  "quantization: fp8\n#78",
	"kv_cache_dtype":                                "cache_dtype: fp8\n#78",
	"mamba_cache_dtype":                             "mamba_cache_dtype: float32\n#83",
	"moe_backend":                                   "moe_backend: triton\n#77",
	"kernel_config.moe_backend":                     "moe_backend: triton\n#77",
	"gpu_memory_utilization":                        "gpu_memory_utilization: 0.8\n#73",
	"enable_dbo":                                    "dbo: {enable_dbo: true}\n#74",
	"enable_eplb":                                   "eplb: {enable_eplb: true}\n#76",
	"eplb_config.num_redundant_experts":             "eplb: {enable_eplb: true, num_redundant_experts: 8}\n#76",
	"engram_config.cpu_offload":                     "engram_config: {cpu_offload: false}\n#78",
	"engram_config.dp_shared_memory":                "engram_config: {dp_shared_memory: true}\n#78",
	"engram_config.use_thp":                         "engram_config: {use_thp: true}\n#78",
	"dcp_comm_backend":                              "dcp_comm_backend: a2a\n#79",
	"dcp_q_replicate":                               "dcp_q_replicate: true\n#8",
	"cp_kv_cache_interleave_size":                   "cp_kv_cache_interleave_size: 4\n#8",
	// Reachable only through Spec.Config: a deployment cannot state them, because
	// deployment.PDTransfer has the connector alone and offload.connector_options refuses all but
	// blocks_per_chunk (offloadOf). blis-vllm#89.
	"kv_transfer_config.kv_load_failure_policy":    "#89",
	"kv_transfer_config.kv_connector_extra_config": "#89",
}

func fieldList(t *testing.T) []fields.Field {
	t.Helper()
	l, err := fields.Load(filepath.Join("..", "fields.json"))
	if err != nil {
		t.Fatal(err)
	}
	return l.Fields
}

// issueURL is the URL a refusal cites for fields.json's "<owner>/<repo>#<n>".
func issueURL(ref string) string {
	repo, n, _ := strings.Cut(ref, "#")
	return "https://github.com/" + repo + "/issues/" + n
}

// withEngine returns the d079 scenario with extra lines added to its pool's engine block.
func withEngine(t *testing.T, extra string) map[string]string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "d079-032.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	body := strings.Replace(string(raw), "      gpu_memory_utilization: 0.92\n", "      gpu_memory_utilization: 0.92\n"+indent(extra, "      "), 1)
	dir := t.TempDir()
	path := filepath.Join(dir, "v.yaml")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return map[string]string{RootScenario: path}
}

func indent(s, pad string) string {
	var b strings.Builder
	for _, l := range strings.SplitAfter(s, "\n") {
		if l != "" {
			b.WriteString(pad + l)
		}
	}
	return b.String()
}

func TestEveryNotModelledFieldIsRefusedOrExplained(t *testing.T) {
	Register()
	// The base fixture builds, so a refusal below is the stated field's.
	if _, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: withEngine(t, "")}); err != nil {
		t.Fatalf("the base scenario does not build: %v", err)
	}
	keyOf := map[string]string{}
	for k, names := range engineKeys {
		for _, n := range names {
			keyOf[n] = k
		}
	}
	nRefused, nPriced, nNoKey := 0, 0, 0
	for _, f := range fieldList(t) {
		if f.Source != fields.SourceConfig || f.Status != fields.StatusNotModelled {
			continue
		}
		name := f.Name
		url := issueURL(f.Evidence.Issue)
		switch snippet, isRefused := refused[name]; {
		case isRefused:
			nRefused++
			// Each build opens its own Data and a scenario of its own, so the refusals run in parallel.
			t.Run(name, func(t *testing.T) {
				t.Parallel()
				_, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: withEngine(t, snippet)})
				if err == nil {
					t.Errorf("%s: stated as %q, away from vLLM's default, and not refused", name, strings.TrimSpace(snippet))
					return
				}
				// The pair: the same key at vLLM's default builds.
				if def, ok := atDefault[name]; ok {
					if _, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: withEngine(t, def)}); err != nil {
						t.Errorf("%s: stated at vLLM's default as %q, and refused: %v", name, strings.TrimSpace(def), err)
					}
				} else if _, ok := noDefaultValue[name]; !ok {
					t.Errorf("%s: refused, with neither an at-default partner nor a stated reason it has no default value", name)
				}
				key := keyOf[name]
				leaf := key[strings.LastIndex(key, ".")+1:]
				if !strings.Contains(err.Error(), name) && !strings.Contains(err.Error(), leaf) {
					t.Errorf("%s: the refusal does not name the field: %v", name, err)
				}
				if !strings.Contains(err.Error(), url) {
					t.Errorf("%s: the refusal does not cite %s: %v", name, url, err)
				}
			})
		case pricedByTheKernel[name] != "":
			nPriced++
			entry := pricedByTheKernel[name]
			issue := entry[strings.LastIndex(entry, "#"):]
			if !strings.HasSuffix(f.Evidence.Issue, issue) {
				t.Errorf("%s: the allow-list cites %s, fields.json cites %s", name, issue, f.Evidence.Issue)
			}
			if strings.HasPrefix(entry, "#") {
				continue // reachable only through Spec.Config: nothing to set in a deployment
			}
			// The latency kernel reads the key: stating it changes the settings it prices from.
			var e engineconfig.Engine
			if err := yaml.Unmarshal([]byte(strings.TrimSuffix(entry[:strings.LastIndex(entry, "#")], "\n")), &e); err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			if reflect.DeepEqual(vllmsettings.Of(e), vllmsettings.Of(engineconfig.Engine{})) {
				t.Errorf("%s: the allow-list says the kernel reads %q, but vllmsettings.Of ignores it", name, entry)
			}
		default:
			nNoKey++
			if _, ok := keyOf[name]; ok {
				t.Errorf("%s is a key of the engine block (%s) and is neither refused nor on the allow-list", name, keyOf[name])
				continue
			}
			// Not a key of an engine block: the strict decoder refuses the name.
			if _, err := engineconfig.Decode(blockOf(t, name)); err == nil || !strings.Contains(err.Error(), name) {
				t.Errorf("%s: the engine block's decoder does not refuse it by name: %v", name, err)
			}
		}
	}
	t.Logf("not-modelled config fields: refused %d, priced by the kernel (allow-list) %d, not a key of the engine block %d", nRefused, nPriced, nNoKey)
	// Exact counts, so a field that moves between the classes is seen.
	if nRefused != len(refused) {
		t.Errorf("%d refused cases are for fields that are not not_modelled entries", len(refused)-nRefused)
	}
	if nPriced != len(pricedByTheKernel) {
		t.Errorf("%d allow-list entries are not not_modelled entries", len(pricedByTheKernel)-nPriced)
	}
}

// Every key of the engine block is accounted for in engineKeys, so a key added to
// engineconfig.Engine cannot dodge the walk above, and each name engineKeys gives is in fields.json.
func TestEveryEngineKeyIsAccountedFor(t *testing.T) {
	var paths []string
	var walk func(ty reflect.Type, prefix string)
	walk = func(ty reflect.Type, prefix string) {
		for i := 0; i < ty.NumField(); i++ {
			tag := strings.Split(ty.Field(i).Tag.Get("yaml"), ",")[0]
			if tag == "" || tag == "-" {
				continue
			}
			ft := ty.Field(i).Type
			for ft.Kind() == reflect.Pointer {
				ft = ft.Elem()
			}
			if ft.Kind() == reflect.Struct && ft.PkgPath() == ty.PkgPath() {
				walk(ft, prefix+tag+".")
				continue
			}
			paths = append(paths, prefix+tag)
		}
	}
	walk(reflect.TypeOf(engineconfig.Engine{}), "")
	sort.Strings(paths)
	have := map[string]bool{}
	for _, f := range fieldList(t) {
		have[f.Name+"/"+string(f.Source)] = true
	}
	for _, p := range paths {
		names, ok := engineKeys[p]
		if !ok {
			t.Errorf("engine block key %q is in no entry of engineKeys", p)
		}
		for _, n := range names {
			if !have[n+"/config"] {
				t.Errorf("engineKeys names %q for %q, which fields.json does not list", n, p)
			}
		}
	}
	for p := range engineKeys {
		i := sort.SearchStrings(paths, p)
		if i == len(paths) || paths[i] != p {
			t.Errorf("engineKeys lists %q, which is not a key of the engine block", p)
		}
	}
}

func blockOf(t *testing.T, key string) deployment.EngineBlock {
	t.Helper()
	var n yaml.Node
	if err := yaml.Unmarshal([]byte("{"+key+": 1}"), &n); err != nil {
		t.Fatal(err)
	}
	return deployment.NewEngineBlock(n.Content[0])
}

// A stated kv_cache_memory_bytes replaces gpu_memory_utilization in vLLM
// (vllm/config/cache.py:257-264@5df63a72). The port sizes the cache from the latter, so a
// deployment that states the former was sized from 0.92, about 23 times too large for 4 GiB on a
// 70B model. It is refused, citing blis-vllm#81, and the same deployment with the key removed builds.
func TestAStatedKVCacheMemoryBytesIsRefusedNotSizedFromTheDefault(t *testing.T) {
	Register()
	raw, err := os.ReadFile(filepath.Join("..", "kerneltests", "testdata", "aisimulate", "llama-3.1-70b-instruct-h200-fp8-vllm-tp2.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	body := strings.Replace(string(raw), "      gpu_memory_utilization: 0.92\n", "      kv_cache_memory_bytes: 4294967296\n", 1)
	if body == string(raw) {
		t.Fatal("the fixture states no gpu_memory_utilization: 0.92 to replace")
	}
	path := filepath.Join(t.TempDir(), "v.yaml")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: map[string]string{RootScenario: path}})
	if err == nil || !strings.Contains(err.Error(), "kv_cache_memory_bytes") || !strings.Contains(err.Error(), "blis-vllm/issues/81") {
		t.Fatalf("err %v, want a refusal naming kv_cache_memory_bytes and citing #81", err)
	}
}

// dp_sync_interval is a key of the engine block, so a deployment can state it, and the data-
// parallel group runs the stated interval: a wave runs on up to dp_sync_interval-1 dummy steps past
// its last request, so a longer interval does more steps for the same requests.
func TestADeploymentStatesDPSyncInterval(t *testing.T) {
	Register()
	// The group's wave runs dummy batches after its last request until the next sync, so the time
	// the group stays busy past the request's last token follows the stated interval.
	end := func(extra string) (engine.Time, int) {
		tr, err := enginetest.Drive(func() (engine.Engine, error) {
			return engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: withEngine(t, extra)})
		}, []enginetest.Arrival{{At: 0, Req: engine.Request{ID: "a", Prompt: make([]int32, 64), Output: make([]int32, 4)}}}, enginetest.Options{})
		if err != nil {
			t.Fatal(err)
		}
		return tr.End, len(tr.Steps)
	}
	e1, n1 := end("dp_sync_interval: 1\n")
	e64, n64 := end("dp_sync_interval: 64\n")
	t.Logf("interval 1: end %d, %d steps; interval 64: end %d, %d steps", e1, n1, e64, n64)
	if e1 >= e64 {
		t.Errorf("dp_sync_interval 1 ends at %d and 64 at %d: the stated interval does not reach the group", e1, e64)
	}
}

// rejection_sample_method is not_modelled (#29) only for the value that fixes the acceptance length.
// "standard" and "block", the values recorded runs state, are accepted: acceptance is supplied, so
// they decide nothing the engine takes from the setting. "synthetic" is refused.
func TestOnlySyntheticRejectionSamplingIsRefused(t *testing.T) {
	Register()
	for _, c := range []struct {
		method  string
		refused bool
	}{{"standard", false}, {"block", false}, {"synthetic", true}} {
		extra := "speculative: {method: eagle3, num_spec_tokens: 3, rejection_sample_method: " + c.method + "}\n"
		_, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: withEngine(t, extra)})
		if c.refused != (err != nil) {
			t.Errorf("rejection_sample_method %s: err %v, want refused %t", c.method, err, c.refused)
		}
	}
}

// The default of kv_events.publisher depends on enable_kv_cache_events: None resolves to "zmq" when
// events are enabled and "null" when not (vllm/config/kv_events.py:51-52@5df63a72). So "null" is the
// default with events off and a deviation with events on, and "zmq" the reverse.
func TestKVEventsPublisherDefaultFollowsEnableKVCacheEvents(t *testing.T) {
	Register()
	for _, c := range []struct {
		enabled   bool
		publisher string
		refused   bool
	}{
		{true, "zmq", false}, {true, "null", true}, {false, "null", false}, {false, "zmq", true},
	} {
		extra := fmt.Sprintf("kv_events: {enable_kv_cache_events: %t, publisher: %q}\n", c.enabled, c.publisher)
		_, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: withEngine(t, extra)})
		if c.refused != (err != nil) {
			t.Errorf("events %t, publisher %s: err %v, want refused %t", c.enabled, c.publisher, err, c.refused)
		}
		if err != nil && !strings.Contains(err.Error(), "issues/60") {
			t.Errorf("events %t, publisher %s: refusal does not cite #60: %v", c.enabled, c.publisher, err)
		}
	}
}

// A deployment's engine block is completed with mamba_model for a model that has recurrent layers,
// and only for one: the hook that rewrites mamba_cache_mode runs for those models alone.
func TestPrepareStatesMambaModelOnlyForARecurrentModel(t *testing.T) {
	has := func(scenario string) bool {
		t.Helper()
		return stated(openPool(t, scenario).complete(yaml.Node{}), "mamba_model") == "true"
	}
	if has("d079-032.yaml") {
		t.Error("d079 (minimax-m2.5) is not a Mamba model, and its block states mamba_model")
	}
	m, err := kernelmodel.Open("nemotron3-ultra-h100-agg.yaml", kernelmodel.Repos{
		Scenarios: filepath.Join("..", "kerneltests", "testdata"), Data: embeddedData(t)})
	if err != nil {
		t.Fatal(err)
	}
	if !m.HasRecurrentLayers() {
		t.Error("nemotron3-ultra has recurrent layers and HasRecurrentLayers says it has none")
	}
}

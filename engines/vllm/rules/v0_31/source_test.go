package v0_31

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/rules"
)

// This file re-derives the pack's constants from an engine checkout, when one is
// available at VLLM_SOURCE. It exists because every constant here was transcribed by
// hand, and hand transcription is where the errors were: an earlier draft carried
// four of seventeen cache dtypes, four of five graph modes, a backend enum for a
// setting that is a boolean, and a field name the engine does not use. Three of those
// would have rejected valid deployments.
//
// constants_test.go pins the values so a refactor cannot quietly narrow them. This
// file checks the values are right in the first place, which no amount of internal
// pinning can establish.
//
// Without a checkout the tests SKIP and say so. A skipped test reads like a passing
// one in a summary line, so the message names what was not checked.

func vllmSource(t *testing.T) string {
	t.Helper()
	root := os.Getenv("VLLM_SOURCE")
	// CI sets VLLM_SOURCE_REQUIRED, so a checkout that went missing fails the job rather
	// than skipping every check and leaving it green.
	skip := t.Skipf
	if os.Getenv("VLLM_SOURCE_REQUIRED") != "" {
		skip = t.Fatalf
	}
	if root == "" {
		skip("VLLM_SOURCE is unset: the pack's constants were NOT checked against " +
			"an engine checkout. Set it to a vllm tree at version " + Version +
			" to verify them.")
	}
	if _, err := os.Stat(filepath.Join(root, "vllm", "config")); err != nil {
		skip("VLLM_SOURCE=%s has no vllm/config: the pack's constants were NOT checked", root)
	}
	return root
}

// readSource returns one engine file's contents, failing rather than skipping: the
// tree exists, so a missing file means the layout moved and the transcription's basis
// is gone.
func readSource(t *testing.T, root string, parts ...string) string {
	t.Helper()
	path := filepath.Join(append([]string{root}, parts...)...)
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v (the engine layout may have moved; the pack's "+
			"constants can no longer be verified against it)", path, err)
	}
	return string(b)
}

// literalMembers extracts the string members of a Python Literal type alias. A member
// that is not a quoted string -- another alias nested in it, which Python flattens into
// the accepted set -- fails the test rather than being dropped, since a dropped member
// would leave a value the engine accepts missing from both sides of the comparison.
func literalMembers(t *testing.T, src, name string) []string {
	t.Helper()
	re := regexp.MustCompile(`(?s)\b` + name + `\s*=\s*Literal\[(.*?)\]`)
	m := re.FindStringSubmatch(src)
	if m == nil {
		t.Fatalf("no Literal named %s found; the engine may have changed how it "+
			"declares this vocabulary", name)
	}
	var vals []string
	for _, line := range strings.Split(m[1], "\n") {
		if i := strings.Index(line, "#"); i >= 0 {
			line = line[:i]
		}
		for _, part := range strings.Split(line, ",") {
			part = strings.TrimSpace(part)
			switch {
			case part == "":
			case len(part) >= 2 && part[0] == '"' && part[len(part)-1] == '"':
				vals = append(vals, part[1:len(part)-1])
			default:
				t.Fatalf("Literal %s has a member that is not a quoted string, %q; resolve it "+
					"before comparing", name, part)
			}
		}
	}
	sort.Strings(vals)
	return vals
}

func assertSameSet(t *testing.T, what string, pack map[string]bool, source []string) {
	t.Helper()
	inSource := map[string]bool{}
	for _, s := range source {
		inSource[s] = true
	}
	var extra, missing []string
	for k := range pack {
		if !inSource[k] {
			extra = append(extra, k)
		}
	}
	for _, s := range source {
		if !pack[s] {
			missing = append(missing, s)
		}
	}
	sort.Strings(extra)
	sort.Strings(missing)
	if len(extra) > 0 {
		t.Errorf("%s: the pack accepts %v, which the engine does not; a deployment "+
			"using one would validate here and fail to launch", what, extra)
	}
	if len(missing) > 0 {
		t.Errorf("%s: the engine accepts %v, which the pack does not; a valid "+
			"deployment would be rejected", what, missing)
	}
}

func TestAll2AllBackendsMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "parallel.py")
	assertSameSet(t, "All2AllBackends", Pack().All2AllBackends,
		literalMembers(t, src, "All2AllBackend"))
}

func TestDCPCommBackendsMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "parallel.py")
	assertSameSet(t, "DCPCommBackends", Pack().DCPCommBackends,
		literalMembers(t, src, "DCPCommBackend"))
}

// The cache-dtype set is the one an earlier draft got most wrong, so it is checked
// member by member rather than only by size.
func TestCacheDTypesMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "cache.py")
	assertSameSet(t, "CacheDTypes", Pack().CacheDTypes,
		literalMembers(t, src, "CacheDType"))
}

func TestMambaCacheModesMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "cache.py")
	assertSameSet(t, "MambaCacheModes", Pack().MambaCacheModes,
		literalMembers(t, src, "MambaCacheMode"))
}

func TestSchedulerPoliciesMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "scheduler.py")
	assertSameSet(t, "SchedulerPolicies", Pack().SchedulerPolicies,
		literalMembers(t, src, "SchedulerPolicy"))
}

// CUDAGraphMode is an enum class rather than a Literal, so it is parsed differently.
// The fifth member an earlier draft missed, FULL_DECODE_ONLY, is declared as a tuple
// of two other members; the pattern accepts both forms.
func TestCUDAGraphModesMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "compilation.py")
	i := strings.Index(src, "class CUDAGraphMode(")
	if i < 0 {
		t.Fatal("no CUDAGraphMode enum found; the engine may have moved it")
	}
	body := src[i:]
	if j := strings.Index(body, "\n\nclass "); j > 0 {
		body = body[:j]
	}
	re := regexp.MustCompile(`(?m)^\s{4}([A-Z][A-Z_]*)\s*=`)
	var members []string
	for _, m := range re.FindAllStringSubmatch(body, -1) {
		members = append(members, m[1])
	}
	sort.Strings(members)
	assertSameSet(t, "CUDAGraphModes", Pack().CUDAGraphModes, members)
}

func TestOffloadSpecsMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "v1", "kv_offload", "factory.py")
	re := regexp.MustCompile(`register_spec\(\s*"(\w+)"`)
	var specs []string
	for _, m := range re.FindAllStringSubmatch(src, -1) {
		specs = append(specs, m[1])
	}
	sort.Strings(specs)
	assertSameSet(t, "OffloadSpecs", Pack().OffloadSpecs, specs)
}

// Quantizations is the QuantizationMethods literal — a plain Literal[...] like the cache
// and backend sets, so it uses the same extractor. The engine extends QUANTIZATION_METHODS
// at runtime for out-of-tree methods, but the literal is the in-tree set the pack mirrors;
// a divergence here means a method was added or renamed in the release.
func TestQuantizationsMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "model_executor", "layers", "quantization", "__init__.py")
	assertSameSet(t, "Quantizations", Pack().Quantizations,
		literalMembers(t, src, "QuantizationMethods"))
}

// Connectors is a registry like OffloadSpecs: names passed to register_connector rather
// than a Literal, so it is extracted the same way. The first string argument of each call
// is the connector name a deployment selects.
func TestConnectorsMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "distributed", "kv_transfer", "kv_connector", "factory.py")
	re := regexp.MustCompile(`register_connector\(\s*"([^"]+)"`)
	var names []string
	for _, m := range re.FindAllStringSubmatch(src, -1) {
		names = append(names, m[1])
	}
	sort.Strings(names)
	assertSameSet(t, "Connectors", Pack().Connectors, names)
}

// EvictionPolicies is the CachePolicyFactory registry — register_cache_policy calls in
// vllm/v1/kv_offload/cpu/policies/factory.py, the same shape as the connector registry (the
// name is the call's first string argument, on the line after the paren). A divergence means
// a built-in policy was added or renamed in the release.
func TestEvictionPoliciesMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "v1", "kv_offload", "cpu", "policies", "factory.py")
	re := regexp.MustCompile(`register_cache_policy\(\s*"([^"]+)"`)
	var names []string
	for _, m := range re.FindAllStringSubmatch(src, -1) {
		names = append(names, m[1])
	}
	sort.Strings(names)
	assertSameSet(t, "EvictionPolicies", Pack().EvictionPolicies, names)
}

// The numeric constants are single values rather than sets, so each is matched
// against its declaration site.
func TestNumericConstantsMatchSource(t *testing.T) {
	root := vllmSource(t)
	p := Pack()

	parallel := readSource(t, root, "vllm", "config", "parallel.py")
	for field, want := range map[string]int{
		"dbo_decode_token_threshold":  p.DefaultDBODecodeThreshold,
		"dbo_prefill_token_threshold": p.DefaultDBOPrefillThreshold,
	} {
		re := regexp.MustCompile(field + `: int = Field\(default=(\d+)`)
		m := re.FindStringSubmatch(parallel)
		if m == nil {
			t.Errorf("%s: no default found in the engine's parallel config", field)
			continue
		}
		if m[1] != itoa(want) {
			t.Errorf("%s: pack has %d, the engine declares %s", field, want, m[1])
		}
	}

	triton := readSource(t, root, "vllm", "v1", "kv_offload", "cpu",
		"swap_blocks_triton.py")
	if m := regexp.MustCompile(`NUM_SMS = (\d+)`).FindStringSubmatch(triton); m != nil {
		if m[1] != itoa(p.TritonFetchSMs) {
			t.Errorf("TritonFetchSMs: pack has %d, the engine declares %s",
				p.TritonFetchSMs, m[1])
		}
	} else {
		t.Error("NUM_SMS not found in the engine's Triton swap-blocks module")
	}
	if m := regexp.MustCompile(`THRESHOLD_BYTES = (\d+) \* 1024`).FindStringSubmatch(triton); m != nil {
		if m[1] != itoa(p.TritonFetchPageBytes/1024) {
			t.Errorf("TritonFetchPageBytes: pack has %d KiB, the engine declares %s KiB",
				p.TritonFetchPageBytes/1024, m[1])
		}
	} else {
		t.Error("THRESHOLD_BYTES not found in the engine's Triton swap-blocks module")
	}

	custom := readSource(t, root, "vllm", "distributed", "device_communicators",
		"custom_all_reduce.py")
	m := regexp.MustCompile(`_SUPPORTED_WORLD_SIZES = \[([\d, ]+)\]`).FindStringSubmatch(custom)
	if m == nil {
		t.Fatal("_SUPPORTED_WORLD_SIZES not found; the custom all-reduce may have moved")
	}
	declared := map[int]bool{}
	for _, part := range strings.Split(m[1], ",") {
		declared[atoi(strings.TrimSpace(part))] = true
	}
	for k := range p.CustomAllReduceWorldSizes {
		if !declared[k] {
			t.Errorf("world size %d is in the pack but not supported by the engine", k)
		}
	}
	for k := range declared {
		if !p.CustomAllReduceWorldSizes[k] {
			t.Errorf("world size %d is supported by the engine but absent from the pack", k)
		}
	}

	modelCfg := readSource(t, root, "vllm", "config", "model.py")
	optIn := strings.Contains(modelCfg, "disable_cascade_attn: bool = True")
	if optIn != p.CascadeAttnOptIn {
		t.Errorf("CascadeAttnOptIn: pack has %v, the engine's default says %v",
			p.CascadeAttnOptIn, optIn)
	}
}

// The sequence-parallel MoE subset is derived from a property body rather than a
// declaration, so it is extracted from that function's text.
func TestSPMoEBackendsMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "parallel.py")
	i := strings.Index(src, "def use_sequence_parallel_moe(self) -> bool:")
	if i < 0 {
		t.Fatal("use_sequence_parallel_moe not found; the engine may have changed " +
			"how it decides sequence-parallel MoE")
	}
	body := src[i:]
	if j := strings.Index(body, "\n    @property"); j > 0 {
		body = body[:j]
	}
	var names []string
	for _, m := range regexp.MustCompile(`"([a-z0-9_]+)"`).FindAllStringSubmatch(body, -1) {
		names = append(names, m[1])
	}
	sort.Strings(names)
	assertSameSet(t, "SPMoEBackends", Pack().SPMoEBackends, names)
}

// The engine has no allreduce-backend enum. This test asserts the absence, because
// the pack's shape depends on it: an earlier draft invented such an enum.
func TestNoAllReduceBackendEnumMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "parallel.py")
	if regexp.MustCompile(`AllReduceBackend\s*=\s*Literal\[`).MatchString(src) {
		t.Error("the engine now declares an AllReduceBackend literal; the pack " +
			"models this as a boolean and should be revisited")
	}
	if !strings.Contains(src, "disable_custom_all_reduce: bool") {
		t.Error("disable_custom_all_reduce is no longer a boolean; the deployment " +
			"field mirrors it and should be revisited")
	}
}

// itoa is declared in pack.go and reused here.

func atoi(s string) int {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return -1
		}
		n = n*10 + int(c-'0')
	}
	return n
}

// literalTree resolves a Literal alias whose members may name other Literal aliases, as
// SpeculativeMethod nests EagleModelTypes, which nests MTPModelTypes. Python flattens a
// nested Literal into its members, so the engine accepts exactly the leaves.
func literalTree(t *testing.T, src, name string) []string {
	t.Helper()
	re := regexp.MustCompile(`(?s)\b` + name + `\s*=\s*Literal\[(.*?)\]\n`)
	m := re.FindStringSubmatch(src)
	if m == nil {
		t.Fatalf("no Literal named %s found; the engine may have changed how it "+
			"declares this vocabulary", name)
	}
	var out []string
	for _, part := range strings.Split(m[1], ",") {
		part = strings.TrimSpace(part)
		switch {
		case part == "":
		case strings.HasPrefix(part, `"`):
			out = append(out, strings.Trim(part, `"`))
		default:
			out = append(out, literalTree(t, src, part)...)
		}
	}
	sort.Strings(out)
	return out
}

// The speculative-method set is the whole SpeculativeMethod literal, so unlike v0.29's
// curated set it can be checked member by member.
func TestSpeculativeMethodsMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "speculative.py")
	assertSameSet(t, "SpeculativeMethods", Pack().SpeculativeMethods,
		literalTree(t, src, "SpeculativeMethod"))
}

// The async-compatible set is what vllm/config/vllm.py admits when async scheduling is
// requested: the EAGLE and n-gram-GPU literals plus three methods named one by one. The
// test reads the condition's text, so a method added to or dropped from it fails here.
func TestAsyncCompatibleSpecMatchSource(t *testing.T) {
	root := vllmSource(t)
	spec := readSource(t, root, "vllm", "config", "speculative.py")
	cfg := readSource(t, root, "vllm", "config", "vllm.py")
	// The check is the one that hard-fails an explicit request, identified by its
	// comment; an earlier branch on the same flag does something else.
	i := strings.Index(cfg, "# Async scheduling explicitly enabled, hard fail any incompatibilities.")
	if i < 0 {
		t.Fatal("the async-scheduling check was not found; it may have moved")
	}
	body := cfg[i:]
	k := strings.Index(body, "if self.speculative_config is not None:")
	if k < 0 {
		t.Fatal("the async-scheduling check no longer tests the speculative method")
	}
	body = body[k:]
	if j := strings.Index(body, "raise ValueError("); j > 0 {
		body = body[:j]
	}
	var names []string
	for _, m := range regexp.MustCompile(`get_args\((\w+)\)`).FindAllStringSubmatch(body, -1) {
		names = append(names, literalTree(t, spec, m[1])...)
	}
	for _, m := range regexp.MustCompile(`method != "([a-z_0-9]+)"`).FindAllStringSubmatch(body, -1) {
		names = append(names, m[1])
	}
	sort.Strings(names)
	assertSameSet(t, "AsyncCompatibleSpec", Pack().AsyncCompatibleSpec, names)
}

// Each startup refusal a 0.31.0 rule encodes is asserted present by its message text, and
// the PCP-with-DP refusal 0.29.0 had is asserted absent. This pins that the refusals exist,
// not the conditions that trigger them, which rest on review and on the rules' own tests;
// a release that drops a refusal fails here rather than leaving a rule refusing what the
// engine runs.
func TestRefusalMessagesMatchSource(t *testing.T) {
	root := vllmSource(t)
	parallel := readSource(t, root, "vllm", "config", "parallel.py")
	cfg := readSource(t, root, "vllm", "config", "vllm.py")
	engram := readSource(t, root, "vllm", "config", "engram.py")
	spec := readSource(t, root, "vllm", "config", "speculative.py")
	modelCfg := readSource(t, root, "vllm", "config", "model.py")
	for _, c := range []struct{ rule, src, text string }{
		{"engram-config-valid", engram, "EngramConfig requires a model with supported Engram"},
		{"engram-config-valid", engram, "use_thp requires cpu_offload=True and dp_shared_memory=False"},
		{"deepseek-v41-refuses-mtp", spec, "DeepSeek V4.1 has no classic-MTP draft"},
		{"deepseek-v41-refuses-mtp", spec, `if hf_config.model_type in ("deepseek_v4", "deepseek_v41"):`},
		{"deepseek-v41-refuses-mtp", spec, `if self.method in get_args(MTPModelTypes) and self.method != "mtp":`},
		{"moonep-requirements", cfg, "The moonep all2all backend currently supports BF16 models"},
		{"engram-config-valid", engram, "dp_shared_memory requires cpu_offload=True"},
		{"engram-config-valid", engram, "dp_shared_memory requires data_parallel_size > 1."},
		{"max-model-len-within-context-window", modelCfg, "is greater \"\n                f\"than the derived max_model_len"},
		{"dbo-thresholds-at-least-microbatches", parallel, "be at least the number of microbatches"},
		{"engram-excludes-dbo", cfg, "DeepSeek V4.1 Engram does not support DBO or microbatching"},
		{"moonep-requirements", cfg, "The moonep all2all backend currently supports unquantized"},
		{"moonep-requirements", cfg, "The moonep all2all backend does not support EPLB yet"},
		{"cp-kv-cache-interleave-fits-block", cfg, `has_connector("NixlConnector")`},
	} {
		if !strings.Contains(c.src, c.text) {
			t.Errorf("%s: the engine no longer contains %q", c.rule, c.text)
		}
	}
	if strings.Contains(parallel, "PCP does not support data parallelism yet") {
		t.Error("the engine refuses PCP with DP again; this pack has no rule for it")
	}
}

// A recurrent state's cache dtype has its own literal, MambaDType, which is not CacheDType.
func TestMambaDTypesMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "cache.py")
	assertSameSet(t, "MambaDTypes", Pack().MambaDTypes, literalMembers(t, src, "MambaDType"))
}

func TestMoEBackendsMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "kernel.py")
	assertSameSet(t, "MoEBackends", Pack().MoEBackends, literalMembers(t, src, "MoEBackend"))
	// MEGA_MOE_BACKENDS is deep_gemm_mega_moe united with FLASHINFER_MOE_EP_BACKENDS.
	i := strings.Index(src, "FLASHINFER_MOE_EP_BACKENDS = frozenset(")
	j := strings.Index(src, "MEGA_MOE_BACKENDS = frozenset(")
	if i < 0 || j < 0 {
		t.Fatal("the mega-MoE backend sets were not found; they may have moved")
	}
	ep := src[i:]
	ep = ep[:strings.Index(ep, ")")]
	mega := src[j:]
	mega = mega[:strings.Index(mega, "\n")]
	if !strings.Contains(mega, "| FLASHINFER_MOE_EP_BACKENDS") {
		t.Fatalf("MEGA_MOE_BACKENDS no longer includes the flashinfer set: %s", mega)
	}
	var names []string
	for _, s := range []string{ep, mega} {
		for _, m := range regexp.MustCompile(`"([a-z0-9_]+)"`).FindAllStringSubmatch(s, -1) {
			names = append(names, m[1])
		}
	}
	sort.Strings(names)
	assertSameSet(t, "MegaMoEBackends", Pack().MegaMoEBackends, names)
}

func TestDefaultAll2AllBackendMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "parallel.py")
	m := regexp.MustCompile(`all2all_backend: All2AllBackend = "([a-z_0-9]+)"`).FindStringSubmatch(src)
	if m == nil {
		t.Fatal("the all2all_backend default was not found; it may have moved")
	}
	if m[1] != Pack().DefaultAll2AllBackend {
		t.Errorf("DefaultAll2AllBackend: pack has %q, the engine declares %q", Pack().DefaultAll2AllBackend, m[1])
	}
}

// EngramArchitectures is _NGRAM_LAYER_FIELDS' keys, and EngramCPUOffloadDefault its
// cpu_offload default, which the engram-config-valid and engram-residency-default rules read.
func TestEngramMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "engram.py")
	i := strings.Index(src, "_NGRAM_LAYER_FIELDS = {")
	if i < 0 {
		t.Fatal("_NGRAM_LAYER_FIELDS was not found; it may have moved")
	}
	body := src[i:]
	body = body[:strings.Index(body, "}")]
	var archs []string
	for _, m := range regexp.MustCompile(`"([A-Za-z0-9]+)":`).FindAllStringSubmatch(body, -1) {
		archs = append(archs, m[1])
	}
	sort.Strings(archs)
	assertSameSet(t, "EngramArchitectures", Pack().EngramArchitectures, archs)
	m := regexp.MustCompile(`cpu_offload: bool = (True|False)`).FindStringSubmatch(src)
	if m == nil {
		t.Fatal("EngramConfig.cpu_offload's default was not found")
	}
	if got := m[1] == "True"; got != Pack().EngramCPUOffloadDefault {
		t.Errorf("EngramCPUOffloadDefault: pack has %v, the engine declares %s", Pack().EngramCPUOffloadDefault, m[1])
	}
}

// The deepseek-v41-refuses-mtp rule refuses exactly the MTPModelTypes names, so that list
// is checked against the engine on its own, not only as part of SpeculativeMethods.
func TestMTPMethodsMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "speculative.py")
	assertSameSet(t, "mtpMethods", set(mtpMethods...), literalMembers(t, src, "MTPModelTypes"))
}

// spCondition parses an architecture's _use_sequence_parallel into the condition the pack
// encodes, clause by clause, failing on a clause it does not recognize -- so a release that
// adds, drops or changes one fails here rather than leaving the pack's data stale.
func spCondition(t *testing.T, src string) rules.SequenceParallelCondition {
	t.Helper()
	i := strings.Index(src, "def _use_sequence_parallel(")
	if i < 0 {
		t.Fatal("_use_sequence_parallel not found; the model may have moved its rule")
	}
	body := src[i:]
	r := strings.Index(body, "return (")
	if r < 0 {
		t.Fatal("_use_sequence_parallel no longer returns one parenthesized condition")
	}
	body = body[r+len("return ("):]
	body = body[:strings.Index(body, "\n    )")]
	var c rules.SequenceParallelCondition
	for _, clause := range strings.Split(body, "\n") {
		clause = strings.TrimPrefix(strings.TrimSpace(clause), "and ")
		switch clause {
		case "":
		case "parallel_config.pipeline_parallel_size == 1":
			c.PipelineParallelOne = true
		case "parallel_config.enable_expert_parallel":
			c.ExpertParallel = true
		case "parallel_config.tensor_parallel_size > 1":
			c.TensorParallelAboveOne = true
		case "(use_mega_moe or parallel_config.data_parallel_size > 1)":
			c.MegaMoEOrDataParallelAboveOne = true
		default:
			t.Fatalf("_use_sequence_parallel has a clause the pack cannot encode: %q", clause)
		}
	}
	if !strings.Contains(src, "use_mega_moe = vllm_config.kernel_config.moe_backend in MEGA_MOE_BACKENDS") {
		t.Error("use_mega_moe no longer reads moe_backend against MEGA_MOE_BACKENDS")
	}
	return c
}

// Each architecture the pack gives a sequence-parallelism rule has that rule, parsed from
// its own model module, and every module that defines one is in the pack.
func TestModelSequenceParallelRulesMatchSource(t *testing.T) {
	root := vllmSource(t)
	modules := map[string][]string{
		rules.ArchDeepseekV4:  {"vllm", "models", "deepseek_v4", "nvidia", "model.py"},
		rules.ArchDeepseekV41: {"vllm", "models", "deepseek_v41", "nvidia", "model.py"},
	}
	for arch, parts := range modules {
		if _, err := os.Stat(filepath.Join(append([]string{root}, parts...)...)); err != nil {
			if _, inPack := Pack().ModelSequenceParallelRules[arch]; inPack {
				t.Errorf("%s has a rule in the pack but no module in this release", arch)
			}
			continue
		}
		want := spCondition(t, readSource(t, root, parts...))
		got, ok := Pack().ModelSequenceParallelRules[arch]
		if !ok || got != want {
			t.Errorf("%s: pack has %+v (present %v), the engine's rule is %+v", arch, got, ok, want)
		}
	}
}

// The collectives a model-level sequence parallelism changes, for each architecture the
// pack gives a rule: the attention input is all-gathered and its output reduce-scattered,
// the MoE and its shared expert take the model's flag, the shared expert's weights are
// replicated under it, the embedding runs before the stream is sharded, and FusedMoE skips
// its final all-reduce. The model-sequence-parallel-collectives rule rests on each.
func TestModelSequenceParallelCollectivesMatchSource(t *testing.T) {
	root := vllmSource(t)
	for _, dir := range []string{"deepseek_v4", "deepseek_v41"} {
		base := filepath.Join(root, "vllm", "models", dir)
		if _, err := os.Stat(base); err != nil {
			continue
		}
		err := filepath.WalkDir(base, func(path string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() || !strings.HasSuffix(path, ".py") {
				return err
			}
			b, err := os.ReadFile(path)
			if err == nil && strings.Contains(string(b), "use_sequence_parallel_moe") {
				t.Errorf("%s reads use_sequence_parallel_moe; the rule's premise no longer holds", path)
			}
			return err
		})
		if err != nil {
			t.Fatal(err)
		}
		model := readSource(t, root, "vllm", "models", dir, "nvidia", "model.py")
		for _, want := range []string{
			"x = sp_all_gather(x)[: positions.shape[0]]",
			"x = sp_reduce_scatter(x)",
			"use_sequence_parallel=self.use_sequence_parallel,",
		} {
			if !strings.Contains(model, want) {
				t.Errorf("%s: %q is gone", dir, want)
			}
		}
		embed := strings.Index(model, "hidden_states = self.embed_input_ids(input_ids)")
		shard := strings.Index(model, "hidden_states = sp_shard(hidden_states)")
		if embed < 0 || shard < 0 || embed > shard {
			t.Errorf("%s: the embedding no longer runs before the stream is sharded", dir)
		}
	}
	v4 := readSource(t, root, "vllm", "models", "deepseek_v4", "nvidia", "model.py")
	if strings.Count(v4, "disable_tp=is_sequence_parallel,") < 2 ||
		!strings.Contains(v4, "replicated and no collective ops are needed") {
		t.Error("the shared expert is no longer replicated under sequence parallelism")
	}
	// The mega-MoE path keeps the same reduction outside sequence parallelism, which it
	// leaves only at pp above 1 (it requires expert parallelism, and the model's rule
	// turns sequence parallelism on for a mega backend at pp 1 and tp above 1): the routed
	// experts combine over the expert-parallel group with no tensor-parallel all-reduce,
	// and the shared expert, unfused there, all-reduces its own hidden-width output.
	for _, c := range []struct{ what, text string }{
		{"mega-MoE requiring expert parallelism", "MegaMoE currently requires expert parallel."},
		{"the shared expert reducing itself under mega-MoE", "reduce_results=self.use_mega_moe,"},
		{"the shared expert fusing only under sequence parallelism or tp 1", "and (self.use_sequence_parallel or self.tp_size == 1)"},
		{"the unfused shared expert added to the routed output", "and not self.experts.has_fused_shared_experts"},
	} {
		if !strings.Contains(v4, c.text) {
			t.Errorf("%s changed: %q is gone", c.what, c.text)
		}
	}
	mega := "final_hidden_states = self.experts(\n            hidden_states,\n            topk_weights,"
	at := strings.Index(v4, mega)
	if at < 0 {
		t.Fatal("the mega-MoE forward path was not found; it may have moved")
	}
	block := v4[at:]
	block = block[:strings.Index(block, "return final_hidden_states.view(org_shape)")]
	if strings.Contains(block, "all_reduce") {
		t.Error("the mega-MoE forward path now all-reduces its routed output")
	}
	runner := readSource(t, root, "vllm", "model_executor", "layers", "fused_moe", "runner", "moe_runner.py")
	i := strings.Index(runner, "def _maybe_reduce_final_output(")
	if i < 0 {
		t.Fatal("_maybe_reduce_final_output not found; FusedMoE's reduction may have moved")
	}
	body := runner[i:]
	if j := strings.Index(body[1:], "\n    def "); j > 0 {
		body = body[:j+1]
	}
	if !strings.Contains(body, "not self.moe_config.is_sequence_parallel") ||
		!strings.Contains(body, "tensor_model_parallel_all_reduce(") {
		t.Error("FusedMoE's final all-reduce no longer depends on sequence parallelism")
	}
}

// The facts the schema's sequence-parallel split and draft embedding rest on: sp_shard
// pads the token count to a multiple of tp before slicing, so a rank holds ceil(tokens /
// tp) rows; Engram's wkv is column-split without the mode and replicated with it; and
// V4.1's DSpark drafter embeds, through a vocab-parallel lookup whose forward ends in a
// tensor-parallel all-reduce, before it shards its stream.
func TestSequenceParallelSplitsMatchSource(t *testing.T) {
	root := vllmSource(t)
	sp := readSource(t, root, "vllm", "models", "common", "ops", "sequence_parallel.py")
	for _, want := range []string{
		"sp_pad = (-x.shape[0]) % tp_size",
		"chunk = x.shape[0] // tp_size",
		"return x[tp_rank * chunk : (tp_rank + 1) * chunk]",
	} {
		if !strings.Contains(sp, want) {
			t.Errorf("sp_shard changed: %q is gone", want)
		}
	}
	engram := readSource(t, root, "vllm", "models", "deepseek_v41", "common", "engram.py")
	at := strings.Index(engram, "self.wkv = ColumnParallelLinear(")
	if at < 0 {
		t.Fatal("Engram's wkv is no longer a ColumnParallelLinear")
	}
	if end := strings.Index(engram[at:], "\n        )"); end < 0 ||
		!strings.Contains(engram[at:at+end], "disable_tp=use_sequence_parallel,") {
		t.Error("Engram's wkv no longer turns its split off under sequence parallelism")
	}
	dspark := readSource(t, root, "vllm", "models", "deepseek_v41", "nvidia", "dspark.py")
	if !strings.Contains(dspark, "self.embed_tokens = VocabParallelEmbedding(") {
		t.Error("the DSpark drafter no longer holds a vocab-parallel embedding")
	}
	fwd := strings.Index(dspark, "def forward(")
	if fwd < 0 {
		t.Fatal("the DSpark drafter's forward was not found")
	}
	embed := strings.Index(dspark[fwd:], "inputs_embeds = self.embed_input_ids(input_ids)")
	shard := strings.Index(dspark[fwd:], "inputs_embeds = sp_shard(inputs_embeds)")
	if embed < 0 || shard < 0 || embed > shard {
		t.Error("the DSpark drafter no longer embeds before it shards its stream")
	}
	vocab := readSource(t, root, "vllm", "model_executor", "layers", "vocab_parallel_embedding.py")
	if !strings.Contains(vocab, "        # Reduce across all the model parallel GPUs.\n") ||
		!strings.Contains(vocab, "return tensor_model_parallel_all_reduce(output_parallel)") {
		t.Error("the vocab-parallel lookup no longer ends in a tensor-parallel all-reduce")
	}
}

// pplx and naive run as allgather_reducescatter.
func TestAll2AllBackendRenamesMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "parallel.py")
	if !strings.Contains(src, `if self.all2all_backend in ["pplx", "naive"]:`) ||
		!strings.Contains(src, `self.all2all_backend = "allgather_reducescatter"`) {
		t.Error("the engine no longer replaces pplx and naive with allgather_reducescatter")
	}
	want := map[string]string{"pplx": "allgather_reducescatter", "naive": "allgather_reducescatter"}
	if !reflect.DeepEqual(Pack().All2AllBackendRenames, want) {
		t.Errorf("All2AllBackendRenames: pack has %v, the engine %v", Pack().All2AllBackendRenames, want)
	}
}

// The engine lower-cases --moe-backend and turns dashes into underscores, as
// rules.NormalizeMoEBackend does.
func TestMoEBackendNormalizationMatchSource(t *testing.T) {
	root := vllmSource(t)
	if !strings.Contains(readSource(t, root, "vllm", "config", "kernel.py"),
		`return value.lower().replace("-", "_")`) {
		t.Error("the engine no longer normalizes moe_backend as rules.NormalizeMoEBackend does")
	}
}

// WindowPagesPerRequest's formula is the engine's per-request admission cap.
func TestWindowPagesMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "v1", "kv_cache_interface.py")
	for _, want := range []string{
		"self.sliding_window - 1 + self.extra_retained_tokens + max_in_flight_tokens,",
		"return cdiv(num_tokens, self.block_size) + 1",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("the window admission cap changed: %q is gone", want)
		}
	}
}

// The block size validate_block_size checks is the smallest prefix-cacheable group's,
// which effectiveBlocks encodes.
func TestEffectiveBlockMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "v1", "engine", "core.py")
	for _, want := range []string{"if g.kv_cache_spec.prefix_cacheable", "vllm_config.cache_config.block_size = min(", "vllm_config.validate_block_size()"} {
		if !strings.Contains(src, want) {
			t.Errorf("the block-size replacement changed: %q is gone", want)
		}
	}
}

// The resolutions ResolveCacheDType encodes for DeepSeek-V4.1: on the packed layout auto
// and fp8 become the backend's record; FlashMLA's is fp8_ds_mla and mega attention's
// nvfp4_ds_mla; mega attention is SM100's default where it can serve the width, which
// needs tp to divide the heads and output groups. The pack's TPDivides of 8 is
// DeepSeek-V4.1-Flash's o_groups, which every width dividing it also leaves 8 heads per
// group (64 heads), as the rule's last clause needs.
func TestCacheDTypeResolutionsMatchSource(t *testing.T) {
	root := vllmSource(t)
	attn := readSource(t, root, "vllm", "models", "deepseek_v41", "attention.py")
	mega := readSource(t, root, "vllm", "models", "deepseek_v41", "nvidia", "flash_mla_mega_attn.py")
	sel := readSource(t, root, "vllm", "models", "deepseek_v41", "nvidia", "model.py")
	for _, c := range []struct{ src, what, text string }{
		{attn, "the auto/fp8 rewrite", `if kv_cache_dtype in ("auto", "fp8"):`},
		{attn, "FlashMLA's record", `return "fp8_ds_mla"`},
		{mega, "mega attention's record", `return "nvfp4_ds_mla"`},
		{mega, "the SM100 requirement", "if not current_platform.is_device_capability_family(100):"},
		{mega, "the width rule", "if n_heads % tp_size or n_groups % tp_size:"},
		{mega, "the group rule", "return n_local_heads // n_local_groups == WV_GROUP_SIZE"},
		{sel, "mega attention as the default", "if DeepseekV4MegaAttnAttention.is_available_for(vllm_config):"},
	} {
		if !strings.Contains(c.src, c.text) {
			t.Errorf("%s changed: %q is gone", c.what, c.text)
		}
	}
}

// pyFunction returns a top-level Python function's text, from its def line to the next
// top-level def, one newline per line.
func pyFunction(t *testing.T, src, name string) string {
	t.Helper()
	var b strings.Builder
	in := false
	for _, line := range strings.Split(src, "\n") {
		if strings.HasPrefix(line, "def ") {
			in = strings.HasPrefix(line, "def "+name+"(")
		}
		if in {
			b.WriteString(line + "\n")
		}
	}
	if b.Len() == 0 {
		t.Fatalf("no top-level def %s", name)
	}
	return b.String()
}

// KVCacheGroups transcribes these three functions, and its goldens were computed by
// running them (rules/testdata/kvgroups/golden.py). A changed body is a changed grouping,
// so each is pinned by digest; re-derive the transcription and the goldens when one moves.
// The specs it maps caches to are pinned too: a state cache is a CircularBufferSpec.
func TestKVCachePackingMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "v1", "core", "kv_cache_utils.py")
	for fn, want := range map[string]string{
		"_approximate_gcd":              "22370757cf8c20dba515cf9452594dfc484f3057a709c704214ebb68328b4f1c",
		"_get_packed_kv_cache_groups":   "5f0f15282481429cf5939d3d63a201c95b01363079ee3021c9cfe2e14f718ff3",
		"_get_kv_cache_bytes_per_block": "169b900872e70e8bcd132d27bde67480512caa16e67784d13f20ff5ca8afb733",
	} {
		sum := sha256.Sum256([]byte(pyFunction(t, src, fn)))
		if got := hex.EncodeToString(sum[:]); got != want {
			t.Errorf("%s changed (sha256 %s); re-derive KVCacheGroups and its goldens", fn, got)
		}
	}
	comp := readSource(t, root, "vllm", "models", "deepseek_v41", "compressor.py")
	if !strings.Contains(comp, "return CircularBufferSpec(") {
		t.Error("DeepSeek-V4.1's compressor state is no longer a CircularBufferSpec")
	}
	for _, want := range []string{
		"rows_per_step = get_current_vllm_config().num_speculative_tokens + 2",
		"self.block_size = max(8, 1 << (rows_per_step - 1).bit_length())",
	} {
		if !strings.Contains(comp, want) {
			t.Errorf("the compressor ring size changed: %q is gone; re-derive CompressorRingRows", want)
		}
	}
	// Only the MLA specs pad a page to an alignment; a circular buffer never does, which
	// is why a state cache's page_alignment is refused.
	iface := readSource(t, root, "vllm", "v1", "kv_cache_interface.py")
	for _, class := range []string{"class MLAAttentionSpec(", "class SlidingWindowMLASpec("} {
		body := classBody(iface, class)
		if !strings.Contains(body, "_apply_alignment_padding(self)") {
			t.Errorf("%s no longer pads its page", class)
		}
	}
	if body := classBody(iface, "class CircularBufferSpec("); body == "" || strings.Contains(body, "_apply_alignment_padding") || strings.Contains(body, "    alignment:") {
		t.Error("CircularBufferSpec now carries an alignment; re-derive the state-cache page")
	}
}

// Pack.CUDAGraphCaptureSizes and the cudagraph-capture-sizes rule transcribe
// _set_cudagraph_sizes; the defaults the documentation states are its too. Each
// transcribed line must still be in the release's source.
func TestCUDAGraphCaptureSizesMatchSource(t *testing.T) {
	root := vllmSource(t)
	src := readSource(t, root, "vllm", "config", "vllm.py")
	for _, want := range []string{
		"1024 if current_platform.is_device_capability_family(100) else 512",
		"max_num_seqs * decode_query_len * 2, default_max_graph_size",
		"max_cudagraph_capture_size = min(max_num_tokens, max_cudagraph_capture_size)",
		"dedup_sizes = list(set(self.compilation_config.cudagraph_capture_sizes))",
		"i for i in dedup_sizes if i <= max_num_tokens",
		"cudagraph_capture_sizes = [\n                        i for i in [1, 2, 4] if i <= max_cudagraph_capture_size\n                    ]",
		"range(8, min(max_cudagraph_capture_size + 1, 256), 8)",
		"range(256, max_cudagraph_capture_size + 1, 16)",
		"max_num_tokens <= max_cudagraph_capture_size",
		"cudagraph_capture_sizes[-1] if cudagraph_capture_sizes else 0",
		"\"customized max_cudagraph_capture_size\"",
		"and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("_set_cudagraph_sizes changed: %q is gone", want)
		}
	}
	// The stated-maximum path adds no uniform decode sizes: they are populated only when
	// the default is computed.
	if !strings.Contains(src, "uniform_decode_sizes: list[int] = []") {
		t.Error("the uniform decode sizes are no longer empty for a stated maximum")
	}
}

// classBody returns a top-level Python class's text, from its class line to the next
// top-level definition or decorator.
func classBody(src, header string) string {
	i := strings.Index(src, "\n"+header)
	if i < 0 {
		return ""
	}
	rest := src[i+1:]
	lines := strings.Split(rest, "\n")
	var b strings.Builder
	for j, line := range lines {
		if j > 0 && line != "" && line[0] != ' ' && line[0] != '\t' && line[0] != ')' {
			break
		}
		b.WriteString(line + "\n")
	}
	return b.String()
}

// long_prefill_token_threshold: its default and floor, the startup refusal above
// max_model_len, the per-step cap, and this release's waiver for a lone request.
func TestLongPrefillTokenThresholdMatchSource(t *testing.T) {
	root := vllmSource(t)
	cfg := readSource(t, root, "vllm", "config", "scheduler.py")
	for _, want := range []string{
		"long_prefill_token_threshold: int = Field(default=0, ge=0)",
		"if self.long_prefill_token_threshold > max_model_len:",
	} {
		if !strings.Contains(cfg, want) {
			t.Errorf("scheduler.py changed: %q is gone", want)
		}
	}
	sched := readSource(t, root, "vllm", "v1", "core", "sched", "scheduler.py")
	for _, want := range []string{
		"if 0 < long_prefill_token_threshold < num_new_tokens:",
		"if num_eligible_reqs > 1\n            else 0",
	} {
		if !strings.Contains(sched, want) {
			t.Errorf("the scheduler's cap changed: %q is gone", want)
		}
	}
	if !Pack().LongPrefillExemptsLoneRequest {
		t.Error("the pack no longer records the lone-request waiver the source states")
	}
}

// The three load-balancing modes, how the engine resolves hybrid, the refusals the
// data-parallel-load-balancing rule encodes, and which client each mode gets.
func TestDataParallelLoadBalancingMatchSource(t *testing.T) {
	root := vllmSource(t)
	args := readSource(t, root, "vllm", "engine", "arg_utils.py")
	for _, want := range []string{
		"data_parallel_hybrid_lb: bool = False",
		"data_parallel_external_lb: bool = False",
		`"--data-parallel-multi-port-external-lb",`,
		"Non-MoE models do not support external data parallel mode.",
		"`--data-parallel-hybrid-lb` requires ",
		"data_parallel_hybrid_lb is not eligible when ",
		"if data_parallel_size_local == self.data_parallel_size:",
		"`--data-parallel-external-lb` cannot be enabled together.",
	} {
		if !strings.Contains(args, want) {
			t.Errorf("arg_utils.py changed: %q is gone", want)
		}
	}
	client := readSource(t, root, "vllm", "v1", "engine", "core_client.py")
	ext := strings.Index(client, "if parallel_config.data_parallel_external_lb:")
	if ext < 0 || !strings.Contains(client[ext:ext+300], "DPAsyncMPClient(") ||
		!strings.Contains(client[ext:ext+600], "DPLBAsyncMPClient(") {
		t.Error("core_client.py no longer gives external a per-rank client and internal the balancing one")
	}
	serve := readSource(t, root, "vllm", "entrypoints", "cli", "serve.py")
	if !strings.Contains(serve, "Cannot use more than one data parallel load balancing mode.") {
		t.Error("serve.py no longer refuses two load-balancing modes at once")
	}
	// The api server count only defaults per mode; it is not part of the mode.
	if !strings.Contains(serve, "# Default api_server_count if not explicitly set.") {
		t.Error("serve.py no longer defaults api_server_count per mode")
	}
}

// The frontend is chosen by VLLM_USE_RUST_FRONTEND, off by default, and serve launches
// the Rust binary only when it is set: what deployment.Engine.Frontend's python default
// and rust value transcribe.
func TestFrontendMatchSource(t *testing.T) {
	root := vllmSource(t)
	env := readSource(t, root, "vllm", "envs.py")
	for _, want := range []string{
		"VLLM_USE_RUST_FRONTEND: bool = False",
		`int(os.getenv("VLLM_USE_RUST_FRONTEND", "0"))`,
	} {
		if !strings.Contains(env, want) {
			t.Errorf("envs.py changed: %q is gone", want)
		}
	}
	serve := readSource(t, root, "vllm", "entrypoints", "cli", "serve.py")
	if !strings.Contains(serve, "envs.VLLM_RUST_FRONTEND_PATH if envs.VLLM_USE_RUST_FRONTEND else None") {
		t.Error("serve.py no longer launches the Rust frontend only when VLLM_USE_RUST_FRONTEND is set")
	}
}

// swa_bounded_replay defaults to true and is the --swa-bounded-replay flag: what
// deployment.Engine.SWABoundedReplay's unstated (engine-resolved) value and name transcribe.
func TestSWABoundedReplayMatchSource(t *testing.T) {
	root := vllmSource(t)
	cache := readSource(t, root, "vllm", "config", "cache.py")
	if !strings.Contains(cache, "swa_bounded_replay: bool = True") {
		t.Error("cache.py changed: swa_bounded_replay no longer defaults to True")
	}
	args := readSource(t, root, "vllm", "engine", "arg_utils.py")
	if !strings.Contains(args, `"--swa-bounded-replay", **cache_kwargs["swa_bounded_replay"]`) {
		t.Error("arg_utils.py no longer exposes swa_bounded_replay as --swa-bounded-replay")
	}
}

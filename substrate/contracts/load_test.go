package contracts

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

// The yaml tags on every spec type are a contract with the files in blis-catalog and
// blis-registry. Until something parses a file, those tags are untested: a misspelled
// tag compiles, validates, and silently drops the field it was meant to read.

func write(t *testing.T, name, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("writing %s: %v", name, err)
	}
	return path
}

func TestLoadScenario(t *testing.T) {
	path := write(t, "s.yaml", `
kind: Scenario
name: granite-230b-h200-tp8
model: granite-5-230b
coefficients: [cost-model-primitives-h200]
engine_version: "0.29.0"
workload:
  shape: chatbot
cluster:
  hardware: h200
  nodes: 1
  gpus_per_node: 8
  storage: [cpu_dram, nvme_gen4]
`)
	s, err := LoadScenario(path)
	if err != nil {
		t.Fatalf("LoadScenario: %v", err)
	}
	// Spot-check a field from each nesting level: a tag error at any depth would
	// leave a zero value that the type system cannot catch.
	if s.Name != "granite-230b-h200-tp8" {
		t.Errorf("name = %q", s.Name)
	}
	// The workload is a sum-type binding now, not a bare name: the distributional arm
	// names a catalog shape under `shape`.
	if s.Workload == nil || s.Workload.Shape != "chatbot" || s.Workload.Trace != nil {
		t.Errorf("workload binding did not parse as a shape ref: %+v", s.Workload)
	}
	// The hardware and fabric references and the storage inventory are folded into
	// the cluster rather than sitting loose at the top level.
	if s.Cluster.Hardware != "h200" {
		t.Errorf("cluster.hardware = %q, want h200", s.Cluster.Hardware)
	}
	if s.Cluster.GPUsPerNode != 8 {
		t.Errorf("gpus_per_node = %d, want 8", s.Cluster.GPUsPerNode)
	}
	if len(s.Cluster.Storage) != 2 || s.Cluster.Storage[0] != "cpu_dram" {
		t.Errorf("cluster.storage did not parse: %v", s.Cluster.Storage)
	}
	if rep := Validate(Bundle{Scenario: s}); !rep.Field.OK() {
		t.Errorf("a loaded scenario failed field validation:\n%s", rep.Field.Error())
	}
}

// TestLoadScenarioWithTraceBinding exercises the trace arm of the workload binding end to
// end: the data-file path, the integrity/provenance fields, and the small header metadata
// (including the server sub-block and the per-class SLO targets) must all parse through
// the strict loader, and the loaded scenario must validate. A tag error anywhere in the
// nested trace reference would leave a zero value the type system cannot catch, which is
// the one failure this package exists to rule out.
func TestLoadScenarioWithTraceBinding(t *testing.T) {
	path := write(t, "s.yaml", `
kind: Scenario
name: granite-230b-h200-replay
model: granite-5-230b
coefficients: [cost-model-primitives-h200]
engine_version: "0.29.0"
workload:
  trace:
    data: traces/agentic-run.csv
    sha256: `+strings.Repeat("a", 64)+`
    rows: 1048576
    header:
      trace_version: 3
      time_unit: microseconds
      mode: real
      workload_seed: 0
      server:
        type: vllm
        model: granite-5-230b
        tensor_parallel: 8
        max_num_seqs: 1024
        block_size: 16
        gpu_memory_utilization: 0.9
        max_model_len: 131072
      goodput_slo_targets:
        critical:
          ttft_ms: 500
          itl_ms: 50
          e2e_ms: 30000
cluster:
  hardware: h200
  nodes: 1
  gpus_per_node: 8
`)
	s, err := LoadScenario(path)
	if err != nil {
		t.Fatalf("LoadScenario: %v", err)
	}
	b := s.Workload
	if b == nil || b.Shape != "" || b.Trace == nil {
		t.Fatalf("workload binding did not parse as a trace ref: %+v", b)
	}
	tr := b.Trace
	if tr.Data != "traces/agentic-run.csv" || tr.Rows != 1048576 || len(tr.SHA256) != 64 {
		t.Errorf("trace reference fields did not parse: %+v", tr)
	}
	h := tr.Header
	if h.Version != 3 || h.TimeUnit != "microseconds" || h.Mode != workload.ModeReal {
		t.Errorf("trace header scalars did not parse: %+v", h)
	}
	// workload_seed: 0 is a recorded zero, which the pointer must preserve as distinct
	// from absent.
	if h.WorkloadSeed == nil || *h.WorkloadSeed != 0 {
		t.Errorf("workload_seed did not parse as a recorded zero: %v", h.WorkloadSeed)
	}
	if h.Server == nil || h.Server.TensorParallel != 8 || h.Server.GPUMemoryUtilization != 0.9 {
		t.Errorf("trace server block did not parse: %+v", h.Server)
	}
	crit, ok := h.GoodputSLOTargets["critical"]
	if !ok || crit.TTFTMs != 500 || crit.ITLMs != 50 || crit.E2EMs != 30000 {
		t.Errorf("goodput_slo_targets did not parse: %+v", h.GoodputSLOTargets)
	}
	if rep := Validate(Bundle{Scenario: s}); !rep.Field.OK() {
		t.Errorf("a loaded trace-bound scenario failed field validation:\n%s", rep.Field.Error())
	}
}

// TestLoadScenarioRejectsUnknownTraceField pins strict decoding through EVERY nesting
// level the trace reference introduces: a misspelled sub-key must fail loudly rather than
// leave a zero value, the same guarantee the top-level loaders give. The trace types are
// plain structs — none implements a custom UnmarshalYAML, which is what would silently
// drop the decoder's KnownFields setting (see internal/validate/strict.go) — so strict
// decoding recurses through all of them for free. That is exactly the kind of property
// that is assumed and then quietly lost, so each new level is pinned: at the
// `workload` binding itself, directly under `trace`, under its `header`, under the
// `header.server` sub-block, and inside a `header.goodput_slo_targets.<class>` map VALUE (the one
// struct the feature reaches only through a map, where it is least obvious the strict
// check still applies).
func TestLoadScenarioRejectsUnknownTraceField(t *testing.T) {
	// head is a valid scenario up to the workload block; each case supplies the workload
	// subtree with an injection that must be rejected, naming the stray key.
	cases := []struct {
		name      string
		traceYAML string
		badKey    string
	}{
		{
			name: "unknown key under the workload binding",
			traceYAML: `  shape: chatbot
  invalid_field: yes`,
			badKey: "invalid_field",
		},
		{
			name: "unknown key directly under trace",
			traceYAML: `  trace:
    data: traces/run.csv
    rowz: 10
    header:
      trace_version: 3
      time_unit: microseconds
      mode: real`,
			badKey: "rowz",
		},
		{
			name: "unknown key under trace.header",
			traceYAML: `  trace:
    data: traces/run.csv
    header:
      trace_version: 3
      time_unit: microseconds
      mode: real
      time_uint: microseconds`,
			badKey: "time_uint",
		},
		{
			name: "unknown key under trace.header.server",
			traceYAML: `  trace:
    data: traces/run.csv
    header:
      trace_version: 3
      time_unit: microseconds
      mode: real
      server:
        tensor_parallel: 8
        tensor_paralel: 8`,
			badKey: "tensor_paralel",
		},
		{
			name: "unknown key inside a goodput_slo_targets map value",
			traceYAML: `  trace:
    data: traces/run.csv
    header:
      trace_version: 3
      time_unit: microseconds
      mode: real
      goodput_slo_targets:
        critical:
          ttft_ms: 500
          ttft_mss: 500`,
			badKey: "ttft_mss",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := write(t, "s.yaml", `
kind: Scenario
name: typo
model: granite-5-230b
coefficients: [c]
engine_version: "0.29.0"
workload:
`+tc.traceYAML+`
cluster:
  hardware: h200
  nodes: 1
  gpus_per_node: 8
`)
			_, err := LoadScenario(path)
			if err == nil {
				t.Fatalf("a misspelled %q was accepted; the trace would validate while silently dropping it", tc.badKey)
			}
			if !strings.Contains(err.Error(), tc.badKey) {
				t.Errorf("the error should name the offending field %q, got: %v", tc.badKey, err)
			}
		})
	}
}

// TestLoadWorkload is the by-filename identity contract for a workload Shape. The file
// carries no name — exactly like a chip or a fabric — so a nameless document must load,
// LoadWorkload must stamp Shape.Name from the path stem, and the loaded shape must then
// validate clean (Validate requires a Name, now always loader-supplied). This is the
// catalog's target form from issue #21: a chatbot.yaml that is prefix/prompt/output and
// nothing else.
func TestLoadWorkload(t *testing.T) {
	path := write(t, "chatbot.yaml", `
prefix_tokens: 0
prompt:
  tokens: 256
  tokens_stdev: 100
  tokens_min: 2
  tokens_max: 800
output:
  tokens: 256
  tokens_stdev: 100
  tokens_min: 1
  tokens_max: 1024
`)
	w, err := LoadWorkload(path)
	if err != nil {
		t.Fatalf("LoadWorkload: %v", err)
	}
	// Identity is the filename stem, as with LoadChip/LoadFabric.
	if w.Name != "chatbot" {
		t.Errorf("name = %q; it should come from the filename", w.Name)
	}
	// Spot-check a field from each nesting level: a tag error at any depth would leave a
	// zero value the type system cannot catch.
	if w.PrefixTokens != 0 {
		t.Errorf("prefix_tokens = %d, want 0", w.PrefixTokens)
	}
	if w.Prompt.Mean != 256 || w.Prompt.StdDev != 100 || w.Prompt.Min != 2 || w.Prompt.Max != 800 {
		t.Errorf("prompt distribution did not parse: %+v", w.Prompt)
	}
	if w.Output.Mean != 256 || w.Output.Max != 1024 {
		t.Errorf("output distribution did not parse: %+v", w.Output)
	}
	// A loader-named shape validates: the Name requirement is satisfied by the stamp, not
	// by a line in the file.
	if p := w.Validate(); !p.OK() {
		t.Errorf("a loaded workload failed validation:\n%s", p.Error())
	}
}

// TestLoadWorkloadRejectsInFileName pins that identity is a filename fact, not a file
// field: an in-file `name:` is no longer part of the contract, so under strict decoding
// it is an unknown field and must fail loudly rather than quietly become a second source
// of truth for the identity the filename already fixes. The Name field is tagged
// `yaml:"-"`, so yaml.v3 maps no `name` key to it and KnownFields rejects the stray key.
func TestLoadWorkloadRejectsInFileName(t *testing.T) {
	path := write(t, "chatbot.yaml", `
name: not-chatbot
prefix_tokens: 0
prompt:
  tokens: 256
output:
  tokens: 256
`)
	if _, err := LoadWorkload(path); err == nil {
		t.Fatal("an in-file name was accepted; the filename and the field could disagree")
	} else if !strings.Contains(err.Error(), "name") {
		t.Errorf("the error should name the offending field %q, got: %v", "name", err)
	}
}

// TestLoadWorkloadAcceptsCommentKeys is #28: a workload Shape now carries the catalog's
// `_comment`-prefixed provenance narrative that Chip, Fabric and StorageDevice already
// accept, so a workloads/*.yaml can document why its token figures are what they are —
// following the pattern every hardware/*.yaml sets — rather than failing strict decoding on
// the first `_comment`. The notes are accepted at both nesting levels (top-level and inside
// a prompt/output distribution) and stripped before decoding, so no prose reaches the struct.
func TestLoadWorkloadAcceptsCommentKeys(t *testing.T) {
	path := write(t, "chatbot.yaml", `
_comment: "token figures from the vLLM chat benchmark; see the issue that set them"
prefix_tokens: 0
prompt:
  _comment_min: "min is p01 of the sampled trace, not an assumption"
  tokens: 256
  tokens_stdev: 100
  tokens_min: 2
output:
  tokens: 256
`)
	w, err := LoadWorkload(path)
	if err != nil {
		t.Fatalf("comment keys were rejected: %v", err)
	}
	// The prose is stripped, and the real fields decode normally around it.
	if w.Prompt.Mean != 256 || w.Prompt.StdDev != 100 || w.Prompt.Min != 2 {
		t.Errorf("distribution did not parse around the comment: %+v", w.Prompt)
	}
}

// TestLoadWorkloadStillRejectsNestedTypos guards the regression the #28 hook could have
// introduced. Giving Shape a custom UnmarshalYAML means node.Decode runs, which does NOT
// honour the decoder's KnownFields setting — so without Distribution's own hook a misspelled
// key nested under prompt:/output: would silently decode to a zero value. Distribution
// restores the check, so a typo at the inner level is still a loud error, as it was before
// #28. A comment key is accepted; a typo is not — the distinction strict decoding exists for.
func TestLoadWorkloadStillRejectsNestedTypos(t *testing.T) {
	path := write(t, "chatbot.yaml", `
prefix_tokens: 0
prompt:
  tokens: 256
  tokens_minn: 2
output:
  tokens: 256
`)
	if _, err := LoadWorkload(path); err == nil {
		t.Fatal("a nested misspelled field was accepted; the shape would decode with a wrong " +
			"bound and nothing would report it")
	} else if !strings.Contains(err.Error(), "tokens_minn") {
		t.Errorf("the error should name the offending nested field, got: %v", err)
	}
}

// TestLoadChipRejectsInFileName is the by-filename identity contract for a Chip, the same
// one TestLoadWorkloadRejectsInFileName pins for a Shape. #27 made Chip.Name a filename
// fact: the field is tagged `yaml:"-"`, so yaml.v3 binds no `name` key to it and the strict
// decoder rejects a stray one rather than letting an in-file name become a second source of
// truth for an identity the filename already fixes.
func TestLoadChipRejectsInFileName(t *testing.T) {
	// A valid chip body — but written to h100.yaml with an in-file name that disagrees.
	path := write(t, "h100.yaml", `
name: not-h100
Provenance: vendor_spec
TFlopsPeak: 989.5
BwPeakTBs: 4.8
MemoryGiB: 141.0
IntraNodeBwGBps: 450
SMCount: 132
`)
	if _, err := LoadChip(path); err == nil {
		t.Fatal("an in-file name was accepted; the filename and the field could disagree")
	} else if !strings.Contains(err.Error(), "name") {
		t.Errorf("the error should name the offending field %q, got: %v", "name", err)
	}

	// An in-file name that AGREES with the filename is still rejected: the strict decoder
	// keys off the field, not the value, so a human author mirroring the stem into the body
	// — the most likely real-world mistake — fails loudly rather than being tolerated as a
	// harmless-looking special case that would re-admit the second source of truth.
	agreeing := write(t, "h100.yaml", `
name: h100
Provenance: vendor_spec
TFlopsPeak: 989.5
BwPeakTBs: 4.8
MemoryGiB: 141.0
IntraNodeBwGBps: 450
SMCount: 132
`)
	if _, err := LoadChip(agreeing); err == nil {
		t.Fatal("an in-file name matching the filename was accepted; identity must have one source")
	}
}

// TestLoadFabricRejectsInFileName is TestLoadChipRejectsInFileName for a Fabric: identity is
// the filename, Fabric.Name is tagged `yaml:"-"`, and an in-file `name` is a rejected
// unknown field (#27).
func TestLoadFabricRejectsInFileName(t *testing.T) {
	path := write(t, "ib-400g.yaml", `
name: not-ib-400g
Provenance: vendor_spec
InterNodeBwGBps: 50
RDMA: true
`)
	if _, err := LoadFabric(path); err == nil {
		t.Fatal("an in-file name was accepted; the filename and the field could disagree")
	} else if !strings.Contains(err.Error(), "name") {
		t.Errorf("the error should name the offending field %q, got: %v", "name", err)
	}
}

// TestLoadChipAndFabricStampNameFromFilename is the positive half of the identity contract:
// a nameless file (as every catalog chip and fabric is) loads, and Name is stamped from the
// path stem as its only source — exactly as TestLoadWorkload checks for a Shape. A stamped
// value then validates, since Validate requires a Name it can no longer get from the file.
func TestLoadChipAndFabricStampNameFromFilename(t *testing.T) {
	chipPath := write(t, "h100.yaml", `
Provenance: vendor_spec
TFlopsPeak: 989.5
BwPeakTBs: 4.8
MemoryGiB: 141.0
IntraNodeBwGBps: 450
SMCount: 132
`)
	c, err := LoadChip(chipPath)
	if err != nil {
		t.Fatalf("LoadChip: %v", err)
	}
	if c.Name != "h100" {
		t.Errorf("name = %q; it should come from the filename", c.Name)
	}
	if p := c.Validate(); !p.OK() {
		t.Errorf("a loader-named chip failed validation:\n%s", p.Error())
	}

	fabPath := write(t, "ib-400g.yaml", `
Provenance: vendor_spec
InterNodeBwGBps: 50
RDMA: true
`)
	f, err := LoadFabric(fabPath)
	if err != nil {
		t.Fatalf("LoadFabric: %v", err)
	}
	if f.Name != "ib-400g" {
		t.Errorf("name = %q; it should come from the filename", f.Name)
	}
	if p := f.Validate(); !p.OK() {
		t.Errorf("a loader-named fabric failed validation:\n%s", p.Error())
	}
}

// TestLoadDeploymentDCPKnobs: a deployment can state each of the engine's three
// decode-context-parallel knobs under the engine's own names, and an omitted one stays
// unstated rather than being read as a value. The prefill pool is the shape of a
// glm-5.3 canary from an internal llm-d deployment, which passes --dcp-comm-backend ag_rs.
func TestLoadDeploymentDCPKnobs(t *testing.T) {
	path := write(t, "d.yaml", `
kind: Deployment
name: glm-5.3-dcp
pools:
  - role: colocated
    nodes: 4
    parallel:
      tp: 1
      pp: 1
      dp: 4
      dp_local: 1
      pcp: 8
      dcp: 8
    engine:
      block_size: 64
      dcp_comm_backend: a2a
      dcp_q_replicate: true
      cp_kv_cache_interleave_size: 16
  - role: colocated
    nodes: 1
    parallel:
      tp: 8
      pp: 1
      dp: 1
      dcp: 8
    engine:
      dcp_q_replicate: false
`)
	d, err := LoadDeployment(path)
	if err != nil {
		t.Fatalf("LoadDeployment: %v", err)
	}
	// The engine block is held as written, keys and values the engine reads, for the release's
	// rules to open.
	var e map[string]any
	if err := d.Pools[0].Engine.Decode(&e); err != nil {
		t.Fatal(err)
	}
	if e["dcp_comm_backend"] != "a2a" || e["dcp_q_replicate"] != true || e["cp_kv_cache_interleave_size"] != 16 {
		t.Errorf("the engine block did not load as written: %v", e)
	}
	var e2 map[string]any
	if err := d.Pools[1].Engine.Decode(&e2); err != nil {
		t.Fatal(err)
	}
	if v, ok := e2["dcp_q_replicate"]; !ok || v != false {
		t.Errorf("a stated false must stay stated: %v", e2)
	}
	if _, ok := e2["dcp_comm_backend"]; ok {
		t.Errorf("an omitted knob must stay unstated: %v", e2)
	}
	if p := d.Validate(); !p.OK() {
		t.Fatalf("the deployment is well formed and should validate:\n%s", p.Error())
	}
}

func TestLoadDeployment(t *testing.T) {
	path := write(t, "d.yaml", `
kind: Deployment
name: granite-230b-h200-tp8
pools:
  - role: colocated
    nodes: 1
    parallel:
      tp: 8
      pp: 1
      dp: 1
    engine:
      cache_dtype: fp8
      block_size: 16
      max_num_batched_tokens: 32768
      gpu_memory_utilization: 0.9
      scheduling_policy: priority
      eplb:
        enable_eplb: true
        num_redundant_experts: 32
      dbo:
        enable_dbo: true
        dbo_decode_token_threshold: 32
`)
	d, err := LoadDeployment(path)
	if err != nil {
		t.Fatalf("LoadDeployment: %v", err)
	}
	if d.Name != "granite-230b-h200-tp8" {
		t.Errorf("name = %q", d.Name)
	}
	if len(d.Pools) != 1 || d.Pools[0].Parallel.TP != 8 {
		t.Fatalf("pools did not parse: %+v", d.Pools)
	}
	var e map[string]any
	if err := d.Pools[0].Engine.Decode(&e); err != nil {
		t.Fatal(err)
	}
	// The engine's own names, held as written, nested blocks included.
	if e["cache_dtype"] != "fp8" || e["scheduling_policy"] != "priority" || e["gpu_memory_utilization"] != 0.9 {
		t.Errorf("the engine block did not load as written: %v", e)
	}
	if eplb, ok := e["eplb"].(map[string]any); !ok || eplb["enable_eplb"] != true || eplb["num_redundant_experts"] != 32 {
		t.Errorf("eplb did not load as written: %v", e["eplb"])
	}
	// Validate the loaded deployment on its own terms; the composition layer requires a
	// scenario alongside a deployment (TestDeploymentWithoutScenarioIsRejected covers that).
	if p := d.Validate(); !p.OK() {
		t.Errorf("a loaded deployment failed field validation:\n%s", p.Error())
	}
}

// TestLoadRejectsUnknownFields is the property that makes strict decoding worth
// having. A misspelled key must fail loudly rather than leave a zero value. The engine
// block it misspells lives in a deployment now, so that is what is loaded.
func TestLoadRejectsUnknownFields(t *testing.T) {
	path := write(t, "d.yaml", `
kind: Deployment
name: typo
pools:
  - role: colocated
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1}
    engin:
      cache_dtype: fp8
`)
	_, err := LoadDeployment(path)
	if err == nil {
		t.Fatal("a misspelled field was accepted; the deployment would validate while " +
			"silently omitting the setting its author intended")
	}
	if !strings.Contains(err.Error(), "engin") {
		t.Errorf("the error should name the offending field, got: %v", err)
	}
}

// A misspelled key inside a pool's engine block loads, because the block is the engine's,
// held unopened; the release's rules refuse it when the bundle is validated, naming the key at
// its line in the file.
func TestAnEngineKeyIsTheReleasesToRefuse(t *testing.T) {
	path := write(t, "d.yaml", `
kind: Deployment
name: typo
pools:
  - role: colocated
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1}
    engine:
      blok_size: 16
`)
	d, err := LoadDeployment(path)
	if err != nil {
		t.Fatalf("the engine block is the engine's to check, but loading refused it: %v", err)
	}
	b := bundle()
	b.Deployment = d
	rep := Validate(b)
	if rep.Field.OK() || !strings.Contains(rep.Field.Error(), "line 9: field blok_size not found") {
		t.Errorf("the release's rules did not refuse the misspelled key at its line:\n%s", renderAll(rep))
	}
}

func TestLoadModelGraph(t *testing.T) {
	path := write(t, "g.yaml", `
kind: ModelGraph
name: granite-5-230b
modality: text_only
derived_from:
  format: hf_config_json
  path: config.json
  sha256: `+strings.Repeat("a", 64)+`
  deriver_version: 1
global:
  hidden_size: 3072
  vocab_size: 100352
  weight_dtype: fp8
layer_kinds:
  - id: attn_moe
    nodes:
      - {op: Elementwise, role: input_norm}
      - {op: Attention, kind: gqa, n_q: 48, n_kv: 8, d_h: 64}
      - {op: GroupedGEMM, role: experts, n: 1536, k: 3072, experts: 224, top_k: 8, shared_experts: 1}
      - {op: All2All, role: moe, emit: expert_parallel}
    edges: [[0, 1], [1, 2], [2, 3]]
stack:
  pattern: [attn_moe]
  repeat: 72
`)
	g, err := LoadModelGraph(path)
	if err != nil {
		t.Fatalf("LoadModelGraph: %v", err)
	}
	if g.Stack.Layers() != 72 {
		t.Errorf("layers = %d, want 72", g.Stack.Layers())
	}
	if g.Global.WeightDType != "fp8" {
		t.Errorf("weight_dtype = %q", g.Global.WeightDType)
	}
	n := g.LayerKinds[0].Nodes
	if len(n) != 4 {
		t.Fatalf("parsed %d nodes, want 4", len(n))
	}
	// The attention node's kind uses the tag "kind", not "attention_kind".
	if n[1].AttentionKind != "gqa" || n[1].NumKVHeads != 8 || n[1].HeadDim != 64 {
		t.Errorf("attention node did not parse: %+v", n[1])
	}
	if n[2].Experts != 224 || n[2].TopK != 8 || n[2].SharedExperts != 1 {
		t.Errorf("grouped-GEMM node did not parse: %+v", n[2])
	}
	if n[3].Emit != model.EmitExpertParallel {
		t.Errorf("collective condition = %q, want %q", n[3].Emit, model.EmitExpertParallel)
	}
	if len(g.LayerKinds[0].Edges) != 3 || g.LayerKinds[0].Edges[2] != [2]int{2, 3} {
		t.Errorf("edges did not parse: %v", g.LayerKinds[0].Edges)
	}
	if p := g.Validate(); !p.OK() {
		t.Errorf("a loaded graph failed validation:\n%s", p.Error())
	}
}

// TestLoadModelIdentity parses a model.yaml identity manifest end to end: the name and
// every source provenance field must arrive through the strict loader, and the loaded
// card must validate. A tag error at any depth would leave a zero value the type system
// cannot catch — the one failure this package exists to rule out.
func TestLoadModelIdentity(t *testing.T) {
	path := write(t, "model.yaml", `
name: deepseek-v4-pro
source:
  provider: huggingface
  repo: deepseek-ai/DeepSeek-V4-Pro
  revision: b5968e9190ef611bbf34a7229255be88a0e937c1
  retrieved: 2026-10-02
`)
	id, err := LoadModelIdentity(path)
	if err != nil {
		t.Fatalf("LoadModelIdentity: %v", err)
	}
	// Unlike a chip or a workload, the name is a FILE field here, not the filename stem.
	if id.Name != "deepseek-v4-pro" {
		t.Errorf("name = %q; it is a file field for a model card", id.Name)
	}
	if id.Source == nil {
		t.Fatal("source did not parse")
	}
	if id.Source.Provider != "huggingface" || id.Source.Repo != "deepseek-ai/DeepSeek-V4-Pro" {
		t.Errorf("source identity did not parse: %+v", id.Source)
	}
	if id.Source.Revision != "b5968e9190ef611bbf34a7229255be88a0e937c1" || id.Source.Retrieved != "2026-10-02" {
		t.Errorf("source provenance did not parse: %+v", id.Source)
	}
	if p := id.Validate(); !p.OK() {
		t.Errorf("a loaded identity failed validation:\n%s", p.Error())
	}
}

// TestLoadModelIdentityRejectsUnknownField pins strict decoding through the nested source
// block: a misspelled sub-key must fail loudly rather than leave a zero value, the same
// guarantee every other loader gives. Identity and Source are plain structs — neither
// implements a custom UnmarshalYAML, which is what would silently drop the decoder's
// KnownFields setting — so strict decoding recurses through the source block for free, and
// this pins it at both levels.
func TestLoadModelIdentityRejectsUnknownField(t *testing.T) {
	cases := []struct {
		name   string
		body   string
		badKey string
	}{
		{
			name: "unknown key at the top level",
			body: `
name: m
provdier: huggingface
source:
  provider: huggingface
  repo: r
  revision: v`,
			badKey: "provdier",
		},
		{
			name: "unknown key inside source",
			body: `
name: m
source:
  provider: huggingface
  repo: r
  revision: v
  revison: v2`,
			badKey: "revison",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := LoadModelIdentity(write(t, "model.yaml", tc.body))
			if err == nil {
				t.Fatalf("a misspelled %q was accepted; the card would validate while silently dropping it", tc.badKey)
			}
			if !strings.Contains(err.Error(), tc.badKey) {
				t.Errorf("the error should name the offending field %q, got: %v", tc.badKey, err)
			}
		})
	}
}

// catalogFixtures is a pinned, in-tree copy of blis-catalog's data, vendored once by hand
// under testdata/ in the catalog's own layout: models/<name>/{config.json,model.yaml,
// graph.yaml}, hardware/, networks/, devices/, workloads/. The source commit is recorded
// in the vendoring commit message, not in a marker file, so testdata/ stays pure catalog
// data. The committed-data tests resolve this snapshot rather than a BLIS_CATALOG env var
// or a live checkout of the catalog's main: a laptop env var skips (a skip reads as a
// pass), and a live main is a moving target that reddens unrelated builds. A pinned
// snapshot only changes when someone deliberately re-vendors it in a reviewable commit.
const catalogFixtures = "../catalog"

// TestLoadCatalogModels loads and VALIDATES the per-model artifacts in the vendored
// catalog snapshot. It walks each models/<name>/ directory rather than globbing the two
// YAML kinds independently: independent globs only notice a kind that has vanished from
// the tree entirely, so a single directory missing one of its files would slip past them,
// whereas walking the directory FAILS on it. The issue's layout gives every model entry
// three files — config.json, model.yaml and graph.yaml — so each must be present
// (config.json is checked for presence only; blis-schemas never parses it). model.yaml and
// graph.yaml are then loaded and validated with the same Validate() cmd/validate-catalog
// runs, and each must name itself what its directory is called. A MISSING fixture is a
// FAILURE, not a skip: a skip is indistinguishable from a pass and would let the check
// silently stop running. (The live validator treats a derived graph.yaml as optional,
// because a live catalog entry may lack one; this is a pinned snapshot vendored in the
// catalog's full layout, and re-vendoring is a deliberate, reviewable commit, so the
// stronger per-entry completeness check belongs here.)
func TestLoadCatalogModels(t *testing.T) {
	modelsDir := filepath.Join(catalogFixtures, "models")
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		t.Fatalf("reading %s: %v", modelsDir, err)
	}
	var names []string
	for _, e := range entries {
		// A model entry is a directory; skip hidden entries so a stray editor/VCS dir
		// (a .foo) is never mistaken for a model and reported as missing its files.
		if e.IsDir() && !strings.HasPrefix(e.Name(), ".") {
			names = append(names, e.Name())
		}
	}
	if len(names) == 0 {
		t.Fatalf("no model directories under %s: the vendored catalog snapshot is missing",
			modelsDir)
	}
	for _, name := range names {
		dir := filepath.Join(modelsDir, name)
		// config.json is part of the vendored layout but blis-schemas never parses it,
		// so only its presence is checked; a missing one is a missing fixture and fails.
		if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
			t.Errorf("%s: missing config.json: %v", name, err)
		}
		// model.yaml and graph.yaml must be present, load, and validate. The load is
		// gated on presence so a missing file reports once (the stat error) rather than
		// a second, redundant open error from the loader.
		modelPath := filepath.Join(dir, "model.yaml")
		if _, err := os.Stat(modelPath); err != nil {
			t.Errorf("%s: missing model.yaml: %v", name, err)
		} else if id, err := LoadModelIdentity(modelPath); err != nil {
			t.Errorf("LoadModelIdentity(%s/model.yaml): %v", name, err)
		} else {
			// The stated name must match the directory, and the source provenance must
			// be whole: those are what make a card an identity rather than a note.
			if id.Name != name {
				t.Errorf("%s/model.yaml: name %q does not match directory %q", name, id.Name, name)
			}
			if p := id.Validate(); !p.OK() {
				t.Errorf("%s/model.yaml failed validation:\n%s", name, p.Error())
			}
		}
		graphPath := filepath.Join(dir, "graph.yaml")
		if _, err := os.Stat(graphPath); err != nil {
			t.Errorf("%s: missing graph.yaml: %v", name, err)
		} else if g, err := LoadModelGraph(graphPath); err != nil {
			t.Errorf("LoadModelGraph(%s/graph.yaml): %v", name, err)
		} else {
			// A graph, like an identity, names itself what its directory is called.
			if g.Name != name {
				t.Errorf("%s/graph.yaml: name %q does not match directory %q", name, g.Name, name)
			}
			if p := g.Validate(); !p.OK() {
				t.Errorf("%s/graph.yaml failed validation:\n%s", name, p.Error())
			}
		}
	}
	t.Logf("validated model.yaml + graph.yaml and checked config.json presence for %d model(s) from the vendored catalog",
		len(names))
}

func TestLoadCoefficientSet(t *testing.T) {
	path := write(t, "c.yaml", `
kind: CoefficientSet
name: cost-model-primitives-h200
coefficients:
  - gemm_eps_max_bf16:
      value: 0.72
      units: dimensionless
      method: measured
      fitted: true
      scope:
        hardware: [h200]
      sources:
        - {kind: model, cite: "operator table", role: primary}
      ci95: [0.70, 0.74]
      validated: tpot
`)
	s, err := LoadCoefficientSet(path)
	if err != nil {
		t.Fatalf("LoadCoefficientSet: %v", err)
	}
	if len(s.Coefficients) != 1 {
		t.Fatalf("parsed %d entries, want 1", len(s.Coefficients))
	}
	e := s.Coefficients[0]
	if e.Value != 0.72 || e.Units != "dimensionless" || e.Method != "measured" {
		t.Errorf("entry did not parse: %+v", e)
	}
	if !e.Fitted {
		t.Error("fitted did not parse")
	}
	if len(e.Scope.Hardware) != 1 || e.Scope.Hardware[0] != "h200" {
		t.Errorf("scope did not parse: %+v", e.Scope)
	}
	if len(e.Sources) != 1 || e.Sources[0].Role != "primary" {
		t.Errorf("sources did not parse: %+v", e.Sources)
	}
	if e.CI95 == nil || e.CI95.Low != 0.70 {
		t.Errorf("ci95 did not parse: %+v", e.CI95)
	}
	// The name comes from the key the entry was nested under, not from a field.
	if e.Name != "gemm_eps_max_bf16" {
		t.Errorf("name = %q; it should come from the nesting key", e.Name)
	}
	// A scalar metric decodes to a one-element list, which is how the registry
	// writes the common case.
	if len(e.Validated) != 1 || e.Validated[0] != "tpot" {
		t.Errorf("validated = %v, want [tpot] from a scalar", e.Validated)
	}
	if p := s.Validate(); !p.OK() {
		t.Errorf("a loaded set failed validation:\n%s", p.Error())
	}
}

// TestLoadCatalogFiles loads and VALIDATES every top-level catalog artifact in the
// vendored snapshot: a chip, a fabric, a workload shape, and each storage tier. Like
// TestLoadCatalogModels it is the strongest check of the tags — real files, not
// hand-written fixtures — and it validates rather than merely loading (the gap #17 left),
// calling the same Validate() cmd/validate-catalog runs. A MISSING namespace is a FAILURE,
// not a skip, so the check cannot silently stop running against a snapshot that moved.
func TestLoadCatalogFiles(t *testing.T) {
	chips, err := filepath.Glob(filepath.Join(catalogFixtures, "hardware", "*.yaml"))
	if err != nil {
		t.Fatalf("globbing hardware/*.yaml: %v", err)
	}
	if len(chips) == 0 {
		t.Fatalf("no hardware/*.yaml under %s: the vendored catalog snapshot is missing",
			catalogFixtures)
	}
	for _, path := range chips {
		c, err := LoadChip(path)
		if err != nil {
			t.Errorf("LoadChip(%s): %v", filepath.Base(path), err)
			continue
		}
		if p := c.Validate(); !p.OK() {
			t.Errorf("%s failed validation:\n%s", filepath.Base(path), p.Error())
		}
	}

	fabrics, err := filepath.Glob(filepath.Join(catalogFixtures, "networks", "*.yaml"))
	if err != nil {
		t.Fatalf("globbing networks/*.yaml: %v", err)
	}
	if len(fabrics) == 0 {
		t.Fatalf("no networks/*.yaml under %s: the vendored catalog snapshot is missing",
			catalogFixtures)
	}
	for _, path := range fabrics {
		f, err := LoadFabric(path)
		if err != nil {
			t.Errorf("LoadFabric(%s): %v", filepath.Base(path), err)
			continue
		}
		if p := f.Validate(); !p.OK() {
			t.Errorf("%s failed validation:\n%s", filepath.Base(path), p.Error())
		}
	}

	workloads, err := filepath.Glob(filepath.Join(catalogFixtures, "workloads", "*.yaml"))
	if err != nil {
		t.Fatalf("globbing workloads/*.yaml: %v", err)
	}
	if len(workloads) == 0 {
		t.Fatalf("no workloads/*.yaml under %s: the catalog is missing",
			catalogFixtures)
	}
	for _, path := range workloads {
		w, err := LoadCatalogWorkload(path)
		if err != nil {
			t.Errorf("LoadCatalogWorkload(%s): %v", filepath.Base(path), err)
			continue
		}
		if p := w.Validate(); !p.OK() {
			t.Errorf("%s failed validation:\n%s", filepath.Base(path), p.Error())
		}
	}

	// devices/storage.yaml is one mapping file of tier name to device facts, loaded as a
	// list; it is a required part of the snapshot, so an absent or empty file fails.
	devicesPath := filepath.Join(catalogFixtures, "devices", "storage.yaml")
	devices, err := LoadStorageDevices(devicesPath)
	if err != nil {
		t.Fatalf("LoadStorageDevices(%s): %v", devicesPath, err)
	}
	if len(devices) == 0 {
		t.Fatalf("%s declares no storage tier: the vendored catalog snapshot is missing",
			devicesPath)
	}
	for _, d := range devices {
		if p := d.Validate(); !p.OK() {
			t.Errorf("devices/storage.yaml:%s failed validation:\n%s", d.Name, p.Error())
		}
	}

	t.Logf("validated %d chip(s), %d fabric(s), %d workload(s), %d storage tier(s) from the vendored catalog",
		len(chips), len(fabrics), len(workloads), len(devices))
}

// TestCoefficientOnlyBundleRunsNoRules is a canary for validate-registry. That command
// validates each set with (*coefficient.Set).Validate alone, which is complete only while
// a coefficient-only bundle runs no version-scoped rules: rules run solely against a
// Scenario, and rules.Input carries no coefficient. If a future rules pack ever runs for a
// coefficient-only bundle, this test fails on purpose — the signal to route
// validate-registry through blisschemas.Validate so it enforces those rules rather than
// silently passing sets they would reject.
func TestCoefficientOnlyBundleRunsNoRules(t *testing.T) {
	path := write(t, "c.yaml", `
kind: CoefficientSet
name: canary
coefficients:
  - x:
      value: 1
      units: dimensionless
      method: measured
      fitted: false
      scope:
        hardware: [h200]
`)
	s, err := LoadCoefficientSet(path)
	if err != nil {
		t.Fatalf("LoadCoefficientSet: %v", err)
	}
	rep := Validate(Bundle{Coefficients: []*coefficient.Set{s}})
	if rep.RulesApplied != "" {
		t.Errorf("a rules pack now runs for a coefficient-only bundle (RulesApplied=%q); "+
			"route validate-registry through blisschemas.Validate so it enforces coefficient rules",
			rep.RulesApplied)
	}
	if !rep.Rule.OK() {
		t.Errorf("rule-layer findings on a coefficient-only bundle: %s", rep.Rule.Error())
	}
}

// TestCoefficientEntryFormIsEnforced covers the nested wire form's failure modes. The
// registry writes each entry as a single-key map; a flat entry or a multi-key one
// would leave the name ambiguous or empty, and an entry without a name cannot be
// looked up by the kernel that needs it.
func TestCoefficientEntryFormIsEnforced(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{
			name: "flat entry with no nesting key",
			body: `
kind: CoefficientSet
name: s
coefficients:
  - value: 0.72
    units: dimensionless
`,
		},
		{
			name: "two keys, so the name is ambiguous",
			body: `
kind: CoefficientSet
name: s
coefficients:
  - first: {value: 1, units: dimensionless, method: assumed, fitted: false, scope: {tp: [8]}}
    second: {value: 2, units: dimensionless, method: assumed, fitted: false, scope: {tp: [8]}}
`,
		},
		{
			name: "entry is a scalar rather than a map",
			body: `
kind: CoefficientSet
name: s
coefficients:
  - just_a_string
`,
		},
		{
			name: "unknown field inside an entry",
			body: `
kind: CoefficientSet
name: s
coefficients:
  - a_coefficient:
      value: 1
      units: dimensionless
      method: assumed
      fitted: false
      scope: {tp: [8]}
      confidence: high
`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := LoadCoefficientSet(write(t, "c.yaml", tc.body)); err == nil {
				t.Fatal("expected a decode error, got none")
			}
		})
	}
}

// TestCatalogCommentKeysAreAcceptedButTyposAreNot: the catalog's provenance narrative
// lives in underscore-prefixed keys, which must pass, while a misspelled real field
// must still fail. Accepting everything would make strict decoding pointless.
func TestCatalogCommentKeysAreAcceptedButTyposAreNot(t *testing.T) {
	ok := write(t, "chip.yaml", `
Provenance: vendor_spec
TFlopsPeak: 989.5
TFlopsFP8: 1979.0
BwPeakTBs: 4.8
MemoryGiB: 141.0
IntraNodeBwGBps: 450
_comment: "narrative the catalog carries alongside every number"
_comment_interconnect: "a second one"
`)
	c, err := LoadChip(ok)
	if err != nil {
		t.Fatalf("comment keys were rejected: %v", err)
	}
	if c.Name != "chip" {
		t.Errorf("name = %q; it should come from the filename", c.Name)
	}
	if c.MemoryGiB != 141.0 {
		t.Errorf("MemoryGiB = %v, want 141", c.MemoryGiB)
	}

	bad := write(t, "chip.yaml", `
Provenance: vendor_spec
TFlopsPeak: 989.5
BwPeakTBs: 4.8
MemoryGiB: 141.0
IntraNodeBwGBps: 450
MemorGiB: 999
`)
	if _, err := LoadChip(bad); err == nil {
		t.Fatal("a misspelled field was accepted; the chip would validate with the " +
			"wrong capacity and nothing would report it")
	}
}

// TestStorageDeviceKeysCarryUnits is #18: the StorageDevice wire keys now carry their units
// (read_bandwidth_mb_s, write_bandwidth_mb_s, base_latency_us), because a unit stated only in
// the Go field name or a file comment is not a contract. The rename flows through
// RejectUnknownKeys by reflection, so the OLD unit-less spelling is now an unknown-field
// error rather than a silently-ignored field — the desired behaviour, since a file written
// to the old contract must fail loudly rather than decode to zeros. The new spelling loads.
func TestStorageDeviceKeysCarryUnits(t *testing.T) {
	// New spelling: loads, and the facts land on the right fields.
	okPath := write(t, "storage.yaml", `
nvme_gen4: {read_bandwidth_mb_s: 7.0e3, write_bandwidth_mb_s: 5.0e3, base_latency_us: 80.0}
`)
	devices, err := LoadStorageDevices(okPath)
	if err != nil {
		t.Fatalf("unit-suffixed keys were rejected: %v", err)
	}
	if len(devices) != 1 || devices[0].ReadBandwidthMBs != 7000 ||
		devices[0].WriteBandwidthMBs != 5000 || devices[0].BaseLatencyUs != 80 {
		t.Fatalf("facts did not land on the renamed keys: %+v", devices)
	}

	// Old spelling: each unit-less key is now an unknown field. A file written to the old
	// contract fails rather than loading a tier whose numbers all read as zero.
	oldPath := write(t, "storage.yaml", `
nvme_gen4: {read_bandwidth: 7.0e3, write_bandwidth: 5.0e3, base_latency: 80.0}
`)
	if _, err := LoadStorageDevices(oldPath); err == nil {
		t.Fatal("the old unit-less keys were accepted; a stale file would load as zeros")
	} else if !strings.Contains(err.Error(), "read_bandwidth") {
		t.Errorf("the error should name an offending old key, got: %v", err)
	}
}

// The connector registry fixture loads strictly and validates; a misspelled key fails to load.
func TestLoadConnectorRegistry(t *testing.T) {
	r, err := LoadConnectorRegistry("testdata/connectors/builtin-fixture.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if p := r.Validate(); !p.OK() {
		t.Fatalf("the fixture does not validate:\n%s", p.Error())
	}
	if !r.Serves("MultiConnector", "pd_transfer") || !r.Serves("SimpleCPUOffloadConnector", "offload") {
		t.Errorf("the fixture's roles did not load: %+v", r.Connectors)
	}
	bad := filepath.Join(t.TempDir(), "bad.yaml")
	if err := os.WriteFile(bad, []byte("kind: ConnectorRegistry\nname: x\nconnectors: {}\nentries: {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadConnectorRegistry(bad); err == nil {
		t.Error("a document with an unknown key loaded")
	}
}

// TestLoadResultReadsJSONAndYAMLStrictly: the simulator writes a Result as JSON by default
// and YAML on request, so the one loader reads both, and refuses an unknown key in either.
func TestLoadResultReadsJSONAndYAMLStrictly(t *testing.T) {
	j := filepath.Join("..", "docs", "contracts", "examples", "result.json")
	r, err := LoadResult(j)
	if err != nil {
		t.Fatalf("LoadResult(%s): %v", j, err)
	}
	if r.Kind != "Result" || r.Frame.Generator.Load.Concurrency != 64 || r.Metrics["ttft_p99_ms"].Value != 431.5 {
		t.Errorf("the JSON example did not load field for field: %+v", r)
	}

	dir := t.TempDir()
	y := filepath.Join(dir, "r.yaml")
	if err := os.WriteFile(y, []byte("kind: Result\nschema_version: 2\nname: x\nmetrics: {ttft_p99_ms: {value: 1, samples: 3}}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if r, err := LoadResult(y); err != nil || r.Metrics["ttft_p99_ms"].Value != 1 {
		t.Errorf("LoadResult did not read YAML: %v %+v", err, r)
	}

	for name, body := range map[string]string{
		"json": `{"kind": "Result", "metricz": {}}`,
		"yaml": "kind: Result\nmetricz: {}\n",
	} {
		p := filepath.Join(dir, "unknown."+name)
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadResult(p); err == nil || !strings.Contains(err.Error(), "metricz") {
			t.Errorf("%s: an unknown key must be refused by name, got %v", name, err)
		}
	}
}

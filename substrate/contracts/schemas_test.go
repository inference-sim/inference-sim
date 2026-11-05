package contracts

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

func bundle() Bundle {
	return Bundle{
		Scenario: &scenario.Scenario{
			Kind: "Scenario", Name: "granite-230b-h200-tp8",
			Model:         "granite-5-230b",
			Coefficients:  []string{"cost-model-primitives-h200"},
			EngineVersion: "1.0.0",
			Cluster:       scenario.Cluster{Hardware: "h200", Nodes: 1, GPUsPerNode: 8},
		},
		Deployment: &deployment.Deployment{
			Kind: "Deployment", Name: "granite-230b-h200-tp8",
			Pools: []deployment.Pool{{Role: deployment.RoleColocated, Nodes: 1,
				Parallel: deployment.Parallelism{TP: 8, PP: 1, DP: 1},
				Engine:   engineBlock("{backend: fast, block_size: 16, gpu_memory_utilization: 0.9}")}},
		},
		Model: &model.Graph{
			Kind: "ModelGraph", Name: "granite-5-230b",
			DerivedFrom: model.Derivation{Format: "hf_config_json",
				Path: "config.json", SHA256: strings.Repeat("b", 64),
				DeriverVersion: 1},
			Global: model.GlobalShape{HiddenSize: 3072, VocabSize: 100352,
				WeightDType: model.DTypeFP8},
			LayerKinds: []model.LayerKind{{ID: "attn_moe", Nodes: []model.Node{
				{Op: model.OpAttention, AttentionKind: model.AttentionGQA,
					NumQHeads: 48, NumKVHeads: 8, HeadDim: 64},
				{Op: model.OpGroupedGEMM, N: 1536, K: 3072, Experts: 224, TopK: 8},
			}}},
			Stack: model.Stack{Pattern: []string{"attn_moe"}, Repeat: 72},
		},
		Chip: &hardware.Chip{Name: "h200",
			Provenance: vocab.ProvenanceVendorSpec, BF16Peak: 989.5,
			FP8Peak: 1979.0, MemoryBandwidthTBs: 4.8, MemoryGiB: 141,
			IntraNodeBwGBps: 450, SMCount: 132, GPUsPerNode: 8},
		Coefficients: []*coefficient.Set{{
			Kind: "CoefficientSet", Name: "cost-model-primitives-h200",
			Coefficients: []coefficient.Entry{{
				Name: "gemm_eps_max_bf16", Value: 0.72,
				Units: vocab.UnitDimensionless, Method: vocab.MethodMeasured,
				Fitted: true,
				Scope:  coefficient.Scope{Hardware: []string{"h200"}},
				Sources: []coefficient.Source{{Kind: vocab.SourceModel,
					Cite: "operator table", Role: vocab.RolePrimary}},
			}},
		}},
		Rules:         testRulesFor,
		RulesVersions: testRulesVersions,
	}
}

func TestValidBundlePasses(t *testing.T) {
	rep := Validate(bundle())
	if !rep.OK() {
		t.Fatalf("a valid bundle was rejected:\n%s", renderAll(rep))
	}
	if rep.RulesApplied != "1.0.0" {
		t.Errorf("RulesApplied = %q, want 1.0.0", rep.RulesApplied)
	}
}

// TestUnknownEngineVersionIsReported is the property that keeps silence from
// looking like a pass: a version with no pack must say so.
func TestUnknownEngineVersionIsReported(t *testing.T) {
	b := bundle()
	b.Scenario.EngineVersion = "99.0.0"
	rep := Validate(b)
	if rep.RulesApplied != "" {
		t.Errorf("RulesApplied = %q, want empty", rep.RulesApplied)
	}
	if len(rep.Rule.All()) == 0 {
		t.Fatal("an unknown engine version produced no finding, which is indistinguishable from a checked pass")
	}
	if !strings.Contains(rep.Rule.Error(), "no rules pack") {
		t.Errorf("finding should say no pack applied; got:\n%s", rep.Rule.Error())
	}
	// It is a warning, not an error: the documents are well formed.
	if !rep.OK() {
		t.Error("an unknown version should warn rather than fail")
	}
}

// TestRulesDoNotRunOnMalformedDocuments pins the layering: a rule reading a
// malformed document yields findings that are artifacts of the malformation.
func TestRulesDoNotRunOnMalformedDocuments(t *testing.T) {
	b := bundle()
	b.Scenario.Kind = "NotAScenario"
	b.Deployment.Pools[0].Engine = engineBlock("{backend: telepathy}") // would fire a rule
	rep := Validate(b)
	if rep.Field.OK() {
		t.Fatal("expected a field problem")
	}
	if len(rep.Rule.All()) != 0 {
		t.Errorf("rules ran over a malformed document:\n%s", rep.Rule.Error())
	}
	if rep.RulesApplied != "" {
		t.Errorf("RulesApplied = %q, want empty", rep.RulesApplied)
	}
}

// TestLayersAreDistinguishable: a caller must be able to tell a malformed document
// from one that a particular engine would refuse.
func TestLayersAreDistinguishable(t *testing.T) {
	b := bundle()
	b.Deployment.Pools[0].Engine = engineBlock("{backend: telepathy}")
	rep := Validate(b)
	if !rep.Field.OK() {
		t.Fatalf("field layer should pass; got:\n%s", rep.Field.Error())
	}
	if rep.Rule.OK() {
		t.Fatal("rule layer should fail on an unknown backend")
	}
	for _, p := range rep.Rule.Errors() {
		if p.Rule == "" {
			t.Errorf("rule finding carries no rule name: %s", p)
		}
	}
}

// TestEngineRulesAreRequired: a scenario names an engine release, and this schema holds the
// pools' engine blocks unopened, so without the release's rules nothing checks them. A bundle
// with a scenario and no rules is refused, by a named rule, rather than passed unchecked.
func TestEngineRulesAreRequired(t *testing.T) {
	b := bundle()
	b.Rules, b.RulesVersions = nil, nil
	b.Deployment.Pools[0].Engine = engineBlock("{block_size: -1, backend: telepathy}")
	rep := Validate(b)
	if rep.OK() {
		t.Fatal("a bundle with no engine rules passed")
	}
	if rep.RulesApplied != "" {
		t.Errorf("RulesApplied = %q, want empty", rep.RulesApplied)
	}
	found := false
	for _, p := range rep.Rule.Errors() {
		if p.Rule == "engine-rules-given" && strings.Contains(p.Message, `"1.0.0"`) {
			found = true
		}
	}
	if !found {
		t.Errorf("want an engine-rules-given error naming the release; got:\n%s", renderAll(rep))
	}
	// A bundle with no scenario names no release and needs no rules.
	if rep := Validate(Bundle{Chip: b.Chip}); !rep.OK() {
		t.Errorf("a chip-only bundle needs no engine rules:\n%s", renderAll(rep))
	}
}

// TestEngineBlockIsCheckedByTheRules: the engine block's field checks are the release's, run
// in the field layer at the deployment's paths, and a block the release's rules cannot decode
// is a field problem too.
func TestEngineBlockIsCheckedByTheRules(t *testing.T) {
	b := bundle()
	b.Deployment.Pools[0].Engine = engineBlock("{block_size: -1, gpu_memory_utilization: 1.5}")
	rep := Validate(b)
	var paths []string
	for _, p := range rep.Field.Errors() {
		paths = append(paths, p.Path)
	}
	for _, want := range []string{"deployment.pools[0].engine.block_size", "deployment.pools[0].engine.gpu_memory_utilization"} {
		found := false
		for _, p := range paths {
			found = found || p == want
		}
		if !found {
			t.Errorf("no field problem at %s; got %v", want, paths)
		}
	}
	// An offload's connector options are the release's to check too, at the offload's paths.
	b.Deployment.Pools[0].Engine = engineBlock("{backend: fast}")
	b.Scenario.Cluster.Storage = []string{"cpu_dram"}
	b.Deployment.Offload = &deployment.Offload{Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: 1}},
		ConnectorOptions: engineBlock("{blocks_per_chunk: -1}")}
	rep = Validate(b)
	found := false
	for _, p := range rep.Field.Errors() {
		found = found || p.Path == "deployment.offload.connector_options.blocks_per_chunk"
	}
	if !found {
		t.Errorf("no field problem at deployment.offload.connector_options.blocks_per_chunk:\n%s", renderAll(rep))
	}
	b.Deployment.Offload, b.Scenario.Cluster.Storage = nil, nil

	b.Deployment.Pools[0].Engine = engineBlock("{blok_size: 16}")
	rep = Validate(b)
	if rep.Field.OK() || !strings.Contains(rep.Field.Error(), "blok_size") {
		t.Errorf("an unknown engine key passed the field layer:\n%s", renderAll(rep))
	}
	if len(rep.Rule.All()) != 0 {
		t.Errorf("rules ran over an engine block that does not decode:\n%s", rep.Rule.Error())
	}
}

// TestDeploymentWithoutScenarioIsRejected guards the composition layer against a bundle
// that would otherwise silently under-validate. A Deployment with no Scenario skips the
// cluster-fit check and every version-scoped engine rule (both need the scenario), so a
// deployment that does not fit its hardware would come back green. The layer reports a
// field problem instead. The mirror holds: a Scenario with no Deployment is a complete
// problem on its own and still validates.
func TestDeploymentWithoutScenarioIsRejected(t *testing.T) {
	b := bundle()

	rep := Validate(Bundle{Deployment: b.Deployment})
	if rep.Field.OK() {
		t.Fatal("a deployment with no scenario should be a field problem, not a pass")
	}
	if rep.RulesApplied != "" {
		t.Errorf("RulesApplied = %q, want empty when no scenario is present", rep.RulesApplied)
	}

	// The mirror: a scenario with no deployment is a complete problem and still validates.
	if rep := Validate(Bundle{Scenario: b.Scenario, Rules: testRulesFor}); !rep.OK() {
		t.Fatalf("a scenario with no deployment should pass:\n%s", renderAll(rep))
	}
}

func TestPartialBundlesValidate(t *testing.T) {
	// A catalog contributor checks one chip, with no scenario at all.
	rep := Validate(Bundle{Chip: bundle().Chip})
	if !rep.OK() {
		t.Fatalf("a chip-only bundle was rejected:\n%s", renderAll(rep))
	}
	if rep.RulesApplied != "" {
		t.Error("no scenario means no rules should apply")
	}
}

// TestOffloadTierMustBeInClusterStorage pins the inventory coupling at the layer that
// owns it: the storage classes live on the Scenario's cluster, so only the composition
// layer — which sees both documents — can check that a deployment's offload tiers draw
// from the declared inventory.
func TestOffloadTierMustBeInClusterStorage(t *testing.T) {
	b := bundle()
	b.Scenario.Cluster.Storage = []string{"cpu_dram", "nvme_gen4"}

	// A tier drawn from the declared inventory passes.
	b.Deployment.Offload = &deployment.Offload{
		Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: 1}}}
	if rep := Validate(b); !rep.OK() {
		t.Fatalf("a tier drawn from the cluster inventory should pass:\n%s", renderAll(rep))
	}

	// A tier naming a class the cluster does not list is a field problem.
	b.Deployment.Offload = &deployment.Offload{
		Tiers: []deployment.Tier{{Device: "optane", Bytes: 1}}}
	if rep := Validate(b); rep.Field.OK() {
		t.Fatal("an offload tier outside the cluster storage inventory should be a field problem")
	}
}

// TestEngineMustFitItsPool is the composition-level statement of the rank-count rule:
// no layer used to relate a layout's rank count to the GPUs it was placed on, so the
// issue's example — tp 8 with pcp 4 on one 8-GPU node, needing 32 devices — passed both
// the deployment validator and the cluster-coupling checks. It goes through Validate
// because the claim was about the whole pipeline, not one function.
func TestEngineMustFitItsPool(t *testing.T) {
	// The ordinary bundle is tp 8 on one 8-GPU node: eight ranks on eight devices.
	if rep := Validate(bundle()); !rep.OK() {
		t.Fatalf("tp 8 on an 8-GPU node should fit:\n%s", renderAll(rep))
	}

	b := bundle()
	b.Deployment.Pools[0].Parallel = deployment.Parallelism{TP: 8, PP: 1, DP: 1, PCP: 4}
	rep := Validate(b)
	if rep.Field.OK() {
		t.Fatal("tp 8 with pcp 4 needs 32 GPUs and must not validate on an 8-GPU node")
	}
	found := false
	for _, p := range rep.Field.Errors() {
		if p.Path == "deployment.pools[0].parallel" && strings.Contains(p.Message, "needs 32 GPUs") {
			found = true
		}
	}
	if !found {
		t.Errorf("expected a problem at deployment.pools[0].parallel naming 32 GPUs, got:\n%s",
			renderAll(rep))
	}

	// The same layout fits once the pool owns enough nodes. The rule is about room, not
	// about the layout being unusual: pcp 4 is fine when there is somewhere to put it.
	// A multi-node cluster must also name its fabric, which is unrelated to what is
	// being tested but part of a well-formed scenario.
	b.Scenario.Cluster.Nodes = 4
	b.Scenario.Cluster.Fabric = "ib-400g"
	b.Deployment.Pools[0].Nodes = 4
	if rep := Validate(b); !rep.Field.OK() {
		t.Fatalf("the same layout on four 8-GPU nodes should fit:\n%s", renderAll(rep))
	}
}

func renderAll(r Report) string {
	var b strings.Builder
	for _, p := range r.Problems() {
		b.WriteString(p.String())
		b.WriteString("\n")
	}
	return b.String()
}

// fakeRules is an EngineRules that records the version it was asked for and reports one rule
// finding, so a test can tell the caller's rules ran. It takes every engine block as valid.
type fakeRules struct {
	rules.EngineRules
	version string
}

func (f fakeRules) EngineVersion() string { return f.version }
func (f fakeRules) Check(rules.Input) *rules.Problems {
	p := &validate.Problems{}
	p.RuleErrorf("caller-rule", "the caller's rules ran for %s", f.version)
	return p
}
func (f fakeRules) CheckEngineBlock(string, deployment.EngineBlock) *rules.Problems { return nil }
func (f fakeRules) DraftTokens(deployment.EngineBlock) int                          { return 0 }
func (f fakeRules) CheckConnectorOptions(string, deployment.EngineBlock) *rules.Problems {
	return nil
}

// The caller's rules are what run: asked for the scenario's release, their findings reported,
// and a release they do not know reported rather than passed in silence.
func TestValidate_RunsTheCallersRules(t *testing.T) {
	var asked string
	b := bundle()
	b.Rules = func(v string) rules.EngineRules { asked = v; return fakeRules{version: v} }
	rep := Validate(b)
	if asked != "1.0.0" || rep.RulesApplied != "1.0.0" {
		t.Fatalf("asked %q, applied %q; want the scenario's 1.0.0", asked, rep.RulesApplied)
	}
	if !strings.Contains(rep.Rule.Error(), "the caller's rules ran for 1.0.0") {
		t.Errorf("the caller's rules did not run:\n%s", rep.Rule.Error())
	}

	b.Rules = func(string) rules.EngineRules { return nil }
	b.RulesVersions = func() []string { return []string{"1.1.0"} }
	rep = Validate(b)
	if rep.RulesApplied != "" || !strings.Contains(rep.Rule.Error(), "no rules pack") ||
		!strings.Contains(rep.Rule.Error(), "[1.1.0]") {
		t.Errorf("a version the caller's rules do not know: applied %q, findings:\n%s", rep.RulesApplied, rep.Rule.Error())
	}
}

// nilCheckRules says "no findings" the usual Go way, with a nil Problems.
type nilCheckRules struct{ fakeRules }

func (nilCheckRules) Check(rules.Input) *rules.Problems { return nil }

// ptrRules is a rules type of a pointer type, as an engine model's own rules may be.
type ptrRules struct{ fakeRules }

func (r *ptrRules) EngineVersion() string             { return r.version }
func (r *ptrRules) Check(rules.Input) *rules.Problems { return &validate.Problems{} }

// The caller's rules are held to the contract: a report whose rule layer is never nil, an
// unknown release named as unknown whatever the rules' concrete type, and rules answering for a
// release other than the scenario's refused rather than applied.
func TestValidate_HoldsTheCallersRulesToTheContract(t *testing.T) {
	b := bundle()
	b.Rules = func(v string) rules.EngineRules { return nilCheckRules{fakeRules{version: v}} }
	rep := Validate(b)
	if rep.Rule == nil || !rep.OK() || rep.RulesApplied != "1.0.0" || len(rep.Problems()) != 0 {
		t.Errorf("rules whose Check returns nil: rule layer %v, applied %q; want an empty layer and 1.0.0", rep.Rule, rep.RulesApplied)
	}

	var typedNil *ptrRules
	b.Rules, b.RulesVersions = func(string) rules.EngineRules { return typedNil }, nil
	rep = Validate(b)
	if rep.RulesApplied != "" || !strings.Contains(rep.Rule.Error(), "no rules pack") || !rep.OK() {
		t.Errorf("a typed nil rules value must read as unknown: applied %q, findings:\n%s", rep.RulesApplied, rep.Rule.Error())
	}
	// With no RulesVersions the finding names no registered versions, rather than an empty list
	// that would read as a rules source holding none.
	if strings.Contains(rep.Rule.Error(), "Registered versions") {
		t.Errorf("no RulesVersions: the finding must not list registered versions:\n%s", rep.Rule.Error())
	}

	for _, answered := range []string{"1.1.0", ""} {
		b.Rules = func(string) rules.EngineRules { return fakeRules{version: answered} }
		rep = Validate(b)
		if rep.OK() || rep.RulesApplied != "" {
			t.Errorf("rules answering for %q to a 1.0.0 scenario: OK %v, applied %q; want a refusal and nothing applied",
				answered, rep.OK(), rep.RulesApplied)
		}
		for _, frag := range []string{"engine-version-matches", `"1.0.0"`, `"` + answered + `"`} {
			if !strings.Contains(rep.Rule.Error(), frag) {
				t.Errorf("rules answering for %q: the refusal must name %s; findings:\n%s", answered, frag, rep.Rule.Error())
			}
		}
		if strings.Contains(rep.Rule.Error(), "caller's rules ran") {
			t.Errorf("rules answering for %q were run against a 1.0.0 scenario", answered)
		}
	}
}

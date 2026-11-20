package contracts

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/check"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/coefficient"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/hardware"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/scenario"
	"github.com/inference-sim/inference-sim/substrate/contracts/vocab"
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
				{Op: model.OpAttention, HoldsEngineKV: model.Bool(true), AttentionKind: model.AttentionGQA,
					NumQHeads: 48, NumKVHeads: 8, HeadDim: 64},
				{Op: model.OpGroupedGEMM, Shared: model.Bool(false), N: 1536, K: 3072, Experts: 224, TopK: 8},
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
		Engine:         testRulesFor,
		EngineVersions: testEngineVersions,
	}
}

// ruleFindings are the report's findings that carry a rule name: the bundle-level notes about
// the engine (not given, unknown, mismatched, input unchecked), as against the field checks.
func ruleFindings(rep Report) []validate.Problem {
	var out []validate.Problem
	for _, p := range rep.Field.All() {
		if p.Rule != "" {
			out = append(out, p)
		}
	}
	return out
}

func ruleText(rep Report) string {
	var b strings.Builder
	for _, p := range ruleFindings(rep) {
		b.WriteString(p.String() + "\n")
	}
	return b.String()
}

func TestValidBundlePasses(t *testing.T) {
	rep := Validate(bundle())
	if !rep.OK() {
		t.Fatalf("a valid bundle was rejected:\n%s", renderAll(rep))
	}
	if len(rep.Problems()) != 0 {
		t.Errorf("a valid bundle carries findings:\n%s", renderAll(rep))
	}
}

// TestUnknownEngineVersionIsReported is the property that keeps silence from
// looking like a pass: a version with no engine must say so.
func TestUnknownEngineVersionIsReported(t *testing.T) {
	b := bundle()
	b.Scenario.EngineVersion = "99.0.0"
	rep := Validate(b)
	if len(ruleFindings(rep)) == 0 {
		t.Fatal("an unknown engine version produced no finding, which is indistinguishable from a checked pass")
	}
	if !strings.Contains(ruleText(rep), "no engine for engine version") {
		t.Errorf("finding should say no engine applied; got:\n%s", ruleText(rep))
	}
	// It is a warning, not an error: the documents are well formed.
	if !rep.OK() {
		t.Error("an unknown version should warn rather than fail")
	}
}

// TestEngineChecksDoNotRunOnMalformedDocuments pins the layering: the bundle-level engine
// findings are not reported over a malformed document, where they would be artifacts of it.
func TestEngineChecksDoNotRunOnMalformedDocuments(t *testing.T) {
	b := bundle()
	b.Scenario.Kind = "NotAScenario"
	b.Engine, b.EngineVersions = nil, nil // would fire engine-given on a well-formed bundle
	rep := Validate(b)
	if rep.Field.OK() {
		t.Fatal("expected a field problem")
	}
	if len(ruleFindings(rep)) != 0 {
		t.Errorf("engine findings reported over a malformed document:\n%s", ruleText(rep))
	}
}

// TestEngineRefusalIsAFieldProblem: an engine refuses a block as a field problem at the
// deployment's path, so a caller sees it as the document being wrong, not as a note.
func TestEngineRefusalIsAFieldProblem(t *testing.T) {
	b := bundle()
	b.Deployment.Pools[0].Engine = engineBlock("{backend: telepathy}")
	rep := Validate(b)
	if rep.OK() {
		t.Fatal("an unknown backend should fail validation")
	}
	found := false
	for _, p := range rep.Field.Errors() {
		found = found || (p.Path == "deployment.pools[0].engine.backend" && strings.Contains(p.Message, "telepathy"))
	}
	if !found {
		t.Errorf("want a field error at deployment.pools[0].engine.backend naming the backend; got:\n%s", renderAll(rep))
	}
}

// TestEngineIsRequired: a scenario names an engine release, and this schema holds the
// pools' engine blocks unopened, so without the release's checker nothing checks them. A bundle
// with a scenario and no engine is refused, by a named finding, rather than passed unchecked.
func TestEngineIsRequired(t *testing.T) {
	b := bundle()
	b.Engine, b.EngineVersions = nil, nil
	b.Deployment.Pools[0].Engine = engineBlock("{block_size: -1, backend: telepathy}")
	rep := Validate(b)
	if rep.OK() {
		t.Fatal("a bundle with no engine passed")
	}
	found := false
	for _, p := range rep.Field.Errors() {
		if p.Rule == "engine-given" && strings.Contains(p.Message, `"1.0.0"`) {
			found = true
		}
	}
	if !found {
		t.Errorf("want an engine-given error naming the release; got:\n%s", renderAll(rep))
	}
	// A bundle with no scenario names no release and needs no engine.
	if rep := Validate(Bundle{Chip: b.Chip}); !rep.OK() {
		t.Errorf("a chip-only bundle needs no engine:\n%s", renderAll(rep))
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
	if len(ruleFindings(rep)) != 0 {
		t.Errorf("engine findings reported over an engine block that does not decode:\n%s", ruleText(rep))
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

	// The mirror: a scenario with no deployment is a complete problem and still validates.
	if rep := Validate(Bundle{Scenario: b.Scenario, Engine: testRulesFor}); !rep.OK() {
		t.Fatalf("a scenario with no deployment should pass:\n%s", renderAll(rep))
	}
}

func TestPartialBundlesValidate(t *testing.T) {
	// A catalog contributor checks one chip, with no scenario at all.
	rep := Validate(Bundle{Chip: bundle().Chip})
	if !rep.OK() {
		t.Fatalf("a chip-only bundle was rejected:\n%s", renderAll(rep))
	}
	if len(rep.Problems()) != 0 {
		t.Errorf("no scenario means no engine findings:\n%s", renderAll(rep))
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

// fakeEngine is an EngineChecker that records the version it was asked for and takes every
// engine block as valid.
type fakeEngine struct{ version string }

func (f fakeEngine) EngineVersion() string { return f.version }
func (f fakeEngine) CheckEngineBlock(string, deployment.EngineBlock) *validate.Problems {
	return nil
}
func (f fakeEngine) DraftTokens(deployment.EngineBlock) int { return 0 }
func (f fakeEngine) CheckConnectorOptions(string, deployment.EngineBlock) *validate.Problems {
	return nil
}

// refusingEngine refuses every engine block, so a test can tell the caller's checker ran.
type refusingEngine struct{ fakeEngine }

func (f refusingEngine) CheckEngineBlock(at string, _ deployment.EngineBlock) *validate.Problems {
	p := &validate.Problems{}
	p.Field(at+".engine", "the caller's engine ran for %s", f.version)
	return p
}

// The caller's engine is what runs: asked for the scenario's release, its findings reported,
// and a release it does not know reported rather than passed in silence.
func TestValidate_RunsTheCallersEngine(t *testing.T) {
	var asked string
	b := bundle()
	b.Engine = func(v string) EngineChecker { asked = v; return refusingEngine{fakeEngine{version: v}} }
	rep := Validate(b)
	if asked != "1.0.0" {
		t.Fatalf("asked %q; want the scenario's 1.0.0", asked)
	}
	if !strings.Contains(rep.Field.Error(), "the caller's engine ran for 1.0.0") {
		t.Errorf("the caller's engine did not run:\n%s", rep.Field.Error())
	}

	b.Engine = func(string) EngineChecker { return nil }
	b.EngineVersions = func() []string { return []string{"1.1.0"} }
	rep = Validate(b)
	if !strings.Contains(ruleText(rep), "no engine for engine version") || !strings.Contains(ruleText(rep), "[1.1.0]") {
		t.Errorf("a version the caller's engine does not know; findings:\n%s", ruleText(rep))
	}
}

// ptrEngine is a checker of a pointer type, as an engine module's own may be.
type ptrEngine struct{ fakeEngine }

func (r *ptrEngine) EngineVersion() string { return r.version }

// The caller's engine is held to the contract: an unknown release named as unknown whatever
// the checker's concrete type, and a checker answering for a release other than the
// scenario's refused rather than applied.
func TestValidate_HoldsTheCallersEngineToTheContract(t *testing.T) {
	b := bundle()
	var typedNil *ptrEngine
	b.Engine, b.EngineVersions = func(string) EngineChecker { return typedNil }, nil
	rep := Validate(b)
	if !strings.Contains(ruleText(rep), "no engine for engine version") || !rep.OK() {
		t.Errorf("a typed nil checker must read as unknown; findings:\n%s", ruleText(rep))
	}
	// With no EngineVersions the finding names no registered versions, rather than an empty list
	// that would read as an engine source holding none.
	if strings.Contains(ruleText(rep), "Registered versions") {
		t.Errorf("no EngineVersions: the finding must not list registered versions:\n%s", ruleText(rep))
	}

	for _, answered := range []string{"1.1.0", ""} {
		b.Engine = func(string) EngineChecker { return refusingEngine{fakeEngine{version: answered}} }
		rep = Validate(b)
		if rep.OK() {
			t.Errorf("a checker answering for %q to a 1.0.0 scenario: OK; want a refusal", answered)
		}
		for _, frag := range []string{"engine-version-matches", `"1.0.0"`, `"` + answered + `"`} {
			if !strings.Contains(ruleText(rep), frag) {
				t.Errorf("a checker answering for %q: the refusal must name %s; findings:\n%s", answered, frag, ruleText(rep))
			}
		}
		if strings.Contains(rep.Field.Error(), "caller's engine ran") {
			t.Errorf("a checker answering for %q was run against a 1.0.0 scenario", answered)
		}
	}
}

// A bundle carrying the chip resolves the cluster's packaging against it (SUB-3): the
// deployment's fit is checked at the resolved node size, and a scenario contradicting the
// entry is refused at scenario.cluster. Without the chip, a scenario that leaves its node
// size to the catalog cannot have its deployment fitted, and is refused (SIM-9 pairs).
func TestValidateResolvesPackagingFromTheChip(t *testing.T) {
	nvl := func(gpn, gpr int) Bundle {
		b := bundle()
		b.Scenario.Cluster = scenario.Cluster{Hardware: "gb200-nvl72", Nodes: 2, Fabric: "ib-400g",
			GPUsPerNode: gpn, GPUsPerRack: gpr}
		b.Deployment.Pools[0].Nodes = 2
		b.Chip.Name, b.Chip.GPUsPerNode, b.Chip.GPUsPerRack, b.Chip.IntraRackBwGBps = "gb200-nvl72", 4, 72, 900
		b.Coefficients[0].Coefficients[0].Scope.Hardware = []string{"gb200-nvl72"}
		return b
	}
	at := func(rep Report, path string) bool {
		for _, p := range rep.Field.Errors() {
			if p.Path == path {
				return true
			}
		}
		return false
	}
	for _, c := range []struct {
		name    string
		b       Bundle
		refused string
	}{
		{"NVL72, packaging omitted: tp 8 over two 4-GPU trays fits", nvl(0, 0), ""},
		{"NVL72, packaging restated", nvl(4, 72), ""},
		{"NVL72 stated at 8 per node", nvl(8, 0), "scenario.cluster.gpus_per_node"},
		{"NVL72 stated at 12 per rack", nvl(4, 12), "scenario.cluster.gpus_per_rack"},
	} {
		t.Run(c.name, func(t *testing.T) {
			rep := Validate(c.b)
			if c.refused == "" {
				if !rep.Field.OK() {
					t.Fatalf("refused:\n%s", rep.Field.Error())
				}
				return
			}
			if !at(rep, c.refused) {
				t.Fatalf("not refused at %s:\n%s", c.refused, rep.Field.Error())
			}
		})
	}

	// The resolved node size is the one the fit uses: at 4 GPUs per tray, tp 8 on one tray
	// does not fit, though it would on a stated 8-GPU node.
	one := nvl(0, 0)
	one.Scenario.Cluster.Nodes, one.Scenario.Cluster.Fabric, one.Deployment.Pools[0].Nodes = 1, "", 1
	if rep := Validate(one); !at(rep, "deployment.pools[0].parallel") {
		t.Errorf("tp 8 on one 4-GPU tray is not refused:\n%s", rep.Field.Error())
	}

	noChip := nvl(0, 0)
	noChip.Chip = nil
	if rep := Validate(noChip); !at(rep, "scenario.cluster.gpus_per_node") {
		t.Errorf("a deployment over a node size left to a chip the bundle lacks is not refused:\n%s", rep.Field.Error())
	}
	noChip.Scenario.Cluster.GPUsPerNode = 4
	if rep := Validate(noChip); at(rep, "scenario.cluster.gpus_per_node") {
		t.Errorf("a stated node size without the chip is refused:\n%s", rep.Field.Error())
	}
}

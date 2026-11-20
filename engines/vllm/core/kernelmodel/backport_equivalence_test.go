// Original to blis-vllm.

package kernelmodel

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	latencykernel "github.com/inference-sim/inference-sim/substrate/kernel"

	"github.com/inference-sim/inference-sim/engines/vllm/core/engineconfig"
	vllmrules "github.com/inference-sim/inference-sim/engines/vllm/core/rulepack"
	"github.com/inference-sim/inference-sim/engines/vllm/core/rulepack/pack"
	"github.com/inference-sim/inference-sim/engines/vllm/core/rulepack/v0_33"
	sim "github.com/inference-sim/inference-sim/engines/vllm/core/runconfig"
	"github.com/inference-sim/inference-sim/engines/vllm/internal/upstream"
)

// The equivalence check of a model backport (DC1; FID-14, T-M5): a catalog model that does not use
// a backport has the same plan and the same trace with the backport as without it. A backport
// carries model code for the architectures it lists (upstream.Backport.Architectures); for every
// other model it is not allowed to move a number.
//
// A model is checked through the committed vLLM scenario files (fixtureDirs): the catalog holds a model, not a deployment, so the fixture states the chip,
// parallelism and engine block the plan and trace are read at. One fixture stands for a model: the
// first, in name order, that opens with the build's rules; a model none of whose fixtures opens is
// compared by the refusal it gets. A catalog model with no committed scenario is lowered at a
// synthetic one (synthTemplate), and the test reports which models were reached which way.
//
// The plan is the pool's lowered plan; the trace is what the engine reads from the model beside
// it, the resolved engine block and the KV sizing, and the step times the cost model prices for a
// fixed sequence of batches (prefill chunks, a mixed batch and decodes at three batch widths).
// The cost model prices from the plan, so a rules change that moves a step time moves the plan
// too; the step times are kept as a guard against a pricing path that reads the rules directly,
// and no rule a synthetic patch can change moves them alone.

// observation is everything the check compares for one model with one set of rules.
type observation struct {
	OpenErr     string
	Lowered     plan.Lowered
	Engine      engineconfig.Engine
	EngineErr   string
	Settings    Settings
	SettingsErr string
	Trace       []int64
}

// stateTrace samples a settings' per-sequence state functions, which a comparison of the values
// cannot reach, at a fixed grid, and clears them.
func stateTrace(s *Settings) []int64 {
	var out []int64
	for _, n := range []int64{1, 17, 1024, 8192, 65536} {
		if s.SequenceState.Gate != nil {
			out = append(out, s.SequenceState.Gate(n))
		}
		if s.SequenceState.Hold != nil {
			out = append(out, s.SequenceState.Hold(n, 1), s.SequenceState.Hold(n, 512))
		}
	}
	s.SequenceState = sim.KVSequenceState{}
	return out
}

// traceBatches is the fixed batch sequence the trace prices.
func traceBatches() [][]ReqShape {
	var out [][]ReqShape
	out = append(out, []ReqShape{{Scheduled: 2048, PromptLen: 2048}})
	out = append(out, []ReqShape{{Scheduled: 512, Computed: 1024, PromptLen: 4096}, {Scheduled: 1, Computed: 700, PromptLen: 512}})
	for _, n := range []int{1, 8, 64} {
		b := make([]ReqShape, n)
		for i := range b {
			b[i] = ReqShape{Scheduled: 1, Computed: 1024 + 16*i, PromptLen: 1024}
		}
		out = append(out, b)
	}
	return out
}

func observe(scenarioDir, name string, rules *pack.Pack, base Repos) observation {
	r := base
	r.Scenarios = scenarioDir
	r.Rules = latencykernel.RulesFrom(func(v string) *pack.Pack {
		if v == rules.Version {
			return rules
		}
		return nil
	})
	m, err := Open(name, r)
	if err != nil {
		return observation{OpenErr: err.Error()}
	}
	o := observation{Lowered: m.Lowered()}
	if e, err := m.Engine(); err != nil {
		o.EngineErr = err.Error()
	} else {
		o.Engine = e
	}
	if s, err := m.Settings(); err != nil {
		o.SettingsErr = err.Error()
	} else {
		o.Trace = append(o.Trace, stateTrace(&s)...)
		o.Settings = s
	}
	for _, b := range traceBatches() {
		o.Trace = append(o.Trace, m.StepTimeOfShapes(b))
	}
	return o
}

// describeDifference names where two observations differ, for the failure message.
func describeDifference(a, b observation) string {
	av, bv := reflect.ValueOf(a), reflect.ValueOf(b)
	var parts []string
	for i := 0; i < av.NumField(); i++ {
		if reflect.DeepEqual(av.Field(i).Interface(), bv.Field(i).Interface()) {
			continue
		}
		ja, _ := json.Marshal(av.Field(i).Interface())
		jb, _ := json.Marshal(bv.Field(i).Interface())
		parts = append(parts, fmt.Sprintf("%s: without=%.160s with=%.160s", av.Type().Field(i).Name, ja, jb))
	}
	return strings.Join(parts, "; ")
}

// fixtureModel is one vLLM fixture and the catalog model and architecture it states.
type fixtureModel struct {
	dir, name, model, architecture string
	synthetic                      bool
}

// fixtureDirs are the directories of committed vLLM scenario files the check reads, relative to
// this package, in the order that picks a model's representative.
var fixtureDirs = []string{
	filepath.Join("..", "..", "kerneltests", "testdata", "direct"),
	filepath.Join("..", "..", "kerneltests", "testdata", "aisimulate"),
	filepath.Join("..", "..", "kerneltests", "testdata"),
	filepath.Join("..", "..", "vllmengine", "testdata"),
	filepath.Join("..", "..", "vllmengine", "testdata", "kvstartup"),
	filepath.Join("..", "..", "testdata", "scenarios"),
}

// vllmFixtures lists the vLLM fixtures of the committed corpora, by model, in name order.
func vllmFixtures(t *testing.T) map[string][]fixtureModel {
	t.Helper()
	data := repos().Data
	out := map[string][]fixtureModel{}
	for _, dir := range fixtureDirs {
		files, err := filepath.Glob(filepath.Join(dir, "*.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		sort.Strings(files)
		for _, f := range files {
			raw, err := os.ReadFile(f)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(raw), "kind: vllm") {
				continue
			}
			sc, _, err := contracts.LoadBundle(f)
			if err != nil || sc.Engine == nil || sc.Engine.Kind != "vllm" {
				continue
			}
			g, err := data.Model(sc.Model)
			if err != nil {
				continue
			}
			out[sc.Model] = append(out[sc.Model], fixtureModel{dir: dir, name: filepath.Base(f), model: sc.Model,
				architecture: g.Global.Architecture})
		}
	}
	synthesizeFixtures(t, out)
	return out
}

// synthTemplate is a minimal colocated deployment, with the engine block unstated beyond what
// the plan needs. It stands for no real deployment: it gives a catalog model that has no committed
// fixture a chip and a width to be lowered at, and the check compares only the model with and
// without the backport at that point.
const synthTemplate = `kind: Scenario
name: %[1]s
engine: {kind: vllm}
model: %[2]s
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory]
cluster: {hardware: %[3]s, nodes: 1, gpus_per_node: 8}
---
kind: Deployment
name: %[1]s
pools:
  - role: colocated
    nodes: 1
    parallel: {tp: %[4]d, pp: 1, dp: 1, enable_expert_parallel: false}
    engine:
      max_num_batched_tokens: 8192
      max_num_seqs: 256
      max_model_len: 4096
      gpu_memory_utilization: 0.92
      compilation_config: {cudagraph_mode: PIECEWISE}
`

// synthPoints are the chips and widths tried, in order, for a model with no committed fixture.
var synthPoints = []struct {
	hardware string
	tp       int
}{{"h200", 8}, {"b200", 8}, {"gb200", 8}, {"h200", 4}, {"h100", 8}, {"h100", 2}, {"h200", 1}}

// synthesizeFixtures adds, for each catalog model with no committed vLLM scenario, the scenarios at
// synthPoints, written to a temporary directory.
func synthesizeFixtures(t *testing.T, out map[string][]fixtureModel) {
	t.Helper()
	data := repos().Data
	dir := t.TempDir()
	for _, m := range data.Models() {
		if _, ok := out[m]; ok {
			continue
		}
		g, err := data.Model(m)
		if err != nil {
			continue
		}
		for _, pt := range synthPoints {
			name := fmt.Sprintf("synth-%s-%s-tp%d", m, pt.hardware, pt.tp)
			if err := os.WriteFile(filepath.Join(dir, name+".yaml"),
				[]byte(fmt.Sprintf(synthTemplate, name, m, pt.hardware, pt.tp)), 0o644); err != nil {
				t.Fatal(err)
			}
			out[m] = append(out[m], fixtureModel{dir: dir, name: name + ".yaml", model: m,
				architecture: g.Global.Architecture, synthetic: true})
		}
	}
}

// representative is the fixture that stands for a model: the first that opens with base rules.
func representative(fs []fixtureModel, base *pack.Pack, r Repos) fixtureModel {
	for _, f := range fs {
		if observe(f.dir, f.name, base, r).OpenErr == "" {
			return f
		}
	}
	return fs[0]
}

// checkBackportEquivalence runs the check for every backport in bs, whose patches are given: for
// each catalog model with a fixture that does not use the backport, the plan and the trace
// with all of bs applied equal those with all of bs but that one. It returns how many
// model-and-backport pairs it compared and the differences it found.
func checkBackportEquivalence(t *testing.T, bs []upstream.Backport, patches map[string]vllmrules.Patch) (compared int, differences []string) {
	t.Helper()
	fixtures := vllmFixtures(t)
	if len(fixtures) == 0 {
		t.Fatal("no vLLM fixture found: the check would compare nothing")
	}
	r := repos()
	withAll, err := vllmrules.WithBackports(v0_33.Pack(), bs, patches)
	if err != nil {
		t.Fatal(err)
	}
	var models []string
	for m := range fixtures {
		models = append(models, m)
	}
	sort.Strings(models)
	for _, b := range bs {
		var rest []upstream.Backport
		restPatches := map[string]vllmrules.Patch{}
		for _, o := range bs {
			if o.Commit != b.Commit {
				rest = append(rest, o)
				restPatches[o.Commit] = patches[o.Commit]
			}
		}
		without, err := vllmrules.WithBackports(v0_33.Pack(), rest, restPatches)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range models {
			f := representative(fixtures[m], without, r)
			if b.Uses(f.architecture) {
				continue
			}
			compared++
			a := observe(f.dir, f.name, without, r)
			w := observe(f.dir, f.name, withAll, r)
			if !reflect.DeepEqual(a, w) {
				differences = append(differences, fmt.Sprintf("backport %s, model %s (%s, %s): %s",
					b.Commit[:12], m, f.architecture, f.name, describeDifference(a, w)))
			}
		}
	}
	return compared, differences
}

// The build's own backports, with the patches the rules make for them: none at the pin, and the
// check holds for any that are added.
func TestRecordedBackportsDoNotMoveAModelThatDoesNotUseThem(t *testing.T) {
	bs := upstream.Backports()
	compared, diffs := checkBackportEquivalence(t, bs, vllmrules.Patches())
	for _, d := range diffs {
		t.Error(d)
	}
	if len(bs) == 0 && compared != 0 {
		t.Errorf("no backport is recorded, yet %d comparisons ran", compared)
	}
	if len(bs) > 0 && compared == 0 {
		t.Error("backports are recorded, yet no model was compared")
	}
	logCoverage(t)
}

// logCoverage reports how each catalog model was reached: through a committed scenario, through
// a synthetic one, or by the refusal it gets at every point tried (compared by its message only).
func logCoverage(t *testing.T) {
	t.Helper()
	fixtures := vllmFixtures(t)
	base := v0_33.Pack()
	var committed, synthetic, refused []string
	for _, m := range repos().Data.Models() {
		fs := fixtures[m]
		if len(fs) == 0 {
			t.Errorf("catalog model %s has no scenario to be checked at", m)
			continue
		}
		f := representative(fs, base, repos())
		switch {
		case observe(f.dir, f.name, base, repos()).OpenErr != "":
			refused = append(refused, m)
		case f.synthetic:
			synthetic = append(synthetic, m)
		default:
			committed = append(committed, m)
		}
	}
	t.Logf("catalog models checked: %d at a committed scenario, %d at a synthetic one, %d by refusal only (%s)",
		len(committed), len(synthetic), len(refused), strings.Join(refused, ", "))
}

// An observation fills every dimension the check compares, and is repeatable: a dimension that
// stayed empty would compare equal whatever a backport did.
func TestObservationFillsEveryDimensionAndRepeats(t *testing.T) {
	var f fixtureModel
	for _, fs := range vllmFixtures(t) {
		if !fs[0].synthetic {
			f = representative(fs, v0_33.Pack(), repos())
			break
		}
	}
	if f.name == "" {
		t.Fatal("no committed fixture")
	}
	a := observe(f.dir, f.name, v0_33.Pack(), repos())
	if a.OpenErr != "" || a.EngineErr != "" || a.SettingsErr != "" {
		t.Fatalf("%s did not open cleanly: %+v", f.name, a)
	}
	v := reflect.ValueOf(a)
	for i := 0; i < v.NumField(); i++ {
		name := v.Type().Field(i).Name
		if strings.HasSuffix(name, "Err") {
			continue
		}
		if v.Field(i).IsZero() {
			t.Errorf("observation field %s is empty for %s", name, f.name)
		}
	}
	n := len(traceBatches())
	if len(a.Trace) < n {
		t.Fatalf("trace has %d entries, fewer than the %d batches it prices", len(a.Trace), n)
	}
	for i, d := range a.Trace[len(a.Trace)-n:] {
		if d <= 0 {
			t.Errorf("batch %d is priced at %d, not a step time", i, d)
		}
	}
	if b := observe(f.dir, f.name, v0_33.Pack(), repos()); !reflect.DeepEqual(a, b) {
		t.Errorf("two observations of %s differ: %s", f.name, describeDifference(a, b))
	}
}

// syntheticCommit names the synthetic backport; it is not a vLLM commit.
const syntheticCommit = "0123456789abcdef0123456789abcdef01234567"

func syntheticBackport(architecture string) upstream.Backport {
	return upstream.Backport{
		Commit:        syntheticCommit,
		Subject:       "synthetic: a fixture for the equivalence check, not a vLLM commit",
		Architectures: []string{architecture},
		Paths:         []string{"vllm/model_executor/models/synthetic.py"},
	}
}

// A backport that makes only the entries of its own architecture leaves every other model alone;
// one that also changes what another architecture reads is caught. The cases are a synthetic
// fixture, not a vLLM commit (the pin has no real backport): the first patch is what a faithful
// backport looks like, the others are the ways one leaks.
func TestEquivalenceCheckSeparatesAFaithfulBackportFromALeakyOne(t *testing.T) {
	fixtures := vllmFixtures(t)
	// The architecture the synthetic backport is "for": one a catalog model with a fixture has,
	// and which no rule keys on.
	arch := ""
	var models []string
	for m := range fixtures {
		models = append(models, m)
	}
	sort.Strings(models)
	for _, m := range models {
		if a := fixtures[m][0].architecture; a != "" && a != glmArch {
			arch = a
			break
		}
	}
	if arch == "" {
		t.Skip("no fixture model with a free architecture")
	}
	faithful := func(p *pack.Pack) {
		p.CacheDTypeResolutions[arch] = pack.CacheDTypeResolutions{
			{CapabilityFamily: 10, Requested: []string{"auto"}, ResolvesTo: "fp8_e4m3"},
		}
	}
	tests := []struct {
		name      string
		patch     vllmrules.Patch
		wantLeaks bool
	}{
		{"adds entries for its own architecture only", faithful, false},
		{"also changes another architecture's cache dtype", func(p *pack.Pack) {
			faithful(p)
			p.CacheDTypeResolutions[glmArch] = nil
		}, true},
		{"also narrows a rule every model reads", func(p *pack.Pack) {
			faithful(p)
			delete(p.CustomAllReduceWorldSizes, 4)
			delete(p.CustomAllReduceWorldSizes, 8)
		}, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b := syntheticBackport(arch)
			compared, diffs := checkBackportEquivalence(t, []upstream.Backport{b},
				map[string]vllmrules.Patch{b.Commit: tc.patch})
			if compared == 0 {
				t.Fatal("the check compared no model")
			}
			if tc.wantLeaks && len(diffs) == 0 {
				t.Errorf("a leaking backport went unnoticed across %d models", compared)
			}
			if !tc.wantLeaks && len(diffs) != 0 {
				t.Errorf("a faithful backport was reported as leaking: %v", diffs)
			}
		})
	}
}

// glmArch is the architecture whose unstated cache dtype the pack resolves on a Blackwell chip.
const glmArch = v0_33.ArchGlmMoeDsa

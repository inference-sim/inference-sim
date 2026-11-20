package prereg_test

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/ledger"
	"github.com/inference-sim/inference-sim/substrate/ledger/inferencex"
	"github.com/inference-sim/inference-sim/substrate/ledger/sglangnightly"
	"github.com/inference-sim/inference-sim/substrate/ledger/vllmperf"
	"github.com/inference-sim/inference-sim/substrate/prereg"
)

var committed = prereg.Env{
	LedgerDir:       "../catalog/ledger",
	CoefficientsDir: "../registry/coefficients",
	EngineCodeIDs:   "../registry/inputs/engine_code_ids.json",
}

const committedGates = "../registry/gates"

func mustLoad(t *testing.T, dir string) *prereg.Registry {
	t.Helper()
	r, err := prereg.Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	return r
}

// The committed pre-registrations are well formed, fresh against the ledger they pin, cover
// every coefficient set and gated source, and no fit script reads a source they do not cover.
func TestCommittedPreregistrationsVerify(t *testing.T) {
	r := mustLoad(t, committedGates)
	if err := r.Verify(committed); err != nil {
		t.Fatalf("Verify: %v", err)
	}
	if err := r.ScanScripts(".."); err != nil {
		t.Fatalf("ScanScripts: %v", err)
	}
}

// Both engines are covered, and the shared versus engine-specific split of design section 3.2
// holds: one kernel-keyed record that guards both engines, one engine-path record per engine.
func TestFamiliesCoverBothEngines(t *testing.T) {
	r := mustLoad(t, committedGates)
	kernel, path := 0, map[string]int{}
	for _, rec := range r.Records {
		switch rec.CoefficientClass {
		case prereg.ClassKernel:
			kernel++
			if !reflect.DeepEqual(sorted(rec.TargetEngines), []string{"sglang", "vllm"}) {
				t.Errorf("%s: kernel-keyed target_engines %v, want both engines", rec.ID, rec.TargetEngines)
			}
		case prereg.ClassEnginePath:
			path[rec.TargetEngines[0]]++
		}
	}
	if kernel != 1 || path["vllm"] != 1 || path["sglang"] != 1 || len(path) != 2 {
		t.Errorf("kernel-keyed records %d, engine-path per engine %v; want 1 and one each", kernel, path)
	}
}

// The defaults are the section 1.5 defaults of the requirements; this is the one place a change
// to a threshold must be made twice, so it is deliberate.
func TestDefaultsAreTheSection15Defaults(t *testing.T) {
	d := mustLoad(t, committedGates).Defaults
	checks := []struct {
		name      string
		got, want any
	}{
		{"baseline", d.Baseline, "previous-release-registry"},
		{"min_points", d.MinPoints, 10},
		{"margin.error", d.Margin.Error, 0.0},
		{"margin.distance", d.Margin.Distance, 0.0},
		{"slices", d.Slices, []string{"engine", "chip", "workload"}},
		{"heldout.evaluations", d.Heldout.Evaluations, 1},
		{"heldout.after", d.Heldout.After, "selection"},
		{"revision.retries", d.Revision.Retries, 1},
		{"revision.then", d.Revision.Then, "revert"},
		{"test.method", d.Test.Method, "paired-bootstrap-bca"},
		{"test.resamples", d.Test.Resamples, 10000},
		{"test.alpha", d.Test.Alpha, 0.05},
		{"test.correction", d.Test.Correction, "holm"},
		{"test.resample_over", d.Test.ResampleOver, "role-key"},
		{"roles_drawn_per", d.RolesDrawnPer, "role-key"},
		{"training_roles", d.TrainingRoles, []ledger.Role{ledger.RoleFit, ledger.RoleSelect, ledger.RoleValidate}},
		{"fit_reads", d.FitReads, []ledger.Role{ledger.RoleFit}},
		{"select_reads", d.SelectReads, []ledger.Role{ledger.RoleSelect, ledger.RoleValidate}},
		{"forbidden_sources", d.ForbiddenSources, []string{"configiq"}},
	}
	for _, c := range checks {
		if !reflect.DeepEqual(c.got, c.want) {
			t.Errorf("%s: %v, want %v", c.name, c.got, c.want)
		}
	}
}

// The role-key fields the defaults record are the fields the ingest tools draw the role over,
// and the gated sources are the ledger's training-only sources.
func TestDefaultsAgreeWithTheLedger(t *testing.T) {
	d := mustLoad(t, committedGates).Defaults
	want := map[string][]string{
		"inferencex":     inferencex.RoleKeyFields,
		"vllm-ci-perf":   append([]string{"source"}, vllmperf.IdentityFields...),
		"sglang-nightly": append([]string{"source"}, sglangnightly.RoleKeyFields...),
		"sglang-traces":  {"source", "file"},
	}
	if !reflect.DeepEqual(d.RoleKeyFields, want) {
		t.Errorf("role_key_fields %v, want %v", d.RoleKeyFields, want)
	}
	if !reflect.DeepEqual(sorted(d.GatedSources), sorted(ledger.TrainingOnlySources)) {
		t.Errorf("gated_sources %v, want the ledger's training-only sources %v", d.GatedSources, ledger.TrainingOnlySources)
	}
}

func sorted(xs []string) []string {
	out := append([]string(nil), xs...)
	sort.Strings(out)
	return out
}

// fixture copies the committed gates and the files Verify reads into a temp directory, so a
// test can break one thing and see Verify name it.
type fixture struct {
	t     *testing.T
	gates string
	env   prereg.Env
}

func copyFile(t *testing.T, from, to string) {
	t.Helper()
	b, err := os.ReadFile(from)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(to), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(to, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	dir := t.TempDir()
	f := &fixture{t: t, gates: filepath.Join(dir, "gates"), env: prereg.Env{
		LedgerDir:       filepath.Join(dir, "ledger"),
		CoefficientsDir: filepath.Join(dir, "coefficients"),
		EngineCodeIDs:   filepath.Join(dir, "engine_code_ids.json"),
	}}
	copyFile(t, committedGates+"/defaults.yaml", f.gates+"/defaults.yaml")
	recs, _ := filepath.Glob(committedGates + "/preregistrations/*.yaml")
	for _, p := range recs {
		copyFile(t, p, filepath.Join(f.gates, "preregistrations", filepath.Base(p)))
	}
	for _, n := range []string{"ledger.yaml", "sources.yaml"} {
		copyFile(t, filepath.Join(committed.LedgerDir, n), filepath.Join(f.env.LedgerDir, n))
	}
	sets, _ := filepath.Glob(committed.CoefficientsDir + "/*.yaml")
	for _, p := range sets {
		if err := os.MkdirAll(f.env.CoefficientsDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(f.env.CoefficientsDir, filepath.Base(p)), nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	copyFile(t, committed.EngineCodeIDs, f.env.EngineCodeIDs)
	return f
}

func (f *fixture) edit(path, old, new string) {
	f.t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		f.t.Fatal(err)
	}
	if !strings.Contains(string(b), old) {
		f.t.Fatalf("%s has no %q to replace", path, old)
	}
	if err := os.WriteFile(path, []byte(strings.Replace(string(b), old, new, 1)), 0o644); err != nil {
		f.t.Fatal(err)
	}
}

func (f *fixture) rec(id string) string {
	return filepath.Join(f.gates, "preregistrations", id+".yaml")
}

func (f *fixture) verify() error {
	f.t.Helper()
	r, err := prereg.Load(f.gates)
	if err != nil {
		return err
	}
	return r.Verify(f.env)
}

func TestUnmodifiedFixtureVerifies(t *testing.T) {
	if err := newFixture(t).verify(); err != nil {
		t.Fatalf("the unmodified copy was refused: %v", err)
	}
}

// Each known-bad change is refused, and the error names what is wrong.
func TestVerifyRefuses(t *testing.T) {
	kernel, vllm, sglang := "refit-kernel-keyed", "refit-engine-path-vllm", "refit-engine-path-sglang"
	for name, c := range map[string]struct {
		mut  func(f *fixture)
		want string
	}{
		"the held-out share is read": {func(f *fixture) {
			f.edit(f.gates+"/defaults.yaml", "fit_reads: [fit]", "fit_reads: [fit, held]")
		}, "not a training role"},
		"fit and selection read the same share": {func(f *fixture) {
			f.edit(f.gates+"/defaults.yaml", "select_reads: [select, validate]", "select_reads: [fit, select, validate]")
		}, "share"},
		"configiq is a source": {func(f *fixture) {
			f.edit(f.rec(vllm), "sources: [inferencex, vllm-ci-perf]", "sources: [configiq, inferencex, vllm-ci-perf]")
		}, "anchor-only"},
		"a gated source is in no record": {func(f *fixture) {
			f.edit(f.rec(kernel), ", vllm-ci-perf]", "]")
			f.edit(f.rec(vllm), "sources: [inferencex, vllm-ci-perf]", "sources: [inferencex]")
		}, "gated and no pre-registration covers it"},
		"a threshold departs from the default": {func(f *fixture) {
			f.edit(f.rec(kernel), "{criterion: error, margin: 0}", "{criterion: error, margin: 0.05}")
		}, "section 1.5 default"},
		"min_points departs from the default": {func(f *fixture) {
			f.edit(f.rec(kernel), "min_points: 10", "min_points: 3")
		}, "section 1.5 default"},
		"an override with no merged pull request": {func(f *fixture) {
			f.edit(f.rec(kernel), "{criterion: error, margin: 0}", "{criterion: error, margin: 0.05}")
			f.edit(f.rec(kernel), "measured_sha256: \"\"\n", "measured_sha256: \"\"\nthreshold_override:\n  set_by: someone\n  min_points: 10\n  margin: {error: 0.05, distance: 0}\n")
		}, "merged pull request"},
		"a defaults value changes after registration": {func(f *fixture) {
			f.edit(f.gates+"/defaults.yaml", "alpha: 0.05", "alpha: 0.1")
		}, "defaults changed since the record was registered"},
		"a source's cut-points change after registration": {func(f *fixture) {
			f.edit(f.env.LedgerDir+"/ledger.yaml", "- {role: fit, from: 0, to: 69}\n      - {role: select, from: 70, to: 84}\n      - {role: validate, from: 85, to: 99}\n  sglang-nightly",
				"- {role: fit, from: 0, to: 59}\n      - {role: select, from: 60, to: 84}\n      - {role: validate, from: 85, to: 99}\n  sglang-nightly")
		}, "cut-points changed"},
		"a source's pinned revision changes after registration": {func(f *fixture) {
			f.edit(f.env.LedgerDir+"/sources.yaml", "rows: 7795", "rows: 7796")
		}, "pinned revision changed"},
		"the seed changes after registration": {func(f *fixture) {
			f.edit(f.rec(vllm), "seed_sha256: \"0c74", "seed_sha256: \"1c74")
		}, "seed changed"},
		"a pin names a source the record does not list": {func(f *fixture) {
			f.edit(f.rec(vllm), "  cut_points:\n", "  cut_points:\n    aisimulate: \"00\"\n")
		}, "pins a source the record does not list"},
		"a coefficient set is in no record": {func(f *fixture) {
			if err := os.WriteFile(f.env.CoefficientsDir+"/cost-model-new.yaml", nil, 0o644); err != nil {
				f.t.Fatal(err)
			}
		}, "cost-model-new is in no pre-registration"},
		"a record names a set that does not exist": {func(f *fixture) {
			f.edit(f.rec(kernel), "cost-model-recurrent]", "cost-model-recurrent, cost-model-nope]")
		}, "cost-model-nope does not exist"},
		"a kernel-keyed set is in two records": {func(f *fixture) {
			f.edit(f.rec(vllm), "coefficient_sets: [cost-model-host-overheads]", "coefficient_sets: [cost-model-host-overheads, cost-model-attention]")
		}, "only an engine-path set is in more than one record"},
		"a kernel-keyed record guards one engine": {func(f *fixture) {
			f.edit(f.rec(kernel), "target_engines: [vllm, sglang]", "target_engines: [vllm]")
		}, "every engine is a lane it must not harm"},
		"an engine-path record names another engine's code": {func(f *fixture) {
			f.edit(f.rec(vllm), "host: 1226cbf6", "host: 2226cbf6")
		}, "engine_code_ids.json states"},
		"an engine-path record has no engine code": {func(f *fixture) {
			f.edit(f.rec(sglang), "engine_code_ids:\n  sglang:\n    host: 6771194bbf4376753254d570aa150303edd8b438a726a5e55b59ba56f234e096\n    graphs: e0368a9b8797515769f8c5dc0910a3713afaeb4e63fa9dd4440311424eba1074\n", "")
		}, "engine_code_ids: exactly"},
		"a target lane is for no target engine": {func(f *fixture) {
			f.edit(f.rec(vllm), "  - {engine: vllm}", "  - {engine: sglang}")
		}, "no lane for engine vllm"},
		"a lane field the gate does not take": {func(f *fixture) {
			f.edit(f.rec(vllm), "  - {engine: vllm}", "  - {engine: vllm, workload: chat}")
		}, "lane field"},
		"a source with no select share": {func(f *fixture) {
			f.edit(f.env.LedgerDir+"/ledger.yaml", "  aisimulate-fpm:\n    cut_points:\n      - {role: fit, from: 0, to: 69}\n      - {role: select, from: 70, to: 84}\n      - {role: validate, from: 85, to: 99}",
				"  aisimulate-fpm:\n    cut_points:\n      - {role: fit, from: 0, to: 69}\n      - {role: validate, from: 70, to: 99}")
		}, "has no select share"},
		"the file name is not the id": {func(f *fixture) {
			if err := os.Rename(f.rec(vllm), f.rec("renamed")); err != nil {
				f.t.Fatal(err)
			}
		}, "file name says"},
		"an unknown field": {func(f *fixture) {
			f.edit(f.rec(vllm), "min_points: 10\n", "min_points: 10\nsurprise: 1\n")
		}, "surprise"},
		"an invented partition version": {func(f *fixture) {
			f.edit(f.rec(vllm), "partition_version: 1\nhypothesis", "partition_version: 2\nhypothesis")
		}, "the ledger is at 1"},
		"a measured hash that is not a hash": {func(f *fixture) {
			f.edit(f.rec(vllm), "measured_sha256: \"\"", "measured_sha256: \"abc\"")
		}, "measured_sha256"},
		"an uncovered set is also covered": {func(f *fixture) {
			f.edit(f.gates+"/defaults.yaml", "  - set: cost-model-fabric\n", "  - set: cost-model-attention\n")
		}, "cover it"},
	} {
		t.Run(name, func(t *testing.T) {
			f := newFixture(t)
			c.mut(f)
			err := f.verify()
			if err == nil {
				t.Fatal("Verify accepted it")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error %q does not mention %q", err, c.want)
			}
		})
	}
}

// A threshold set by a maintainer in a merged pull request is accepted when it is complete, and
// the records then hold to it.
func TestOverrideIsAcceptedWhenAMaintainerSetsIt(t *testing.T) {
	f := newFixture(t)
	f.edit(f.rec("refit-kernel-keyed"), "{criterion: error, margin: 0}", "{criterion: error, margin: 0.05}")
	f.edit(f.rec("refit-kernel-keyed"), "measured_sha256: \"\"\n", "measured_sha256: \"\"\nthreshold_override:\n  pull_request: https://github.com/inference-sim/inference-sim/pull/1\n  set_by: maintainer\n  min_points: 10\n  margin: {error: 0.05, distance: 0}\n")
	if err := f.verify(); err != nil {
		t.Fatalf("a complete override was refused: %v", err)
	}
}

// Admit is the guard a fit calls before it reads a source: a source, role, step or pin the
// record does not cover is refused, and nothing is read.
func TestAdmit(t *testing.T) {
	r := mustLoad(t, committedGates)
	for name, c := range map[string]struct {
		id     string
		step   prereg.Step
		source string
		role   ledger.Role
		want   string
	}{
		"fit on a covered source":            {"refit-engine-path-vllm", prereg.StepFit, "vllm-ci-perf", ledger.RoleFit, ""},
		"select on its validate share":       {"refit-engine-path-vllm", prereg.StepSelect, "vllm-ci-perf", ledger.RoleValidate, ""},
		"select on its select share":         {"refit-kernel-keyed", prereg.StepSelect, "sglang-traces", ledger.RoleSelect, ""},
		"a source the record does not cover": {"refit-engine-path-vllm", prereg.StepFit, "sglang-nightly", ledger.RoleFit, "not covered"},
		"a source no record covers":          {"refit-kernel-keyed", prereg.StepFit, "inferencex-new", ledger.RoleFit, "not covered"},
		"no such pre-registration":           {"refit-nothing", prereg.StepFit, "vllm-ci-perf", ledger.RoleFit, "no pre-registration"},
		"anchor-only source":                 {"refit-kernel-keyed", prereg.StepFit, "configiq", ledger.RoleFit, "anchor-only"},
		"fit step reads the select share":    {"refit-kernel-keyed", prereg.StepFit, "vllm-ci-perf", ledger.RoleSelect, "not select"},
		"select step reads the fit share":    {"refit-kernel-keyed", prereg.StepSelect, "vllm-ci-perf", ledger.RoleFit, "not fit"},
		"held role":                          {"refit-kernel-keyed", prereg.StepFit, "inferencex", ledger.RoleHeld, "not held"},
		"anchor role":                        {"refit-kernel-keyed", prereg.StepSelect, "inferencex", ledger.RoleAnchor, "not anchor"},
		"report role":                        {"refit-kernel-keyed", prereg.StepSelect, "inferencex", ledger.RoleReport, "not report"},
		"unknown step":                       {"refit-kernel-keyed", prereg.Step("tune"), "inferencex", ledger.RoleFit, "step"},
	} {
		t.Run(name, func(t *testing.T) {
			err := r.Admit(committed.LedgerDir, c.id, c.step, c.source, c.role)
			switch {
			case c.want == "" && err != nil:
				t.Fatalf("refused: %v", err)
			case c.want != "" && err == nil:
				t.Fatal("admitted")
			case c.want != "" && !strings.Contains(err.Error(), c.want):
				t.Errorf("error %q does not mention %q", err, c.want)
			}
		})
	}
}

// A fit that was admitted when the record was registered is refused once the ledger it pinned
// moves: the pre-registration is fixed, so what changed needs a new one.
func TestAdmitRefusesAStaleRecord(t *testing.T) {
	f := newFixture(t)
	r := mustLoad(t, f.gates)
	if err := r.Admit(f.env.LedgerDir, "refit-engine-path-vllm", prereg.StepFit, "vllm-ci-perf", ledger.RoleFit); err != nil {
		t.Fatalf("a fresh record was refused: %v", err)
	}
	f.edit(f.env.LedgerDir+"/sources.yaml", "rows: 7795", "rows: 7796")
	err := r.Admit(f.env.LedgerDir, "refit-engine-path-vllm", prereg.StepFit, "vllm-ci-perf", ledger.RoleFit)
	if err == nil || !strings.Contains(err.Error(), "stale") {
		t.Fatalf("a record whose source changed was admitted: %v", err)
	}
}

// ComputePins reproduces the committed pins, so a record can be re-registered with the
// command that prints them.
func TestComputePinsMatchesTheCommittedRecords(t *testing.T) {
	r := mustLoad(t, committedGates)
	for _, rec := range r.Records {
		got, err := prereg.ComputePins(committed.LedgerDir, r.Defaults, rec.Sources)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, rec.Pins) {
			t.Errorf("%s: pins %+v, committed %+v", rec.ID, got, rec.Pins)
		}
	}
}

func writeScript(t *testing.T, root, rel, body string) {
	t.Helper()
	p := filepath.Join(root, "registry/scripts", rel)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// A fit script that reads a gated source must name a pre-registration that covers it.
func TestScanScripts(t *testing.T) {
	r := mustLoad(t, committedGates)
	for name, c := range map[string]struct {
		body string
		want string
	}{
		"names no source":                 {"print('fit')\n", ""},
		"names a covered gated source":    {"# prereg: refit-engine-path-vllm\nSRC = 'vllm-ci-perf'\n", ""},
		"covered by one of several":       {"# prereg: refit-engine-path-vllm\n# prereg: refit-engine-path-sglang\nA = 'vllm-ci-perf'\nB = 'sglang-traces'\n", ""},
		"a longer name is another source": {"SRC = 'vllm-ci-perf-v2'\nOTHER = 'xsglang-traces'\n", ""},
		"no prereg line":                  {"SRC = 'vllm-ci-perf'\n", "no \"# prereg: <id>\" line"},
		"a record that does not cover it": {"# prereg: refit-engine-path-sglang\nSRC = 'vllm-ci-perf'\n", "none of the pre-registrations it names covers"},
		"a record that does not exist":    {"# prereg: refit-made-up\nSRC = 'vllm-ci-perf'\n", "does not exist"},
		"an anchor-only source":           {"# prereg: refit-kernel-keyed\nSRC = 'configiq'\n", "anchor-only"},
		"a Go fit":                        {"// prereg: refit-kernel-keyed\nconst s = \"sglang-nightly\"\n", ""},
	} {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			ext := ".py"
			if name == "a Go fit" {
				ext = ".go"
			}
			writeScript(t, root, "fit_x"+ext, c.body)
			err := r.ScanScripts(root)
			switch {
			case c.want == "" && err != nil:
				t.Fatalf("refused: %v", err)
			case c.want != "" && err == nil:
				t.Fatal("accepted")
			case c.want != "" && !strings.Contains(err.Error(), c.want):
				t.Errorf("error %q does not mention %q", err, c.want)
			}
		})
	}
	t.Run("test files are not fits", func(t *testing.T) {
		root := t.TempDir()
		writeScript(t, root, "fit_x_test.go", "package x\nconst s = \"vllm-ci-perf\"\n")
		writeScript(t, root, "test_fit_x.py", "SRC = 'configiq'\n")
		writeScript(t, root, "testdata/y.py", "SRC = 'configiq'\n")
		if err := r.ScanScripts(root); err != nil {
			t.Fatalf("refused: %v", err)
		}
	})
}

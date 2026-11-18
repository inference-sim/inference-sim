package contracts

// Each example document under testdata/examples must load and validate without a single
// finding, so a reader who copies one starts from something correct.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/fields"
	"github.com/inference-sim/blis-substrate/contracts/spec/trace"
)

// fixture returns a loaded document, panicking on a load error: every path here is committed,
// so a failure is a broken checkout rather than a finding.
func fixture[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

func requireSilent(t *testing.T, name string, rep Report) {
	t.Helper()
	for _, p := range rep.Problems() {
		t.Errorf("%s: %s", name, p)
	}
}

func TestDocExamplesValidate(t *testing.T) {
	ex := func(name string) string { return filepath.Join("testdata", "examples", name) }

	t.Run("colocated", func(t *testing.T) {
		requireSilent(t, "colocated", Validate(Bundle{Rules: openAnyEngine,
			Scenario:   fixture(LoadScenario(ex("scenario.yaml"))),
			Deployment: fixture(LoadDeployment(ex("deployment.yaml"))),
			Model:      fixture(LoadModelGraph("../catalog/models/llama-3.1-8b-instruct/graph.yaml")),
			Chip:       fixture(LoadChip("../catalog/hardware/h100.yaml")),
		}))
	})
	t.Run("disaggregated", func(t *testing.T) {
		requireSilent(t, "disaggregated", Validate(Bundle{Rules: openAnyEngine,
			Scenario:   fixture(LoadScenario(ex("scenario-disaggregated.yaml"))),
			Deployment: fixture(LoadDeployment(ex("deployment-disaggregated.yaml"))),
			Model:      fixture(LoadModelGraph("../catalog/models/deepseek-v3/graph.yaml")),
			Chip:       fixture(LoadChip("../catalog/hardware/h200.yaml")),
			Fabric:     fixture(LoadFabric("../catalog/networks/ib-400g.yaml")),
			Devices:    fixture(LoadStorageDevices("../catalog/devices/storage.yaml")),
		}))
	})
	t.Run("coefficient set", func(t *testing.T) {
		requireSilent(t, "coefficient set", Validate(Bundle{
			Coefficients: []*coefficient.Set{fixture(LoadCoefficientSet(ex("coefficient-set.yaml")))},
		}))
	})
	t.Run("acceptance profile set", func(t *testing.T) {
		requireSilent(t, "acceptance profile set", Validate(Bundle{
			AcceptanceProfiles: []*acceptance.ProfileSet{fixture(LoadAcceptanceProfileSet(ex("acceptance-profile-set.yaml")))},
		}))
	})
	t.Run("replay", func(t *testing.T) {
		requireSilent(t, "replay", Validate(Bundle{Rules: openAnyEngine,
			Scenario: fixture(LoadScenario(ex("replay-binding.yaml"))),
		}))
	})
	t.Run("catalog replay and a reference to it", func(t *testing.T) {
		w := fixture(LoadCatalogWorkload(ex("workload-replay.yaml")))
		requireSilent(t, "catalog replay", Validate(Bundle{ReplayWorkload: w.Replay}))
		s := fixture(LoadScenario(ex("replay-reference.yaml")))
		requireSilent(t, "replay reference", Validate(Bundle{Scenario: s, Rules: openAnyEngine}))
		root := t.TempDir()
		if err := os.MkdirAll(filepath.Join(root, "workloads"), 0o755); err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(ex("workload-replay.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, "workloads", "workload-replay.yaml"), b, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := ResolveReplay(root, s.Workload.Replay); err != nil {
			t.Errorf("the reference example does not resolve against the catalog example: %v", err)
		}
	})
	t.Run("acceptance pin", func(t *testing.T) {
		requireSilent(t, "acceptance pin", Validate(Bundle{Rules: openAnyEngine,
			Scenario: fixture(LoadScenario(ex("scenario-acceptance.yaml"))),
		}))
	})
	t.Run("evaluation run", func(t *testing.T) {
		requireSilent(t, "evaluation run", Validate(Bundle{
			Evaluation: fixture(LoadEvaluationRun(ex("evaluation-run.yaml"))),
		}))
	})
	t.Run("measured run and its licence registry", func(t *testing.T) {
		reg := fixture(LoadMeasurementSources(ex("measurement-sources.yaml")))
		for _, p := range reg.Validate().All() {
			t.Errorf("measurement-sources.yaml: %s", p)
		}
		run := fixture(LoadMeasuredRun(ex("measured-run.yaml")))
		for _, p := range run.ValidateWithSources(reg).All() {
			t.Errorf("measured-run.yaml: %s", p)
		}
	})
	t.Run("model graph", func(t *testing.T) {
		requireSilent(t, "model graph", Validate(Bundle{
			Model: fixture(LoadModelGraph(ex("model-graph-deepseek-v4.1.yaml"))),
		}))
	})
	t.Run("connector registry", func(t *testing.T) {
		r := fixture(LoadConnectorRegistry(ex("connector-registry.yaml")))
		for _, p := range r.Validate().All() {
			t.Errorf("connector-registry.yaml: %s", p)
		}
	})
	t.Run("result", func(t *testing.T) {
		requireSilent(t, "result", Validate(Bundle{
			Result: fixture(LoadResult(ex("result.json"))),
		}))
	})
	t.Run("field list", func(t *testing.T) {
		if _, err := fields.Load(ex("field-list.json")); err != nil {
			t.Error(err)
		}
	})
	t.Run("kernel-implementation table", func(t *testing.T) {
		if _, err := LoadKernelImplTable(ex("kernel-impl-table.json")); err != nil {
			t.Error(err)
		}
	})
	t.Run("trace", func(t *testing.T) {
		requireSilent(t, "trace", Validate(Bundle{Rules: openAnyEngine,
			Scenario: fixture(LoadScenario(ex("trace-binding.yaml"))),
		}))
	})
	t.Run("workload specification", func(t *testing.T) {
		s := fixture(LoadWorkloadSpec(ex("workload-spec.yaml")))
		for _, p := range s.Validate().All() {
			t.Errorf("workload-spec.yaml: %s", p)
		}
	})
	t.Run("point record", func(t *testing.T) {
		r := fixture(LoadPointRecord(ex("point-record.json")))
		for _, p := range r.Validate().All() {
			t.Errorf("point-record.json: %s", p)
		}
	})
	t.Run("trace file", func(t *testing.T) {
		h, err := os.Open(ex("trace-header.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := h.Close(); err != nil {
				t.Error(err)
			}
		})
		d, err := os.Open(ex("trace-data.csv"))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := d.Close(); err != nil {
				t.Error(err)
			}
		})
		if _, err := trace.Read(h, d); err != nil {
			t.Error(err)
		}
	})
	// Every file in contracts/testdata/examples/ must be one of the cases above, so an example added
	// later cannot be shown on a page without ever being validated.
	covered := map[string]bool{
		"scenario.yaml": true, "deployment.yaml": true,
		"scenario-disaggregated.yaml": true, "deployment-disaggregated.yaml": true,
		"coefficient-set.yaml": true, "evaluation-run.yaml": true, "trace-binding.yaml": true,
		"model-graph-deepseek-v4.1.yaml": true,
		"acceptance-profile-set.yaml":    true, "scenario-acceptance.yaml": true,
		"replay-binding.yaml": true, "workload-replay.yaml": true, "replay-reference.yaml": true,
		"measured-run.yaml": true, "measurement-sources.yaml": true,
		"connector-registry.yaml": true,
		"result.json":             true,
		"field-list.json":         true,
		"kernel-impl-table.json":  true,
		"trace-header.yaml":       true, "trace-data.csv": true, "workload-spec.yaml": true, "point-record.json": true,
	}
	all, err := filepath.Glob(ex("*"))
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range all {
		if !covered[filepath.Base(path)] {
			t.Errorf("%s is not validated by TestDocExamplesValidate; add a case for it", path)
		}
	}

	// The by-name references the examples make must resolve against the vendored catalog,
	// or an example would name a chip or a workload the reader cannot find.
	for _, name := range []string{"scenario.yaml", "scenario-disaggregated.yaml", "trace-binding.yaml",
		"scenario-acceptance.yaml", "replay-binding.yaml"} {
		s := fixture(LoadScenario(ex(name)))
		rep := ValidateAgainstCatalog(Bundle{Scenario: s}, catalogFixtures, "")
		for _, p := range rep.Problems() {
			if strings.HasPrefix(p.Path, "references.coefficients") {
				continue // the registry is not vendored
			}
			t.Errorf("%s: %s", name, p)
		}
	}
}

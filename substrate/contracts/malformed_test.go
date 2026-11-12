package contracts

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/conformance/doctest"
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/connector"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/evaluation"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/measurement"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/pointrecord"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
	"github.com/inference-sim/blis-substrate/contracts/spec/routing"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

// SUB-12's check: every loader of a BLIS document refuses the malformed-document set of
// contracts/conformance/doctest, made from a valid example of its document. Each loader
// reads from a file, as a caller does.
func TestEveryLoaderRefusesTheMalformedSet(t *testing.T) {
	ex := func(name string) string { return filepath.Join("..", "docs", "contracts", "examples", name) }
	cat := func(rel string) string { return filepath.Join("..", "catalog", rel) }
	via := func(name string, load func(string) error) func([]byte) error {
		return func(b []byte) error {
			p := filepath.Join(t.TempDir(), name)
			if err := os.WriteFile(p, b, 0o644); err != nil {
				t.Fatal(err)
			}
			return load(p)
		}
	}
	ignore := func(_ any, err error) error { return err }
	loaders := []struct {
		name, valid string
		target      any
		load        func(string) error
	}{
		{"LoadScenario", ex("scenario.yaml"), &scenario.Scenario{}, func(p string) error { return ignore(LoadScenario(p)) }},
		{"LoadDeployment", ex("deployment.yaml"), &deployment.Deployment{}, func(p string) error { return ignore(LoadDeployment(p)) }},
		{"LoadModelGraph", cat("models/qwen3-14b/graph.yaml"), &model.Graph{}, func(p string) error { return ignore(LoadModelGraph(p)) }},
		{"LoadModelIdentity", cat("models/qwen3-14b/model.yaml"), &model.Identity{}, func(p string) error { return ignore(LoadModelIdentity(p)) }},
		{"LoadChip", cat("hardware/h100.yaml"), &hardware.Chip{}, func(p string) error { return ignore(LoadChip(p)) }},
		{"LoadFabric", cat("networks/ib-400g.yaml"), &hardware.Fabric{}, func(p string) error { return ignore(LoadFabric(p)) }},
		{"LoadStorageDevices", cat("devices/storage.yaml"), &map[string]*hardware.StorageDevice{}, func(p string) error { return ignore(LoadStorageDevices(p)) }},
		{"LoadCoefficientSet", ex("coefficient-set.yaml"), &coefficient.Set{}, func(p string) error { return ignore(LoadCoefficientSet(p)) }},
		{"LoadAcceptanceProfileSet", ex("acceptance-profile-set.yaml"), &acceptance.ProfileSet{}, func(p string) error { return ignore(LoadAcceptanceProfileSet(p)) }},
		{"LoadRoutingProfileSet", filepath.Join("..", "registry", "routing", "servingstudio-dsv41-flash.yaml"), &routing.ProfileSet{}, func(p string) error { return ignore(LoadRoutingProfileSet(p)) }},
		{"LoadConnectorRegistry", ex("connector-registry.yaml"), &connector.Registry{}, func(p string) error { return ignore(LoadConnectorRegistry(p)) }},
		{"LoadWorkload", cat("workloads/chatbot.yaml"), &workload.Shape{}, func(p string) error { return ignore(LoadWorkload(p)) }},
		{"LoadCatalogWorkload (replay)", ex("workload-replay.yaml"), &catalogReplayFile{}, func(p string) error { return ignore(LoadCatalogWorkload(p)) }},
		{"LoadEvaluationRun", ex("evaluation-run.yaml"), &evaluation.Run{}, func(p string) error { return ignore(LoadEvaluationRun(p)) }},
		{"LoadMeasuredRun", ex("measured-run.yaml"), &measurement.MeasuredRun{}, func(p string) error { return ignore(LoadMeasuredRun(p)) }},
		{"LoadMeasurementSources", ex("measurement-sources.yaml"), &measurement.Registry{}, func(p string) error { return ignore(LoadMeasurementSources(p)) }},
		{"LoadResult (YAML)", ex("result.json"), &result.Result{}, func(p string) error { return ignore(LoadResult(p)) }},
		{"LoadWorkloadSpec", ex("workload-spec.yaml"), &workload.Spec{}, func(p string) error { return ignore(LoadWorkloadSpec(p)) }},
		{"LoadPointRecord", ex("point-record.json"), &pointrecord.Record{}, func(p string) error { return ignore(LoadPointRecord(p)) }},
	}
	covered := map[string]bool{"LoadBundle": true} // LoadBundle's two documents are the scenario's and the deployment's
	for _, l := range loaders {
		covered[strings.Fields(l.name)[0]] = true
	}
	ran := map[string]bool{}
	for _, l := range loaders {
		t.Run(l.name, func(t *testing.T) {
			valid, err := os.ReadFile(l.valid)
			if err != nil {
				t.Fatal(err)
			}
			if filepath.Ext(l.valid) == ".json" {
				valid = jsonToYAML(t, valid)
			}
			for _, c := range doctest.Run(t, valid, l.target, via("doc.yaml", l.load)) {
				ran[c] = true
			}
		})
	}
	// SUB-12 says every loader. A Load function added to load.go without an entry above is a
	// loader the malformed set never reaches, so it fails here, naming itself.
	src, err := os.ReadFile("load.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range regexp.MustCompile(`(?m)^func (Load[A-Za-z]+)\(`).FindAllStringSubmatch(string(src), -1) {
		if !covered[m[1]] {
			t.Errorf("%s loads a BLIS document but is not in the malformed-set loader list", m[1])
		}
	}
	for _, c := range []string{"unknown key", "repeated key", "fraction in an integer field",
		"integer a float64 cannot hold", "NaN", "infinity", "recursive alias"} {
		if !ran[c] {
			t.Errorf("no loader's example exercised %q; add an example with such a field", c)
		}
	}
}

// The Result's JSON encoding is held to the same rules as its YAML one.
func TestTheResultsJSONIsHeldToTheSameRules(t *testing.T) {
	for name, body := range map[string]string{
		"fraction in an integer": `{"kind": "Result", "schema_version": 2.5}`,
		"repeated key":           `{"kind": "Result", "kind": "Result"}`,
	} {
		var r result.Result
		if err := result.Unmarshal([]byte(body), &r); err == nil {
			t.Errorf("%s: loaded", name)
		}
	}
}

// jsonToYAML re-expresses a JSON example as YAML, so the malformed set, which edits YAML,
// can be made from it. LoadResult reads either encoding.
func jsonToYAML(t *testing.T, b []byte) []byte {
	t.Helper()
	n, err := validate.JSONNode(b)
	if err != nil {
		t.Fatal(err)
	}
	out, err := yaml.Marshal(n)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// SUB-12: every declared field is read. Each value in each loader's valid example is
// changed in turn; the loader must load something different, or refuse.
func TestEveryLoaderReadsEveryFieldItsExampleStates(t *testing.T) {
	ex := func(name string) string { return filepath.Join("..", "docs", "contracts", "examples", name) }
	cat := func(rel string) string { return filepath.Join("..", "catalog", rel) }
	dir := t.TempDir()
	file := func(b []byte) string {
		p := filepath.Join(dir, "doc.yaml")
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	loaders := []struct {
		name, valid string
		load        func(string) (any, error)
	}{
		{"LoadScenario", ex("scenario.yaml"), func(p string) (any, error) { return LoadScenario(p) }},
		{"LoadScenario (acceptance)", ex("scenario-acceptance.yaml"), func(p string) (any, error) { return LoadScenario(p) }},
		{"LoadScenario (replay)", ex("replay-binding.yaml"), func(p string) (any, error) { return LoadScenario(p) }},
		{"LoadScenario (trace)", ex("trace-binding.yaml"), func(p string) (any, error) { return LoadScenario(p) }},
		{"LoadDeployment", ex("deployment.yaml"), func(p string) (any, error) { return LoadDeployment(p) }},
		{"LoadDeployment (disaggregated)", ex("deployment-disaggregated.yaml"), func(p string) (any, error) { return LoadDeployment(p) }},
		{"LoadModelGraph", ex("model-graph-deepseek-v4.1.yaml"), func(p string) (any, error) { return LoadModelGraph(p) }},
		{"LoadModelIdentity", cat("models/qwen3-14b/model.yaml"), func(p string) (any, error) { return LoadModelIdentity(p) }},
		{"LoadChip", cat("hardware/gb200-nvl72.yaml"), func(p string) (any, error) { return LoadChip(p) }},
		{"LoadFabric", cat("networks/ib-400g.yaml"), func(p string) (any, error) { return LoadFabric(p) }},
		{"LoadStorageDevices", cat("devices/storage.yaml"), func(p string) (any, error) { return LoadStorageDevices(p) }},
		{"LoadCoefficientSet", ex("coefficient-set.yaml"), func(p string) (any, error) { return LoadCoefficientSet(p) }},
		{"LoadAcceptanceProfileSet", ex("acceptance-profile-set.yaml"), func(p string) (any, error) { return LoadAcceptanceProfileSet(p) }},
		{"LoadRoutingProfileSet", filepath.Join("..", "registry", "routing", "servingstudio-dsv41-flash.yaml"), func(p string) (any, error) { return LoadRoutingProfileSet(p) }},
		{"LoadConnectorRegistry", ex("connector-registry.yaml"), func(p string) (any, error) { return LoadConnectorRegistry(p) }},
		{"LoadCatalogWorkload (shape)", cat("workloads/sharegpt.yaml"), func(p string) (any, error) { return LoadCatalogWorkload(p) }},
		{"LoadCatalogWorkload (replay)", ex("workload-replay.yaml"), func(p string) (any, error) { return LoadCatalogWorkload(p) }},
		{"LoadEvaluationRun", ex("evaluation-run.yaml"), func(p string) (any, error) { return LoadEvaluationRun(p) }},
		{"LoadMeasuredRun", ex("measured-run.yaml"), func(p string) (any, error) { return LoadMeasuredRun(p) }},
		{"LoadMeasurementSources", ex("measurement-sources.yaml"), func(p string) (any, error) { return LoadMeasurementSources(p) }},
		{"LoadResult", ex("result.json"), func(p string) (any, error) { return LoadResult(p) }},
		{"LoadWorkloadSpec", ex("workload-spec.yaml"), func(p string) (any, error) { return LoadWorkloadSpec(p) }},
		{"LoadPointRecord", ex("point-record.json"), func(p string) (any, error) { return LoadPointRecord(p) }},
	}
	for _, l := range loaders {
		t.Run(l.name, func(t *testing.T) {
			valid, err := os.ReadFile(l.valid)
			if err != nil {
				t.Fatal(err)
			}
			if filepath.Ext(l.valid) == ".json" {
				valid = jsonToYAML(t, valid)
			}
			dropped, err := doctest.Dropped(valid, func(b []byte) (any, error) { return l.load(file(b)) })
			if err != nil {
				t.Fatal(err)
			}
			for _, d := range dropped {
				t.Errorf("%s: changing %s changes nothing the loader returns: the field is dropped", l.valid, d)
			}
		})
	}
}

// blis-schemas#107: an engine block with a recursive alias is refused when it is decoded,
// rather than held as a graph whose clone never ends.
func TestAnEngineBlockWithARecursiveAliasIsRefused(t *testing.T) {
	var h struct {
		Engine deployment.EngineBlock `yaml:"engine"`
	}
	err := yaml.Unmarshal([]byte("engine: &a [*a]\n"), &h)
	if err == nil || !strings.Contains(err.Error(), "recursive alias") {
		t.Fatalf("want a refusal of the recursive alias, got %v", err)
	}
}

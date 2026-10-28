// Moved from blis-latency-kernel/internal/harness/load_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package harness

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// A minimal valid pair, as the two documents LoadBundle expects.
const (
	validScenario = `kind: Scenario
name: x
engine_version: "0.29.0"
model: m
coefficients: [c]
cluster:
  hardware: h200
  nodes: 1
  gpus_per_node: 8
`
	validDeployment = `kind: Deployment
name: x
pools:
  - role: colocated
    nodes: 1
    parallel:
      tp: 8
      pp: 1
      dp: 1
`
)

func write(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "f.yaml")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestLoadBundleReadsBothDocuments is the happy path: one file, two documents.
func TestLoadBundleReadsBothDocuments(t *testing.T) {
	sc, dep, err := LoadBundle(write(t, validScenario+"---\n"+validDeployment))
	if err != nil {
		t.Fatalf("LoadBundle: %v", err)
	}
	if sc.Kind != "Scenario" || sc.Cluster.Hardware != "h200" {
		t.Errorf("scenario did not parse: %+v", sc)
	}
	if dep.Kind != "Deployment" || len(dep.Pools) != 1 || dep.Pools[0].Parallel.TP != 8 {
		t.Errorf("deployment did not parse: %+v", dep)
	}
}

// TestLoadBundleRejectsIncompleteFiles covers every way the pair can be malformed.
//
// Each case must produce an error that NAMES THE FILE, because the failure these guard
// against is a kernel failing far from the document at fault — an index panic on a pool
// that is not there, or a zero value priced as if it were measured. A reader needs to know
// which of 685 fixtures to open.
func TestLoadBundleRejectsIncompleteFiles(t *testing.T) {
	cases := []struct{ what, body, wantIn string }{
		{"empty file", "", "is empty"},
		{"scenario only", validScenario, "no deployment"},
		// A bare separator, or one followed by only comments, is a PRESENT but empty
		// document rather than end-of-stream, so it lands on the zero-pools guard instead
		// of the missing-document one. Both reject it and both name the file, which is
		// what matters; the cases are kept distinct so a future change to either message
		// has to decide deliberately which one applies.
		{"separator but no second document", validScenario + "---\n", "no pools"},
		{"second document is only a comment", validScenario + "---\n# gone\n", "no pools"},
		{"second document is explicitly null", validScenario + "---\nnull\n", "no pools"},
		{"second document has no pools", validScenario + "---\nkind: Deployment\nname: x\n", "no pools"},
		{"second document has an empty pool list",
			validScenario + "---\nkind: Deployment\nname: x\npools: []\n", "no pools"},
		{"second document is truncated", validScenario + "---\nkind: \"Deploy", "deployment"},
		// Strict decoding is the point: a misspelled key must not leave a document that
		// validates while silently omitting the setting its author intended. (A misspelled key
		// inside a pool's engine block is the engine model's to refuse: blis-schemas holds the
		// block unopened, so TestOpenRefusesAMisspelledEngineKey covers it.)
		{"misspelled deployment key",
			validScenario + "---\nkind: Deployment\nname: x\npools:\n  - role: colocated\n    nodes: 1\n    paralel: {tp: 8, pp: 1, dp: 1}\n",
			"paralel"},
		{"misspelled scenario key",
			"kind: Scenario\nname: x\nmodle: m\n---\n" + validDeployment, "modle"},
		// The pre-v0.2.0 shape: pools under the Scenario. It must not be silently accepted
		// as a scenario that happens to have an unknown field.
		{"pre-split shape", strings.Replace(validScenario, "cluster:",
			"pools:\n  - role: colocated\n    nodes: 1\ncluster:", 1) + "---\n" + validDeployment,
			"pools"},
	}
	for _, c := range cases {
		t.Run(c.what, func(t *testing.T) {
			path := write(t, c.body)
			_, _, err := LoadBundle(path)
			if err == nil {
				t.Fatal("accepted a file that cannot build a kernel")
			}
			if !strings.Contains(err.Error(), c.wantIn) {
				t.Errorf("error %q does not mention %q", err, c.wantIn)
			}
			if !strings.Contains(err.Error(), filepath.Base(path)) {
				t.Errorf("error %q does not name the file", err)
			}
		})
	}
}

// TestLoadBundleIgnoresDocumentsPastTheSecond pins what a third document does.
//
// It is read by nobody, so a file carrying one is describing something this loader cannot
// honour. Worth a test because the behaviour is silent either way, and a reader deciding
// whether to add one should find the answer here rather than in the decoder.
func TestLoadBundleIgnoresDocumentsPastTheSecond(t *testing.T) {
	path := write(t, validScenario+"---\n"+validDeployment+
		"---\nkind: Deployment\nname: third\npools: []\n")
	_, dep, err := LoadBundle(path)
	if err != nil {
		t.Fatalf("LoadBundle: %v", err)
	}
	if dep.Name != "x" {
		t.Errorf("deployment name = %q; the SECOND document is the deployment", dep.Name)
	}
}

// TestLoadBundleReportsAMissingFile keeps the path in the error.
func TestLoadBundleReportsAMissingFile(t *testing.T) {
	_, _, err := LoadBundle(filepath.Join(t.TempDir(), "absent.yaml"))
	if err == nil {
		t.Fatal("a missing file was accepted")
	}
	if !strings.Contains(err.Error(), "absent.yaml") {
		t.Errorf("error %q does not name the file", err)
	}
}

// TestDefaultRootsPointAtTheVendoredCopy is the hermeticity check.
//
// The suite reads a catalog and registry pinned under testdata (fetched by
// scripts/fetch-testdata.sh). It used to read absolute
// paths in one developer's home directory and SKIP when that failed, which is how a real
// incompatibility stayed hidden for the life of a dependency pin. If these defaults ever
// resolve somewhere else, that silence comes back.
func TestDefaultRootsPointAtTheVendoredCopy(t *testing.T) {
	t.Setenv("BLIS_CATALOG", "")
	t.Setenv("BLIS_REGISTRY", "")

	for _, c := range []struct{ what, got, wantSuffix, probe string }{
		{"catalog", DefaultCatalog(), filepath.Join("testdata", "catalog"),
			filepath.Join("devices", "storage.yaml")},
		{"registry", DefaultRegistry(), filepath.Join("testdata", "registry"),
			filepath.Join("coefficients", "cost-model-primitives.yaml")},
	} {
		t.Run(c.what, func(t *testing.T) {
			if !strings.HasSuffix(c.got, c.wantSuffix) {
				t.Errorf("%s root is %q, want it to end in %q", c.what, c.got, c.wantSuffix)
			}
			// Resolving to a path is not enough; the artifacts have to be there.
			if _, err := os.Stat(filepath.Join(c.got, c.probe)); err != nil {
				t.Errorf("%s root %q does not hold %s (run scripts/fetch-testdata.sh): %v",
					c.what, c.got, c.probe, err)
			}
		})
	}
}

// TestDefaultRootsHonourTheEnvironment keeps the escape hatch working, so a working copy
// can be scored against a live upstream checkout.
func TestDefaultRootsHonourTheEnvironment(t *testing.T) {
	t.Setenv("BLIS_CATALOG", "/tmp/some-catalog")
	t.Setenv("BLIS_REGISTRY", "/tmp/some-registry")
	if got := DefaultCatalog(); got != "/tmp/some-catalog" {
		t.Errorf("DefaultCatalog() = %q; BLIS_CATALOG must win", got)
	}
	if got := DefaultRegistry(); got != "/tmp/some-registry" {
		t.Errorf("DefaultRegistry() = %q; BLIS_REGISTRY must win", got)
	}
}

// Replicas counts schedulers: every data-parallel rank of every engine in the pool, which is
// the pool's GPUs over the pp x tp x pcp GPUs one data-parallel rank occupies. Three
// independent tp=8 engines on 24 GPUs and one tp=8, dp=3 engine on the same 24 GPUs both run
// three schedulers, so a summed `running` count divides by three either way.
func TestReplicasCountsEveryDataParallelRankOfEveryEngine(t *testing.T) {
	bundle := func(nodes, tp, dp, pcp int) string {
		return strings.NewReplacer("NODES", strconv.Itoa(nodes), "TP", strconv.Itoa(tp),
			"DP", strconv.Itoa(dp), "PCP", strconv.Itoa(pcp)).Replace(`kind: Scenario
name: x
engine_version: "0.29.0"
model: m
coefficients: [c]
cluster:
  hardware: h200
  nodes: NODES
  gpus_per_node: 8
---
kind: Deployment
name: x
pools:
  - role: colocated
    nodes: NODES
    parallel:
      tp: TP
      pp: 1
      dp: DP
      pcp: PCP
`)
	}
	for _, c := range []struct {
		name               string
		nodes, tp, dp, pcp int
		want               int
	}{
		{"three independent tp=8 engines", 3, 8, 1, 1, 3},
		{"one tp=8 engine at dp=3", 3, 8, 3, 1, 3},
		{"one engine filling one node", 1, 8, 1, 1, 1},
		{"two tp=4 engines on one node", 1, 4, 1, 1, 2},
		{"one tp=8, pcp=2 engine on two nodes", 2, 8, 1, 2, 1},
	} {
		p := write(t, bundle(c.nodes, c.tp, c.dp, c.pcp))
		got, err := Replicas(filepath.Base(p), Repos{Scenarios: filepath.Dir(p)})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if got != c.want {
			t.Errorf("%s: %d schedulers, want %d", c.name, got, c.want)
		}
	}
}

// A misspelled key inside a pool's engine block is refused when the kernel is opened, at its line
// and naming the file: blis-schemas holds the block unopened, and the engine settings are decoded
// strictly by the engine model (Repos.EngineSettings, vllmsettings.OfPool for vLLM).
func TestOpenRefusesAMisspelledEngineKey(t *testing.T) {
	src, err := os.ReadFile(filepath.Join(repoRoot(), "testdata", "minimax-m25-h200-tp8.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	const key = "      cache_dtype: fp8\n"
	if strings.Count(string(src), key) != 1 {
		t.Fatalf("the fixture states cache_dtype %d times, want once", strings.Count(string(src), key))
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "typo.yaml"),
		[]byte(strings.Replace(string(src), key, "      cache_dytpe: fp8\n", 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadBundle(filepath.Join(dir, "typo.yaml")); err != nil {
		t.Fatalf("LoadBundle refused the engine block it holds unopened: %v", err)
	}
	_, err = Open("typo.yaml", Repos{Scenarios: dir, Catalog: DefaultCatalog(), Registry: DefaultRegistry()})
	if err == nil || !strings.Contains(err.Error(), "cache_dytpe") || !strings.Contains(err.Error(), "typo.yaml") {
		t.Errorf("Open: %v; want a refusal naming the misspelled key and the file", err)
	}
}

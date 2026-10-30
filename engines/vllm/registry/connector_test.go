package registry

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/spec/connector"

	blisvllm "github.com/inference-sim/blis-vllm"
)

// The registry file validates against blis-schemas' connector-registry schema.
func TestTheRegistryValidates(t *testing.T) {
	r, err := schemas.LoadConnectorRegistry("connectors.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for _, pr := range r.Validate().All() {
		t.Errorf("connectors.yaml: %v", pr)
	}
}

// An entry earns behaves_like only by passing the differential tests (#2021 B10): every
// model an entry names must be one this module diff-tests against vLLM v0.32.0, and every
// connector so tested must be registered. "Diff-tested" is checked on the evidence itself, not
// on file names: the model's package must have a diff test that names it, golden traces that
// are vLLM v0.32.0's own output at the release's commit, one per committed scenario, each
// for this connector, and the entry must name a connector vLLM v0.32.0 registers.
func TestEveryEntryIsEarnedByADiffTest(t *testing.T) {
	r, err := schemas.LoadConnectorRegistry("connectors.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range r.Names() {
		e := r.Connectors[name]
		m, ok := models[e.BehavesLike]
		if !ok {
			t.Errorf("%s behaves like %q, which no differential test in this module checks", name, e.BehavesLike)
			continue
		}
		if !contains(m.connectors, name) {
			t.Errorf("%s behaves like %q, but that model's diff test runs only %v", name, e.BehavesLike, m.connectors)
		}
		for _, p := range earned("..", m) {
			t.Errorf("%s behaves like %q: %s", name, e.BehavesLike, p)
		}
		if !strings.Contains(e.Evidence, "v0.32.0") {
			t.Errorf("%s: evidence %q does not name the release it was tested against", name, e.Evidence)
		}
	}
	var tested []string
	for m := range models {
		tested = append(tested, m)
	}
	sort.Strings(tested)
	for _, m := range tested {
		found := false
		for _, name := range r.Names() {
			if r.Connectors[name].BehavesLike == m {
				found = true
			}
		}
		if !found {
			t.Errorf("model %q is diff-tested but no connector entry behaves like it", m)
		}
	}
}

// earned lists what is missing from a model's evidence, its paths relative to the module root: a diff test that runs the model's
// scenarios, and, for every committed scenario, a golden trace that vLLM v0.32.0 produced for
// it with one of the model's connectors. A golden that is not such a trace (an empty object, a
// trace of another release or another scenario set) does not count.
func earned(root string, m model) []string {
	var missing []string
	pkg := filepath.Join(root, filepath.FromSlash(m.pkg))
	src, err := os.ReadFile(filepath.Join(pkg, "diff_test.go"))
	if err != nil {
		return []string{fmt.Sprintf("%s has no diff test", m.pkg)}
	}
	if !strings.Contains(string(src), m.scenarios) {
		missing = append(missing, fmt.Sprintf("%s's diff test does not run %s", m.pkg, m.scenarios))
	}
	scenarios, _ := filepath.Glob(filepath.Join(root, filepath.FromSlash(m.scenarios)))
	if len(scenarios) == 0 {
		return append(missing, fmt.Sprintf("no committed scenarios match %s", m.scenarios))
	}
	for _, sp := range scenarios {
		name := strings.TrimSuffix(filepath.Base(sp), ".json")
		var sc struct {
			Config struct {
				Connector struct {
					Name string `json:"name"`
				} `json:"connector"`
			} `json:"config"`
		}
		if err := readJSONFile(sp, &sc); err != nil || !contains(m.connectors, sc.Config.Connector.Name) {
			missing = append(missing, fmt.Sprintf("scenario %s does not run one of %v", name, m.connectors))
		}
		var g struct {
			Steps      []json.RawMessage `json:"steps"`
			Provenance struct {
				VLLMVersion    string `json:"vllm_version"`
				VLLMCommit     string `json:"vllm_commit"`
				ScenarioSHA256 string `json:"scenario_sha256"`
				Tier           int    `json:"tier"`
			} `json:"provenance"`
		}
		gp := filepath.Join(pkg, "testdata", "golden", name+".json")
		if err := readJSONFile(gp, &g); err != nil {
			missing = append(missing, fmt.Sprintf("scenario %s has no golden trace (%v)", name, err))
			continue
		}
		raw, _ := os.ReadFile(sp)
		sum := sha256.Sum256(raw)
		switch {
		case len(g.Steps) == 0:
			missing = append(missing, fmt.Sprintf("golden %s records no steps", name))
		case g.Provenance.VLLMVersion != blisvllm.Upstream.Version || g.Provenance.VLLMCommit != blisvllm.Upstream.Commit:
			missing = append(missing, fmt.Sprintf("golden %s is vLLM %s@%s, not %s@%s", name,
				g.Provenance.VLLMVersion, g.Provenance.VLLMCommit, blisvllm.Upstream.Version, blisvllm.Upstream.Commit))
		case g.Provenance.ScenarioSHA256 != hex.EncodeToString(sum[:]):
			missing = append(missing, fmt.Sprintf("golden %s was not produced from the committed scenario", name))
		case g.Provenance.Tier != m.tier:
			missing = append(missing, fmt.Sprintf("golden %s is a tier-%d trace, want tier %d", name, g.Provenance.Tier, m.tier))
		}
	}
	return missing
}

func readJSONFile(path string, v any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(raw, v)
}

func contains(xs []string, x string) bool {
	for _, y := range xs {
		if y == x {
			return true
		}
	}
	return false
}

// TestEveryEntryNamesAConnectorVLLMRegisters: a registry entry is a connector name a vLLM
// deployment can state, so it must be one vLLM v0.32.0's KVConnectorFactory registers
// (vllm/distributed/kv_transfer/kv_connector/factory.py@5df63a72). VLLM_SOURCE, when set (CI's
// packs job checks out the release), is read for the list; otherwise the names checked are
// those this module's diff tests run, which an invented entry also fails.
func TestEveryEntryNamesAConnectorVLLMRegisters(t *testing.T) {
	r, err := schemas.LoadConnectorRegistry("connectors.yaml")
	if err != nil {
		t.Fatal(err)
	}
	known := map[string]bool{}
	if src := os.Getenv("VLLM_SOURCE"); src != "" {
		raw, err := os.ReadFile(filepath.Join(src, "vllm", "distributed", "kv_transfer", "kv_connector", "factory.py"))
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range registered.FindAllStringSubmatch(string(raw), -1) {
			known[m[1]] = true
		}
		if len(known) == 0 {
			t.Fatal("no register_connector calls found in vLLM's factory.py")
		}
	} else {
		for _, m := range models {
			for _, c := range m.connectors {
				known[c] = true
			}
		}
	}
	for _, name := range r.Names() {
		if !known[name] {
			t.Errorf("%s is not a connector vLLM v0.32.0 registers", name)
		}
	}
}

var registered = regexp.MustCompile(`register_connector\(\s*"([A-Za-z0-9_]+)"`)

// A connector name that is not registered is not served, whatever role is asked for.
func TestAnUnregisteredConnectorIsNotServed(t *testing.T) {
	r, err := schemas.LoadConnectorRegistry("connectors.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if r.Serves("NotAConnector", connector.RolePDTransfer) {
		t.Fatal("an unregistered connector is served")
	}
}

// Load reads the embedded registry: NIXL is served for P/D transfer and nothing else, the
// OffloadingConnector for offload and nothing else.
func TestLoadServesTheRegisteredConnectors(t *testing.T) {
	r, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"NixlConnector", "NixlPullConnector"} {
		if !r.Serves(name, connector.RolePDTransfer) {
			t.Errorf("%s is not served for pd_transfer", name)
		}
		if r.Serves(name, connector.RoleOffload) {
			t.Errorf("%s is served for offload, which no diff test checks", name)
		}
	}
	if !r.Serves("OffloadingConnector", connector.RoleOffload) {
		t.Error("OffloadingConnector is not served for offload")
	}
	if r.Serves("OffloadingConnector", connector.RolePDTransfer) {
		t.Error("OffloadingConnector is served for pd_transfer, which no diff test checks")
	}
}

// A field the schema does not declare is refused, as blis-schemas' own loaders refuse it,
// even in a registry that is otherwise valid.
func TestLoadRefusesUnknownFields(t *testing.T) {
	entry := "kind: ConnectorRegistry\nname: x\nconnectors:\n  NixlConnector:\n    upstream: KVConnectorBase_V1\n" +
		"    behaves_like: nixl-pull\n    roles: [pd_transfer]\n    evidence: \"diff-test @ vLLM v0.32.0\"\n"
	if _, err := decode([]byte(entry)); err != nil {
		t.Fatalf("the valid registry is refused: %v", err)
	}
	if _, err := decode([]byte(entry + "    priority: 1\n")); err == nil {
		t.Fatal("a registry with an undeclared field decoded")
	}
}

// A registry that fails the schema's validation is refused.
func TestLoadRefusesAnInvalidRegistry(t *testing.T) {
	if _, err := decode([]byte("kind: ConnectorRegistry\nname: x\nconnectors: {}\n")); err == nil {
		t.Fatal("a registry with no connectors loaded")
	}
	// An interface vLLM does not define is refused here, where the release's interfaces are
	// known; blis-schemas checks only that an entry names one.
	entry := "kind: ConnectorRegistry\nname: x\nconnectors:\n  NixlConnector:\n    upstream: KVConnectorBase_V2\n" +
		"    behaves_like: nixl-pull\n    roles: [pd_transfer]\n    evidence: \"diff-test @ vLLM v0.32.0\"\n"
	if _, err := decode([]byte(entry)); err == nil ||
		!strings.Contains(err.Error(), `connectors.NixlConnector.upstream: "KVConnectorBase_V2" is not a known connector interface (KVConnectorBase_V1)`) {
		t.Errorf("an unknown upstream interface: %v", err)
	}
}

// The evidence check refuses what is not a trace: each case plants one defect in a scratch copy
// of a model's evidence (one scenario, its golden and a diff test), and earned must report it.
func TestEvidenceRefusesWhatIsNotATrace(t *testing.T) {
	scenario := []byte(`{"config": {"connector": {"name": "NixlConnector"}}}`)
	sum := sha256.Sum256(scenario)
	golden := func(version, commit, sha string, tier int, steps string) string {
		return fmt.Sprintf(`{"steps": %s, "provenance": {"vllm_version": %q, "vllm_commit": %q, "scenario_sha256": %q, "tier": %d}}`,
			steps, version, commit, sha, tier)
	}
	good := golden(blisvllm.Upstream.Version, blisvllm.Upstream.Commit, hex.EncodeToString(sum[:]), 4, `[{}]`)
	m := model{pkg: "core/x", scenarios: "oracle/scenarios/x/x-*.json", connectors: []string{"NixlConnector"}, tier: 4}
	for _, c := range []struct {
		name, golden, diffTest string
		scenario               []byte
		want                   string
	}{
		{"valid evidence", good, m.scenarios, scenario, ""},
		{"an empty object", `{}`, m.scenarios, scenario, "records no steps"},
		{"another release", golden("0.31.0", blisvllm.Upstream.Commit, hex.EncodeToString(sum[:]), 4, `[{}]`), m.scenarios, scenario, "is vLLM 0.31.0"},
		{"another scenario", golden(blisvllm.Upstream.Version, blisvllm.Upstream.Commit, "00", 4, `[{}]`), m.scenarios, scenario, "not produced from the committed scenario"},
		{"another tier", golden(blisvllm.Upstream.Version, blisvllm.Upstream.Commit, hex.EncodeToString(sum[:]), 2, `[{}]`), m.scenarios, scenario, "tier-2 trace"},
		{"no golden", "", m.scenarios, scenario, "has no golden trace"},
		{"a diff test that runs other scenarios", good, "oracle/scenarios/y/*.json", scenario, "does not run"},
		{"a scenario of another connector", good, m.scenarios, []byte(`{"config": {"connector": {"name": "LMCacheConnectorV1"}}}`), "does not run one of"},
	} {
		t.Run(c.name, func(t *testing.T) {
			root := t.TempDir()
			write := func(rel string, data []byte) {
				p := filepath.Join(root, filepath.FromSlash(rel))
				if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(p, data, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			write("oracle/scenarios/x/x-a.json", c.scenario)
			write("core/x/diff_test.go", []byte("package x // runs "+c.diffTest+"\n"))
			if c.golden != "" {
				write("core/x/testdata/golden/x-a.json", []byte(c.golden))
			}
			got := strings.Join(earned(root, m), "; ")
			if c.want == "" && got != "" {
				t.Fatalf("valid evidence refused: %s", got)
			}
			if c.want != "" && !strings.Contains(got, c.want) {
				t.Fatalf("got %q, want a refusal containing %q", got, c.want)
			}
		})
	}
}

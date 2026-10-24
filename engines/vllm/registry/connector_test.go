package registry

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/spec/connector"
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
// model an entry names must be one this module diff-tests against vLLM v0.32.0, with its
// golden traces present, and every connector so tested must be registered.
func TestEveryEntryIsEarnedByADiffTest(t *testing.T) {
	r, err := schemas.LoadConnectorRegistry("connectors.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range r.Names() {
		e := r.Connectors[name]
		pkg, ok := models[e.BehavesLike]
		if !ok {
			t.Errorf("%s behaves like %q, which no differential test in this module checks", name, e.BehavesLike)
			continue
		}
		goldens, _ := filepath.Glob(filepath.Join("..", filepath.FromSlash(pkg), "testdata", "golden", "*.json"))
		if len(goldens) == 0 {
			t.Errorf("%s behaves like %q, but %s has no golden traces from vLLM", name, e.BehavesLike, pkg)
		}
		if _, err := os.Stat(filepath.Join("..", filepath.FromSlash(pkg), "diff_test.go")); err != nil {
			t.Errorf("%s behaves like %q, but %s has no diff test", name, e.BehavesLike, pkg)
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

// Load reads the embedded registry: NIXL is served for P/D transfer and nothing else.
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
}

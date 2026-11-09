package cmd

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/result"
	"github.com/inference-sim/blis-substrate/load"
)

// TestPathFrom_Precedence is the location law of --catalog and --registry (design §3.3):
// the flag wins over the environment variable, either one replaces the embedded copy, and
// neither means the embedded copy (empty).
func TestPathFrom_Precedence(t *testing.T) {
	for _, tt := range []struct{ name, flag, env, want string }{
		{"flag only", "/from/flag", "", "/from/flag"},
		{"env only", "", "/from/env", "/from/env"},
		{"both: the flag wins", "/from/flag", "/from/env", "/from/flag"},
		{"neither: the embedded copy", "", "", ""},
	} {
		if got := pathFrom("catalog", tt.flag, catalogEnvVar, tt.env); got != tt.want {
			t.Errorf("%s: pathFrom(%q, %q) = %q, want %q", tt.name, tt.flag, tt.env, got, tt.want)
		}
	}
}

// TestDataOptions_ReadEachFlagAndVariable: the catalog and registry are located independently
// (SIM-6), each by its flag or its variable, and the overlays pass through in order.
func TestDataOptions_ReadEachFlagAndVariable(t *testing.T) {
	saved := []any{catalogPath, registryPath, catalogOverlays, registryOverlays}
	t.Cleanup(func() {
		catalogPath, registryPath = saved[0].(string), saved[1].(string)
		catalogOverlays, registryOverlays = saved[2].([]string), saved[3].([]string)
	})
	catalogPath, registryPath, catalogOverlays, registryOverlays = "", "", nil, nil
	t.Setenv(catalogEnvVar, "")
	t.Setenv(registryEnvVar, "/env/registry")
	if o := dataOptions(); o.Catalog != "" || o.Registry != "/env/registry" {
		t.Errorf("BLIS_REGISTRY alone: %+v, want the embedded catalog and the variable's registry", o)
	}
	catalogPath, catalogOverlays, registryOverlays = "/flag/catalog", []string{"/o1", "/o2"}, []string{"/r1"}
	o := dataOptions()
	if o.Catalog != "/flag/catalog" || strings.Join(o.CatalogOverlays, ",") != "/o1,/o2" ||
		strings.Join(o.RegistryOverlays, ",") != "/r1" {
		t.Errorf("flags: %+v", o)
	}
}

// TestRunData_EmbeddedByDefault: a run that names no catalog or registry reads the copy
// embedded in the pinned substrate release (SIM-6), and the Result's data record says so.
func TestRunData_EmbeddedByDefault(t *testing.T) {
	saved := []any{catalogPath, registryPath, catalogOverlays, registryOverlays, openedData}
	t.Cleanup(func() {
		catalogPath, registryPath = saved[0].(string), saved[1].(string)
		catalogOverlays, registryOverlays = saved[2].([]string), saved[3].([]string)
		openedData = saved[4].(*load.Data)
	})
	catalogPath, registryPath, catalogOverlays, registryOverlays, openedData = "", "", nil, nil, nil
	t.Setenv(catalogEnvVar, "")
	t.Setenv(registryEnvVar, "")
	d, err := runData()
	if err != nil {
		t.Fatal(err)
	}
	rec := resultData(d)
	if rec.Catalog.Source != result.SourceEmbedded || rec.Registry.Source != result.SourceEmbedded {
		t.Errorf("no path given: sources %s, %s; want embedded", rec.Catalog.Source, rec.Registry.Source)
	}
	if rec.Build != result.BuildUnreleased || rec.ReleasedData {
		t.Errorf("an untagged build: build %q, released_data %v; want unreleased, false", rec.Build, rec.ReleasedData)
	}

	// A path replaces the embedded copy, and the record names it a path with the same content
	// hash: the substrate module's catalog/ is the embedded copy's files (SUB-9).
	openedData = nil
	catalogPath = testCatalogDir()
	p, err := runData()
	if err != nil {
		t.Fatal(err)
	}
	pr := resultData(p)
	if pr.Catalog.Source != result.SourcePath || pr.Catalog.Hash != rec.Catalog.Hash {
		t.Errorf("--catalog %s: %+v; want source path with the embedded hash %s", catalogPath, pr.Catalog, rec.Catalog.Hash)
	}
}

// TestResultData_ReleasedOnlyFromATaggedEmbeddedBuild is X8's rule: released data is a tagged
// simulator release that read the embedded copy of both with no overlay.
func TestResultData_ReleasedOnlyFromATaggedEmbeddedBuild(t *testing.T) {
	saved := buildRelease
	t.Cleanup(func() { buildRelease = saved })
	emb, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	buildRelease = "v0.16.0"
	if r := resultData(emb); !r.ReleasedData || r.Build != "v0.16.0" {
		t.Errorf("tagged build on the embedded copy: %+v, want released", r)
	}
	path, err := load.Open(load.Options{Catalog: testCatalogDir()})
	if err != nil {
		t.Fatal(err)
	}
	if r := resultData(path); r.ReleasedData {
		t.Errorf("a path catalog is experimental: %+v", r)
	}
	buildRelease = "v0.16.0-rc.1"
	if r := resultData(emb); r.ReleasedData || r.Build != result.BuildUnreleased {
		t.Errorf("a pre-release tag is not a release: %+v", r)
	}
}

// TestRunData_RefusesAPathThatIsNotACatalog: a mistyped path is refused naming it, never
// replaced by the embedded copy.
func TestRunData_RefusesAPathThatIsNotACatalog(t *testing.T) {
	saved := []any{catalogPath, openedData}
	t.Cleanup(func() { catalogPath, openedData = saved[0].(string), saved[1].(*load.Data) })
	t.Setenv(catalogEnvVar, "")
	missing := filepath.Join(t.TempDir(), "no-such-catalog")
	catalogPath, openedData = missing, nil
	if _, err := runData(); err == nil || !strings.Contains(err.Error(), "no-such-catalog") {
		t.Errorf("--catalog %s: err = %v, want a refusal naming the path", missing, err)
	}
}

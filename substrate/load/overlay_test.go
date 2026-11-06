package load

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
)

const fabricOverlay = `kind: CoefficientSet
name: cost-model-fabric
coefficients:
  - fabric_bandwidth_efficiency:
      value: 0.9
      units: dimensionless
      method: assumed
      fitted: false
      scope: {fabric: [ethernet-100gbe]}
      rationale: an overlay's replacement
  - fabric_bandwidth_efficiency:
      value: 0.7
      units: dimensionless
      method: assumed
      fitted: false
      scope: {fabric: [ethernet-100gbe], model: [qwen3-14b]}
      rationale: an overlay's addition
`

// A registry overlay entry with the identity of an existing one (set, name, scope, method)
// replaces it; any other is added. The hashes record the overlay, and the combined data
// are checked in full.
func TestARegistryOverlayReplacesByIdentityAndAdds(t *testing.T) {
	ov := t.TempDir()
	write(t, filepath.Join(ov, "coefficients", "cost-model-fabric.yaml"), fabricOverlay)
	base, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	d, err := Open(Options{RegistryOverlays: []string{ov}})
	if err != nil {
		t.Fatal(err)
	}
	baseSet, _ := base.CoefficientSet("cost-model-fabric")
	set, _ := d.CoefficientSet("cost-model-fabric")
	if len(set.Coefficients) != len(baseSet.Coefficients)+1 {
		t.Fatalf("the overlay should add one entry: %d entries, the base has %d",
			len(set.Coefficients), len(baseSet.Coefficients))
	}
	if set.Coefficients[0].Value != 0.9 || set.Coefficients[0].Rationale != "an overlay's replacement" {
		t.Errorf("the same-identity entry was not replaced in place: %+v", set.Coefficients[0])
	}
	if baseSet.Coefficients[0].Value == 0.9 {
		t.Error("the overlay changed the base data")
	}
	if d.Registry.Source != SourceEmbedded || d.Registry.Hash != base.Registry.Hash ||
		len(d.Registry.Overlays) != 1 || d.Registry.Overlays[0].Path != ov ||
		d.Registry.CombinedHash == d.Registry.Hash || base.Registry.CombinedHash != base.Registry.Hash {
		t.Errorf("provenance: %+v (base %+v)", d.Registry, base.Registry)
	}
	if d.Registry.Overlays[0].Hash != Hash(map[string][]byte{"coefficients/cost-model-fabric.yaml": []byte(fabricOverlay)}) {
		t.Error("the overlay's hash is not the content hash of its files")
	}

	// An overlay entry the precedence cannot order against an existing one is refused once
	// combined (VAL-4): same name, no model on either, overlapping scope, different method.
	bad := strings.Replace(fabricOverlay, "      scope: {fabric: [ethernet-100gbe], model: [qwen3-14b]}\n      rationale: an overlay's addition",
		"      scope: {fabric: [ib-400g], tp: [8]}\n      rationale: an overlay's addition", 1)
	write(t, filepath.Join(ov, "coefficients", "cost-model-fabric.yaml"), bad)
	if _, err := Open(Options{RegistryOverlays: []string{ov}}); err == nil || !strings.Contains(err.Error(), "VAL-4") {
		t.Errorf("an unordered pair made by an overlay was not refused: %v", err)
	}
}

// A catalog overlay replaces a whole model directory, adds a new model, and may not hold
// ledger files. A registry that no longer fits the combined catalog is refused.
func TestACatalogOverlay(t *testing.T) {
	base, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	ov := t.TempDir()
	for _, f := range []string{"graph.yaml", "model.yaml"} {
		b, err := os.ReadFile(filepath.Join(repoDir, "catalog", "models", "qwen3-14b", f))
		if err != nil {
			t.Fatal(err)
		}
		body := strings.ReplaceAll(string(b), "qwen3-14b", "private-model")
		write(t, filepath.Join(ov, "models", "private-model", f), body)
	}
	d, err := Open(Options{CatalogOverlays: []string{ov}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := d.Model("private-model"); err != nil {
		t.Error(err)
	}
	if _, err := base.Model("private-model"); err == nil {
		t.Error("the base data gained the overlay's model")
	}
	if len(d.Models()) != len(base.Models())+1 {
		t.Errorf("models: %d, base %d", len(d.Models()), len(base.Models()))
	}

	// A model directory is replaced whole: an overlay's qwen3-14b with only a graph.yaml
	// leaves no model.yaml, and the combined catalog is refused.
	ov2 := t.TempDir()
	b, _ := os.ReadFile(filepath.Join(repoDir, "catalog", "models", "qwen3-14b", "graph.yaml"))
	write(t, filepath.Join(ov2, "models", "qwen3-14b", "graph.yaml"), string(b))
	if _, err := Open(Options{CatalogOverlays: []string{ov2}}); err == nil ||
		!strings.Contains(err.Error(), "models/qwen3-14b/: has a graph.yaml but no model.yaml") {
		t.Errorf("a model directory was not replaced whole: %v", err)
	}

	ov3 := t.TempDir()
	write(t, filepath.Join(ov3, "ledger", "published", "fit.yaml"), "units: []\n")
	if _, err := Open(Options{CatalogOverlays: []string{ov3}}); err == nil ||
		!strings.Contains(err.Error(), "ledger/published/fit.yaml: an overlay may not contain ledger files") {
		t.Errorf("an overlay holding ledger files was not refused: %v", err)
	}
	ov4 := t.TempDir()
	write(t, filepath.Join(ov4, "ledger", "x.yaml"), "x\n")
	if _, err := Open(Options{RegistryOverlays: []string{ov4}}); err == nil ||
		!strings.Contains(err.Error(), "an overlay may not contain ledger files") {
		t.Errorf("a registry overlay holding ledger files was not refused: %v", err)
	}
}

// The selected-entry report lists what lookups returned, by kind, and every registry file
// no reader loaded, for every reader: coefficients, acceptance and routing.
func TestTheSelectedEntryReport(t *testing.T) {
	d, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	r := d.Report()
	if len(r.Selected) != 0 || len(r.Unloaded) != 10 {
		t.Fatalf("a fresh load selects nothing and has loaded no file: %+v", r)
	}
	d.RecordCoefficient("cost-model-fabric", "fabric_bandwidth_efficiency", coefficient.Scope{Fabric: []string{"ib-400g"}}, "literature")
	acc := d.AcceptanceSets()
	if len(acc) != 1 {
		t.Fatalf("acceptance sets: %d", len(acc))
	}
	p := acc[0].Profiles[0]
	d.RecordAcceptance(&acceptance.Resolution{Set: acc[0].Name, Profile: &p})
	d.RecordAcceptance(nil)
	r = d.Report()
	if len(r.Selected) != 2 || r.Selected[0].Kind != KindAcceptance || r.Selected[1].Kind != KindCoefficient ||
		r.Selected[1].File != "coefficients/cost-model-fabric.yaml" || r.Selected[0].Entry != p.Key().String() {
		t.Errorf("selected: %+v", r.Selected)
	}
	for _, f := range r.Unloaded {
		if f == "coefficients/cost-model-fabric.yaml" || f == "acceptance/inferencex-golden-al.yaml" {
			t.Errorf("%s was read and is reported unloaded", f)
		}
	}
	if len(r.Unloaded) != 8 || r.Unloaded[len(r.Unloaded)-1] != "routing/servingstudio-dsv41-flash.yaml" {
		t.Errorf("unloaded: %v", r.Unloaded)
	}
}

// An overlay file whose path differs from a base path only in letter case would leave the
// base entry in force while the hash says the overlay applied: it is refused, naming both.
func TestAnOverlayPathThatDiffersOnlyInCaseIsRefused(t *testing.T) {
	b, err := os.ReadFile(filepath.Join(repoDir, "catalog", "hardware", "h100.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	ov := t.TempDir()
	write(t, filepath.Join(ov, "hardware", "H100.yaml"), strings.Replace(string(b), "MemoryGiB: 79.1895752", "MemoryGiB: 200", 1))
	_, err = Open(Options{CatalogOverlays: []string{ov}})
	if err == nil || !strings.Contains(err.Error(), "hardware/H100.yaml") || !strings.Contains(err.Error(), "hardware/h100.yaml") ||
		!strings.Contains(err.Error(), "letter case") {
		t.Errorf("a case-only collision with a base hardware file was not refused: %v", err)
	}
	// A file at the exact path still replaces.
	ov2 := t.TempDir()
	write(t, filepath.Join(ov2, "hardware", "h100.yaml"), strings.Replace(string(b), "MemoryGiB: 79.1895752", "MemoryGiB: 200", 1))
	d, err := Open(Options{CatalogOverlays: []string{ov2}})
	if err != nil {
		t.Fatal(err)
	}
	if c, err := d.Chip("h100"); err != nil || c.MemoryGiB != 200 {
		t.Errorf("an exact-path overlay did not replace: %v %v", c, err)
	}
	// Networks and workloads are merged by path too.
	for _, kind := range []string{"networks", "workloads"} {
		entries, err := os.ReadDir(filepath.Join(repoDir, "catalog", kind))
		if err != nil || len(entries) == 0 {
			t.Fatalf("%s: %v", kind, err)
		}
		name := entries[0].Name()
		body, err := os.ReadFile(filepath.Join(repoDir, "catalog", kind, name))
		if err != nil {
			t.Fatal(err)
		}
		o := t.TempDir()
		write(t, filepath.Join(o, kind, strings.ToUpper(strings.TrimSuffix(name, ".yaml"))+".yaml"), string(body))
		if _, err := Open(Options{CatalogOverlays: []string{o}}); err == nil || !strings.Contains(err.Error(), "letter case") {
			t.Errorf("%s: a case-only collision was not refused: %v", kind, err)
		}
	}
}

// An overlay that changes a set's role or caveats, which the merge cannot apply, is refused
// naming the field; one that repeats them loads.
func TestARegistryOverlayThatChangesSetLevelFieldsIsRefused(t *testing.T) {
	base, err := os.ReadFile(filepath.Join(repoDir, "registry", "routing", "servingstudio-dsv41-flash.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	src := string(base)
	if !strings.Contains(src, "role: sensitivity\n") {
		t.Fatal("the routing set no longer has role: sensitivity")
	}
	open := func(body string) error {
		ov := t.TempDir()
		write(t, filepath.Join(ov, "routing", "servingstudio-dsv41-flash.yaml"), body)
		_, err := Open(Options{RegistryOverlays: []string{ov}})
		return err
	}
	if err := open(src); err != nil {
		t.Fatalf("an overlay that repeats the set's role and caveats is refused: %v", err)
	}
	err = open(strings.Replace(src, "role: sensitivity\n", "role: default\n", 1))
	if err == nil || !strings.Contains(err.Error(), "role") {
		t.Errorf("an overlay that changes role was not refused: %v", err)
	}
	err = open(strings.Replace(src, "caveats:\n", "caveats:\n- an added caveat\n", 1))
	if err == nil || !strings.Contains(err.Error(), "caveats") {
		t.Errorf("an overlay that changes caveats was not refused: %v", err)
	}
}

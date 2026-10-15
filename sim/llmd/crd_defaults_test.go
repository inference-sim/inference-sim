package llmd

import (
	"bytes"
	"flag"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

var updateCRDs = flag.Bool("update-crds", false, "copy the router CRDs from the modules into crds/")

// crdSources are the modules and files the crds/ copies come from.
var crdSources = map[string]string{
	"llm-d.ai_inferenceobjectives.yaml":               "github.com/llm-d/llm-d-router",
	"llm-d.ai_inferencemodelrewrites.yaml":            "github.com/llm-d/llm-d-router",
	"inference.networking.k8s.io_inferencepools.yaml": "sigs.k8s.io/gateway-api-inference-extension",
}

// TestCRDsMatchModules: each embedded CRD is byte-identical to the one in the module version
// go.mod selects, so a router bump that changes a CRD's defaults is caught here. Regenerate with
// go test ./sim/llmd -run TestCRDsMatchModules -update-crds.
func TestCRDsMatchModules(t *testing.T) {
	for name, mod := range crdSources {
		out, err := exec.Command("go", "list", "-m", "-f", "{{.Dir}}", mod).Output()
		if err != nil {
			t.Fatalf("go list -m %s: %v", mod, err)
		}
		want, err := os.ReadFile(filepath.Join(strings.TrimSpace(string(out)), "config", "crd", "bases", name))
		if err != nil {
			t.Fatal(err)
		}
		if *updateCRDs {
			if err := os.WriteFile(filepath.Join("crds", name), want, 0o644); err != nil {
				t.Fatal(err)
			}
			continue
		}
		got, err := crdFiles.ReadFile("crds/" + name)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("crds/%s differs from %s's; rerun with -update-crds", name, mod)
		}
	}
	entries, _ := crdFiles.ReadDir("crds")
	if len(entries) != len(crdSources) {
		t.Errorf("crds/ holds %d files, crdSources names %d", len(entries), len(crdSources))
	}
}

// TestCRDDefaulter_AppliesAPIServerDefaults: an InferenceObjective poolRef without a group gets
// the CRD's default group, as admission to an API server gives it.
func TestCRDDefaulter_AppliesAPIServerDefaults(t *testing.T) {
	res, err := parseManifests(newScheme(), []byte(poolManifests), defaultNamespace)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.objectives) != 2 {
		t.Fatalf("parsed %d objectives, want 2", len(res.objectives))
	}
	for _, o := range res.objectives {
		ref := o.Spec.PoolRefs[0]
		if ref.Group != "inference.networking.k8s.io" || ref.Kind != "InferencePool" {
			t.Errorf("objective %s poolRef = %s/%s, want the CRD defaults inference.networking.k8s.io/InferencePool", o.Name, ref.Group, ref.Kind)
		}
	}
	if got := res.pools[0].Spec.EndpointPickerRef.Kind; got != "Service" {
		t.Errorf("InferencePool endpointPickerRef.kind = %q, want the CRD default Service", got)
	}
}

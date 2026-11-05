// Moved from blis-latency-kernel v0.5.0 (560bda6) fixtures_test.go with the fixtures
// (release plan §2). The walk starts at Root() instead of "testdata", and the loader is
// contracts.LoadBundle.

package kernelfixtures

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	blisschemas "github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
)

// Every committed fixture is a Scenario and the Deployment applied to it, two YAML
// documents in one file. These tests hold that shape to its contract.
//
// The contract is worth a test rather than a convention because the failure it prevents is
// silent. A fixture whose second document went missing, or whose pools landed in the
// scenario half, does not fail to parse — strict decoding would accept a scenario and then
// a kernel would fail far away, on a pool index that has no pool, with nothing naming the
// file. There are 688 of these and 679 are generated, so one generator bug is 679 bad
// fixtures.

// fixturePaths returns every committed scenario+deployment file.
func fixturePaths(t *testing.T) []string {
	t.Helper()
	var out []string
	err := filepath.WalkDir(Root(), func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		// Both spellings: a fixture written as .yml would otherwise be invisible to every
		// contract below, which is the silent kind of gap this file exists to close.
		if !d.IsDir() && (strings.HasSuffix(path, ".yaml") || strings.HasSuffix(path, ".yml")) {
			out = append(out, path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walking testdata: %v", err)
	}
	// A floor, not just non-empty. 679 of these are generated, so a glob or generator
	// regression that emits one fixture instead of 679 would satisfy every assertion below
	// while testing almost nothing — the "zero tests ran" failure in a subtler form. The
	// bound is deliberately loose: it catches a collapse without breaking on the ordinary
	// addition or removal of a handful of scenarios.
	const floor = 600
	if len(out) < floor {
		t.Fatalf("found %d fixtures, want at least %d: the corpus is 688 files, of which "+
			"679 are generated, so a count this low means a glob or the generator dropped "+
			"most of them rather than that scenarios were deleted", len(out), floor)
	}
	return out
}

// TestEveryFixtureHoldsAScenarioAndItsDeployment is the shape contract. It decodes with
// the same strictness the loaders use, so a misspelled key fails here too.
func TestEveryFixtureHoldsAScenarioAndItsDeployment(t *testing.T) {
	for _, path := range fixturePaths(t) {
		t.Run(rel(path), func(t *testing.T) {
			sc, dep, err := blisschemas.LoadBundle(path)
			if err != nil {
				t.Fatalf("loading: %v", err)
			}
			if sc.Kind != "Scenario" {
				t.Errorf("first document kind = %q, want Scenario", sc.Kind)
			}
			if dep.Kind != "Deployment" {
				t.Errorf("second document kind = %q, want Deployment", dep.Kind)
			}
			// The two halves describe one deployment, so they carry one name. A mismatch
			// means a file was assembled from two different sources.
			if sc.Name != dep.Name {
				t.Errorf("scenario name %q and deployment name %q disagree",
					sc.Name, dep.Name)
			}
			// The name is the filename: that is how a measurement row addresses a
			// deployment, so a disagreement makes a row point at something else.
			if stem := strings.TrimSuffix(filepath.Base(path), ".yaml"); sc.Name != stem {
				t.Errorf("name %q does not match the filename stem %q", sc.Name, stem)
			}
			if len(dep.Pools) == 0 {
				t.Error("no pools: there is nothing for a kernel to price")
			}
			// The hardware inventory belongs to the cluster. A fixture that omitted it
			// would send the loader looking for a chip named "".
			if sc.Cluster.Hardware == "" {
				t.Error("cluster.hardware is empty")
			}
			if sc.Cluster.Nodes <= 0 || sc.Cluster.GPUsPerNode <= 0 {
				t.Errorf("cluster declares nodes=%d gpus_per_node=%d; both must be positive",
					sc.Cluster.Nodes, sc.Cluster.GPUsPerNode)
			}
		})
	}
}

// TestNoFixtureKeepsPoolsInItsScenarioHalf is the regression guard for the split itself.
//
// `pools` under a Scenario is the pre-v0.2.0 shape. Strict decoding rejects it, so this
// cannot pass silently today — but it is asserted against the raw YAML rather than the
// typed struct so that the check keeps meaning if the field is ever re-added upstream, and
// so the failure names the file instead of surfacing as a decode error.
func TestNoFixtureKeepsPoolsInItsScenarioHalf(t *testing.T) {
	for _, path := range fixturePaths(t) {
		t.Run(rel(path), func(t *testing.T) {
			f, err := os.Open(path)
			if err != nil {
				t.Fatalf("opening: %v", err)
			}
			defer func() { _ = f.Close() }()

			var first map[string]any
			if err := yaml.NewDecoder(f).Decode(&first); err != nil {
				t.Fatalf("decoding the first document: %v", err)
			}
			for _, key := range []string{"pools", "hardware", "fabric"} {
				if _, found := first[key]; found {
					t.Errorf("the scenario half states %q; pools belong to the deployment "+
						"document and hardware/fabric inside cluster", key)
				}
			}
		})
	}
}

// TestFixtureDeploymentsFitTheirClusters runs the cross-document check that only exists
// once the documents are split: that a deployment's pools fill the cluster its scenario
// declares, and that each local data-parallel width divides a node.
//
// Nothing checked this before, because before the split there was one document and no
// cross-document check to run. It is the coupling most likely to drift, since a pool's node
// count and a cluster's are edited in different halves of the file.
func TestFixtureDeploymentsFitTheirClusters(t *testing.T) {
	for _, path := range fixturePaths(t) {
		t.Run(rel(path), func(t *testing.T) {
			sc, dep, err := blisschemas.LoadBundle(path)
			if err != nil {
				t.Fatalf("loading: %v", err)
			}
			problems := dep.ValidateAgainstCluster(deployment.ClusterConstraints{
				Nodes:       sc.Cluster.Nodes,
				GPUsPerNode: sc.Cluster.GPUsPerNode,
				Storage:     sc.Cluster.Storage,
			})
			if !problems.OK() {
				t.Errorf("deployment does not fit its cluster:\n%s", problems.Error())
			}
		})
	}
}

// TestFixturesValidateFieldByField runs each document's own field validation, which is the
// layer blis-schemas owns and this repository otherwise takes on trust.
func TestFixturesValidateFieldByField(t *testing.T) {
	for _, path := range fixturePaths(t) {
		t.Run(rel(path), func(t *testing.T) {
			sc, dep, err := blisschemas.LoadBundle(path)
			if err != nil {
				t.Fatalf("loading: %v", err)
			}
			if p := sc.Validate(); !p.OK() {
				t.Errorf("scenario:\n%s", p.Error())
			}
			if p := dep.Validate(); !p.OK() {
				t.Errorf("deployment:\n%s", p.Error())
			}
		})
	}
}

// rel names a fixture by its path under testdata/kernel/.
func rel(path string) string {
	r, err := filepath.Rel(Root(), path)
	if err != nil {
		return path
	}
	return r
}

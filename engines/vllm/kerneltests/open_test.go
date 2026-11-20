// Moved from blis-latency-kernel/open_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/inference-sim/substrate/kernel"

	"strings"
	"testing"
)

// TestOpenRefusesAnUnregisteredEngineVersion keeps the version coupling honest.
//
// The rules pack is the one the caller's rules source returns for the release the build runs
// (Repos.Release), so a release the caller has no rules for must be refused rather than priced
// with whatever pack happens to be current. The error names the versions that do exist,
// because the next question a reader has is which one to use.
func TestOpenRefusesAnUnregisteredEngineVersion(t *testing.T) {
	_, err := Open("minimax-m25-h200-ep8.yaml", withRules(Repos{
		Scenarios: "testdata", Data: testData, Release: "0.0.1-not-a-release",
	}))
	if err == nil {
		t.Fatal("a build running an unregistered engine release built a kernel")
	}
	for _, want := range []string{"0.0.1-not-a-release", "known versions"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

// TestNewRejectsADeploymentThatDoesNotFitItsCluster is the cross-document check that had no
// production caller.
//
// Before the v0.2.0 document split, Scenario carried Pools and one Validate covered this.
// After it, the check lives in deployment.ValidateAgainstCluster and needs the cluster
// supplied from the other document — which only this repository's fixture tests did, so an
// external caller could build a kernel for a deployment that does not fit its hardware.
//
// Both directions, because an under-filled cluster is as wrong as an over-filled one: a
// deployment lays out the whole cluster it is handed, so a pool count that does not sum to
// the declared nodes describes a placement no engine would run.
func TestNewRejectsADeploymentThatDoesNotFitItsCluster(t *testing.T) {
	for _, tc := range []struct {
		what      string
		poolNodes int
	}{
		{"pools claim more nodes than the cluster has", 4},
		{"pools claim fewer nodes than the cluster has", 1},
	} {
		t.Run(tc.what, func(t *testing.T) {
			in := fixtureInputs(t, "minimax-m25-h200-ep16.yaml") // 2 nodes
			in.Deployment.Pools[0].Nodes = tc.poolNodes

			_, err := newKernel(in)
			if err == nil {
				t.Fatalf("a deployment whose pools sum to %d nodes built a kernel against "+
					"a %d-node cluster", tc.poolNodes, in.Scenario.Cluster.Nodes)
			}
			if !strings.Contains(err.Error(), "do not validate") {
				t.Errorf("error %q does not read as a validation failure", err)
			}
		})
	}
}

// TestNewRejectsALocalDataParallelWidthThatDoesNotDivideANode is the second coupled check.
//
// dp_local is how many data-parallel ranks sit on one node, so a width that does not divide
// the node's GPU count describes a placement that cannot be realized. Distinct from the
// pool-fit check above and worth its own case: it reads a different pair of fields.
func TestNewRejectsALocalDataParallelWidthThatDoesNotDivideANode(t *testing.T) {
	in := fixtureInputs(t, "minimax-m25-h200-ep16.yaml") // 8 GPUs per node
	in.Deployment.Pools[0].Parallel.DPLocal = 3

	_, err := newKernel(in)
	if err == nil {
		t.Fatal("dp_local 3 built a kernel on 8-GPU nodes; 3 does not divide 8")
	}
	if !strings.Contains(err.Error(), "do not validate") {
		t.Errorf("error %q does not read as a validation failure", err)
	}
}

// TestNewStillRejectsTheCheapThingsFirst guards the ORDER of the checks.
//
// Validation runs over whole documents, so it must not become the thing that reports a nil
// pointer or an out-of-range index. Those have their own messages, they are cheaper, and
// they name the caller's mistake precisely; a validation failure in their place would send a
// reader looking at the documents for a defect that is in the Inputs struct.
func TestNewStillRejectsTheCheapThingsFirst(t *testing.T) {
	base := fixtureInputs(t, "minimax-m25-h200-ep8.yaml")

	t.Run("nil documents", func(t *testing.T) {
		in := base
		in.Deployment = nil
		_, err := newKernel(in)
		if err == nil {
			t.Fatal("a nil deployment built a kernel")
		}
		if !strings.Contains(err.Error(), "needs a scenario, a deployment") {
			t.Errorf("error %q is not the nil-document message", err)
		}
	})

	t.Run("pool index out of range", func(t *testing.T) {
		in := base
		in.PoolIndex = 7
		_, err := newKernel(in)
		if err == nil {
			t.Fatal("an out-of-range pool index built a kernel")
		}
		if !strings.Contains(err.Error(), "outside the deployment's") {
			t.Errorf("error %q is not the bounds message", err)
		}
	})

	t.Run("missing engine rules", func(t *testing.T) {
		in := base
		in.Rules = nil
		_, err := newKernel(in)
		if err == nil {
			t.Fatal("a nil rules pack built a kernel")
		}
		if !strings.Contains(err.Error(), "needs engine rules") {
			t.Errorf("error %q is not the rules message", err)
		}
	})
}

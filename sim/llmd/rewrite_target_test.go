package llmd

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// rewriteManifests is poolManifests plus an InferenceModelRewrite that sends every request for
// the scenario's model to target.
// rewriteManifests is poolManifests plus an InferenceModelRewrite that sends 99 parts in 100 of
// the requests for the scenario's model to stable and 1 part to canary.
func rewriteManifests(stable, canary string) string {
	return poolManifests + `---
apiVersion: llm-d.ai/v1alpha2
kind: InferenceModelRewrite
metadata:
  name: rewrite
spec:
  poolRef:
    name: vllm
  rules:
  - matches:
    - model:
        value: test-model
    targets:
    - modelRewrite: ` + stable + `
      weight: 99
    - modelRewrite: ` + canary + `
      weight: 1
`
}

// runRewrite builds the cluster and starts the router with no request in the run, so only a
// start-up refusal can fail it.
func runRewrite(t *testing.T, stable, canary string, adapters []string) error {
	t.Helper()
	cs := newCluster(testDeployment(2), cluster.NewSliceRequestSource(nil), nil)
	cs.SetGateway(New(Options{Model: "test-model", EPPFlags: []string{"--pool-name=vllm", "--config-text=" + loadOnlyConfig},
		Manifests: []byte(rewriteManifests(stable, canary)), Adapters: adapters,
		PodLabels: func(cluster.InstanceID, cluster.PoolRole) map[string]string { return map[string]string{"app": "vllm"} }}))
	return cs.Run()
}

// A model rewrite may name the scenario's model or an adapter the scenario declares. BLIS runs one
// model graph, so any other target would be priced as the base model without saying so: it is
// refused at start-up, before any request, naming the resource, the rule and the target (SIM-9).
// The check covers every target a rule lists, so a 99/1 canary to an undeclared model is refused
// whatever the seed and however few requests the run sends. Each refusal has its closest
// accepted case.
func TestModelRewrite_EveryTargetMustBeTheModelOrADeclaredAdapter(t *testing.T) {
	t.Run("the scenario's own model and a declared adapter are accepted", func(t *testing.T) {
		if err := runRewrite(t, "test-model", "adapter-a", []string{"adapter-a", "adapter-b"}); err != nil {
			t.Fatalf("a 99/1 split between the model and a declared adapter: %v", err)
		}
	})
	t.Run("a 1 percent canary to an undeclared model is refused at start-up", func(t *testing.T) {
		err := runRewrite(t, "test-model", "unknown-canary-model", nil)
		for _, want := range []string{"unknown-canary-model", "no adapters", `"rewrite"`, "rule 0"} {
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("a 1%% canary to an undeclared model, no request sent: %v, want a start-up refusal containing %q", err, want)
			}
		}
	})
	t.Run("an undeclared adapter in the main target, with others declared, is refused", func(t *testing.T) {
		err := runRewrite(t, "adapter-c", "adapter-a", []string{"adapter-a", "adapter-b"})
		if err == nil || !strings.Contains(err.Error(), "adapter-c") || !strings.Contains(err.Error(), "adapter-a, adapter-b") {
			t.Fatalf("a rewrite to an undeclared adapter: %v, want a refusal naming the target and the declared adapters", err)
		}
	})
}

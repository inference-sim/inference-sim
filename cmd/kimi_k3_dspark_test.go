package cmd

import (
	"os"
	"path/filepath"
	"testing"
)

// Kimi-K3 with a DSpark drafter starts: the drafter's MLA layers are sized as latents like the target's
// (MultiHeadLatentAttention, one latent a state, no V), so the memory model and the layer sizes agree.
func TestKimiK3WithADSparkDrafterRuns(t *testing.T) {
	const scenario = `kind: Scenario
name: k3-dspark
engine: {kind: vllm}
model: kimi-k3
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory]
cluster: {hardware: b200, fabric: ib-400g, nodes: 2, gpus_per_node: 8}
---
kind: Deployment
name: k3-dspark
pools:
  - role: colocated
    nodes: 2
    parallel: {tp: 8, pp: 1, dp: 2, enable_expert_parallel: true}
    engine:
      max_model_len: 32768
      speculative_config: {method: "dspark", num_speculative_tokens: 3, model: "kimi-k3-dspark"}
`
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "k3-dspark.yaml"), []byte(scenario), 0o644); err != nil {
		t.Fatal(err)
	}
	stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "k3-dspark.yaml", "--rate", "4", "--num-requests", "8",
		"--prompt-tokens", "256", "--output-tokens", "32", "--seed", "42", "--speculative-acceptance-length", "2.5")
	if err != nil {
		t.Fatalf("Kimi-K3 with a DSpark drafter must start: %v\n%s", err, lastLines(stderr, 4))
	}
	if !completedIs(stdout, 8) {
		t.Errorf("want 8 completed requests")
	}
}

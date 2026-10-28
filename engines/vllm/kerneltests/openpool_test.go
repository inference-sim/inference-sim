// Moved from blis-latency-kernel/openpool_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A disaggregated deployment: a prefill pool and a decode pool on one two-node cluster,
// differing in the parallelism that sets step time. Written to a temp dir rather than
// committed under testdata, because nothing else in this repository prices a disaggregated
// deployment and a committed fixture would be dead weight the fixture-contract tests then
// have to carry.
//
// The two pools differ in tensor-parallel width (8 against 4) specifically so a test can
// tell which one was priced: if OpenPool ignored its index, both kernels would resolve the
// same layout and the assertion below could not fail.
const disaggregated = `kind: Scenario
name: disagg-probe
engine_version: "0.29.0"

model: minimax-m2.5
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory]

cluster:
  hardware: h200
  fabric: ib-400g
  nodes: 2
  gpus_per_node: 8
---
kind: Deployment
name: disagg-probe

pools:
  - role: prefill
    nodes: 1
    parallel:
      tp: 8
      pp: 1
      dp: 1
    engine:
      quantization: fp8
      cache_dtype: fp8
      block_size: 16
      max_num_batched_tokens: 8192
  - role: decode
    nodes: 1
    parallel:
      tp: 4
      pp: 1
      dp: 2
    engine:
      quantization: fp8
      cache_dtype: fp8
      block_size: 16
      max_num_batched_tokens: 2048

# Required once the pools carry distinct roles: prefill computes a KV cache the decode pool
# must then read, so a disaggregated deployment that names no transfer describes two engines
# with no path between them. New's validation rejects the omission, which is how this
# fixture came to state it.
pd_transfer:
  connector: NixlConnector
`

func writeDisaggregated(t *testing.T) Repos {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(
		filepath.Join(dir, "disagg.yaml"), []byte(disaggregated), 0o644); err != nil {
		t.Fatal(err)
	}
	return withRules(Repos{Scenarios: dir, Catalog: catalogRoot, Registry: registryRoot})
}

// TestOpenPoolNamesTheFileWhenThePoolIsAbsent: the bounds error says which scenario.
//
// New reports "outside the deployment's N pool(s)", which is right when a caller handed it
// documents. A caller that handed OpenPool a FILENAME needs to know which of several
// scenarios lacks the pool it asked for, so the check is repeated here to name it.
func TestOpenPoolNamesTheFileWhenThePoolIsAbsent(t *testing.T) {
	r := writeDisaggregated(t)

	for _, idx := range []int{-1, 2, 99} {
		_, err := OpenPool("disagg.yaml", r, idx)
		if err == nil {
			t.Fatalf("pool %d built a kernel from a two-pool deployment", idx)
		}
		if !strings.Contains(err.Error(), "disagg.yaml") {
			t.Errorf("pool %d: error %q does not name the scenario", idx, err)
		}
		if !strings.Contains(err.Error(), "2 pool(s)") {
			t.Errorf("pool %d: error %q does not say how many pools there are", idx, err)
		}
	}
}

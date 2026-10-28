// Moved from blis-latency-kernel/openpool_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"os"
	"path/filepath"
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

// TestOpenPoolPricesTheRequestedPool is why OpenPool exists.
//
// Each pool of a disaggregated deployment runs its own engine with its own settings, so one
// kernel prices one pool. Pricing both from pool 0 would charge the decode pool the prefill
// pool's parallelism and the simulation would still run — a wrong number with nothing
// reporting it, which is the failure this repository is built to avoid.
//
// Asserted on the RESOLVED tensor-parallel width rather than on a step time: the width is
// the thing the index selects, and reading it says which pool was priced rather than merely
// that the two differ.
func TestOpenPoolPricesTheRequestedPool(t *testing.T) {
	r := writeDisaggregated(t)

	prefill, err := OpenPool("disagg.yaml", r, 0)
	if err != nil {
		t.Fatalf("pool 0: %v", err)
	}
	decode, err := OpenPool("disagg.yaml", r, 1)
	if err != nil {
		t.Fatalf("pool 1: %v", err)
	}

	if got := prefill.Resolved().ExpertParallelWidth; got < 1 {
		t.Errorf("prefill pool resolved no expert width: %d", got)
	}
	// tp 8 against tp 4: the layouts must not be the same kernel twice.
	if prefill.layout.TP != 8 {
		t.Errorf("pool 0 resolved TP=%d, want 8 — the prefill pool's width",
			prefill.layout.TP)
	}
	if decode.layout.TP != 4 {
		t.Errorf("pool 1 resolved TP=%d, want 4 — the decode pool's width; a kernel that "+
			"ignored the index would report 8", decode.layout.TP)
	}
}

// TestOpenPricesTheFirstPool pins Open's relationship to OpenPool.
//
// Open is OpenPool at zero, which is what a colocated deployment has. Stated as a test
// because the alternative reading — that Open picks a pool by role, or errors on a
// disaggregated deployment — is equally plausible from the name alone.
func TestOpenPricesTheFirstPool(t *testing.T) {
	r := writeDisaggregated(t)

	viaOpen, err := Open("disagg.yaml", r)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	viaPool, err := OpenPool("disagg.yaml", r, 0)
	if err != nil {
		t.Fatalf("OpenPool: %v", err)
	}
	if viaOpen.layout.TP != viaPool.layout.TP {
		t.Errorf("Open resolved TP=%d where OpenPool(0) resolved TP=%d; Open must be "+
			"OpenPool at zero", viaOpen.layout.TP, viaPool.layout.TP)
	}
}

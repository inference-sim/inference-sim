package cmd

import (
	"strings"
	"testing"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

// TestNetworkTopology_HasNoCLIFlag encodes the design decision behind the inter-node
// network cost (#1530): the node-span signal that decides whether a collective
// crosses a node boundary is DERIVED FROM PLACEMENT — which nodes an instance's GPUs
// actually occupy — and is never a user-declared knob.
//
// This matters for correctness, not just taste. A declared "GPUs per node" flag can
// contradict the real node_pools placement (say a bundle places a TP=16 instance on
// two 8-GPU nodes while the flag claims 16), and then the model charges a cross-node
// cost for a placement that never happened, or misses one that did. It also matters
// for run/replay parity: because the signal can only come from a PlacementManager,
// and `blis replay` rejects node_pools outright, a cross-node cost is unreachable
// from replay and needs no TraceV2 round-trip (INV-13).
//
// If a future change deliberately introduces a topology flag, it must also settle
// how the flag and the placement reconcile, and how replay reproduces it — then
// update this test with that reasoning.
func TestNetworkTopology_HasNoCLIFlag(t *testing.T) {
	// Names that would declare a topology rather than derive one.
	forbidden := []string{"gpus-per-node", "gpus_per_node", "node-span", "nodes-spanned", "network-topology"}

	for _, cmd := range []struct {
		name string
		c    *cobra.Command
	}{
		{"run", runCmd},
		{"replay", replayCmd},
		{"observe", observeCmd},
	} {
		cmd.c.Flags().VisitAll(func(f *pflag.Flag) {
			for _, bad := range forbidden {
				if strings.Contains(f.Name, bad) {
					t.Errorf("%s registers --%s: the inter-node topology (#1530) must be derived from "+
						"placement, not declared on the CLI; see this test's doc comment", cmd.name, f.Name)
				}
			}
		})
	}
}

// TestInterconnectCalibration_IsHardwareConfigOnly documents where the fabric SPEEDS come
// from, as distinct from the topology: they are per-GPU-type entries in the file
// --hardware-config already points at, so an IB-vs-RoCE comparison is an edit to that
// file. There is deliberately no separate bandwidth flag to drift from it.
//
// Note the per-GPU keying is not yet per-PLACED-GPU: a node-pool instance uses whichever
// entry --hardware resolved, because DeploymentConfig.HWConfigByGPU has no policy-bundle
// key today (issue #893). Fabric values therefore inherit that pre-existing limitation.
func TestInterconnectCalibration_IsHardwareConfigOnly(t *testing.T) {
	for _, cmd := range []struct {
		name string
		c    *cobra.Command
	}{
		{"run", runCmd},
		{"replay", replayCmd},
	} {
		if f := cmd.c.Flags().Lookup("hardware-config"); f == nil {
			t.Errorf("%s must expose --hardware-config: it is the only source of interconnect calibration (#1530)", cmd.name)
		}
		for _, bad := range []string{"inter-node-bandwidth", "intra-node-bandwidth", "interconnect-bandwidth"} {
			if f := cmd.c.Flags().Lookup(bad); f != nil {
				t.Errorf("%s registers --%s: interconnect bandwidths belong in the hardware config, "+
					"which is resolved per placed GPU type; a global flag cannot describe a mixed fleet", cmd.name, bad)
			}
		}
	}
}

// TestCrossNodeSpanForTrace verifies the normalization that keeps the trace header
// byte-identical for every run without multi-node placement: only a real span (>1) is
// recorded, and the omitempty zero covers both "no node pools" (0) and "every instance on
// one node" (1).
func TestCrossNodeSpanForTrace(t *testing.T) {
	for _, tc := range []struct {
		name           string
		maxNodes, want int
	}{
		{"no placement", 0, 0},
		{"single-node fleet", 1, 0},
		{"spans two", 2, 2},
		{"spans four", 4, 4},
		{"negative is treated as no placement", -3, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := crossNodeSpanForTrace(tc.maxNodes); got != tc.want {
				t.Errorf("crossNodeSpanForTrace(%d) = %d, want %d", tc.maxNodes, got, tc.want)
			}
		})
	}
}

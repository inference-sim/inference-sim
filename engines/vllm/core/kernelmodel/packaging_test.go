// Original to blis-vllm.

package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
)

// Placement reads the cluster's packaging as blis-substrate resolves it from the scenario
// and the chip's catalog entry (scenario.Cluster.Packaging, design SUB-3). The P/D + DP
// fixture states neither gpus_per_node nor gpus_per_rack, so its trays and racks are
// GB200 NVL72's 4 and 72. Its 19 trays span two racks. Its two decode ranks sit on trays 17
// and 18, either side of the rack boundary.
func TestPlacementReadsTheCatalogsPackaging(t *testing.T) {
	r := repos()
	r.Scenarios = withMNNVL(t)
	const fixture = "variant.yaml"
	dec, err := OpenRole(fixture, r, deployment.RoleDecode)
	if err != nil {
		t.Fatal(err)
	}
	if dec.id.gpusPerNode != 4 || dec.id.gpusPerRack != 72 {
		t.Fatalf("decode pool placed with %d GPUs per node and %d per rack, want the catalog's 4 and 72",
			dec.id.gpusPerNode, dec.id.gpusPerRack)
	}
	if pk := dec.Packaging(); pk != (scenario.Packaging{GPUsPerNode: 4, GPUsPerNodeSource: scenario.SourceCatalog,
		GPUsPerRack: 72, GPUsPerRackSource: scenario.SourceCatalog}) {
		t.Errorf("Packaging() = %+v, want the catalog's 4 and 72", pk)
	}
	a, b := dec.PlacementOf(0), dec.PlacementOf(1)
	if a.Node != 17 || a.Rack != 0 || b.Node != 18 || b.Rack != 1 {
		t.Errorf("decode ranks placed at %+v and %+v, want tray 17 in rack 0 and tray 18 in rack 1", a, b)
	}
	if _, _, declared := dec.EndpointPlacement(0); !declared {
		t.Error("the catalog's rack is not reported as declared, so no rack label reaches the router")
	}
}

// withMNNVL is the P/D + DP fixture with UCX_CUDA_IPC_ENABLE_MNNVL=y stated in both pools, the
// launch that lets NIXL hand KV over multi-node NVLink.
func withMNNVL(t *testing.T) string {
	t.Helper()
	dir := variant(t, "minimax-m2.5-gb200-1p1d-dp2.yaml", "      kv_cache_dtype: fp8\n",
		"      kv_cache_dtype: fp8\n      env: {UCX_CUDA_IPC_ENABLE_MNNVL: \"y\"}\n")
	body, err := os.ReadFile(filepath.Join(dir, "variant.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	// variant replaces the first match (the prefill pool); state the decode pool's too.
	i := strings.Index(string(body), "  - role: decode")
	head, tail := string(body[:i]), string(body[i:])
	tail = strings.Replace(tail, "      kv_cache_dtype: fp8\n", "      kv_cache_dtype: fp8\n      env: {UCX_CUDA_IPC_ENABLE_MNNVL: \"y\"}\n", 1)
	if err := os.WriteFile(filepath.Join(dir, "variant.yaml"), []byte(head+tail), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// NIXL hands KV over multi-node NVLink only when UCX_CUDA_IPC_ENABLE_MNNVL=y (and the KV is VMM
// registered); otherwise it uses RDMA across nodes, even inside one NVLink rack
// (docs/features/nixl_connector_usage.md:401@5df63a72). Without the flag the two decode trays that
// sit in one rack are therefore in different racks as far as a handoff is concerned, so the cost
// model prices it over the NIC.
func TestWithoutTheMNNVLFlagEveryNodeIsItsOwnRackForAHandoff(t *testing.T) {
	r := repos()
	r.Scenarios = "../../testdata/scenarios"
	dec, err := OpenRole("minimax-m2.5-gb200-1p1d-dp2.yaml", r, deployment.RoleDecode)
	if err != nil {
		t.Fatal(err)
	}
	a, b := dec.PlacementOf(0), dec.PlacementOf(1)
	if a.Node != 17 || b.Node != 18 || a.Rack == b.Rack || a.Rack != 17 || b.Rack != 18 {
		t.Errorf("decode ranks placed at %+v and %+v, want trays 17 and 18 in racks of their own", a, b)
	}
	pre, err := OpenRole("minimax-m2.5-gb200-1p1d-dp2.yaml", r, deployment.RolePrefill)
	if err != nil {
		t.Fatal(err)
	}
	// The same prefill-to-decode pair, priced with and without the flag: the NIC is slower than
	// the rack's NVLink, which is the point of stating the path.
	slow := dec.PDTransferTicks(8192, pre.PlacementOf(0), a)
	r2 := repos()
	r2.Scenarios = withMNNVL(t)
	pre2, err := OpenRole("variant.yaml", r2, deployment.RolePrefill)
	if err != nil {
		t.Fatal(err)
	}
	dec2, err := OpenRole("variant.yaml", r2, deployment.RoleDecode)
	if err != nil {
		t.Fatal(err)
	}
	fast := dec2.PDTransferTicks(8192, pre2.PlacementOf(0), dec2.PlacementOf(0))
	if slow <= 3*fast {
		t.Errorf("an 8192-token handoff costs %d ticks over the NIC and %d over the rack's NVLink; want the NIC at least 3 times dearer", slow, fast)
	}
	// The path that priced the handoff is disclosed, and differs between the two launches.
	nic, nvl := strings.Join(dec.EngineAssumptions(), "\n"), strings.Join(dec2.EngineAssumptions(), "\n")
	if nic == nvl || !strings.Contains(nic, envMNNVL) || !strings.Contains(nvl, envMNNVL) {
		t.Errorf("the handoff path is not disclosed apart for the two launches:\n%q\n%q", nic, nvl)
	}
}

// variant writes a copy of the committed scenario src, with old replaced by new once, as
// variant.yaml in a fresh directory beside a link to the fixture's router directory.
func variant(t *testing.T, src, old, new string) string {
	t.Helper()
	scenarios, err := filepath.Abs(filepath.Join("..", "..", "testdata", "scenarios"))
	if err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(filepath.Join(scenarios, src))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), old) {
		t.Fatalf("%s has no %q to replace", src, old)
	}
	dir := t.TempDir()
	if err := os.Symlink(filepath.Join(scenarios, "router"), filepath.Join(dir, "router")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "variant.yaml"), []byte(strings.Replace(string(body), old, new, 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

package deployment

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// A prefill pool runs NIXL beside a CPU tier (llm-d's MultiConnector) while the decode pool runs
// NIXL alone: each engine is a server with its own --kv-transfer-config, so the deployment
// states the connector and the tier per pool, and the deployment-level blocks are the default
// for a pool that states none.

const kvPoolsDoc = `kind: Deployment
name: d
pools:
  - role: prefill
    nodes: 1
    parallel: {tp: 1, pp: 1, dp: 8}
    engine: {}
%s
  - role: decode
    nodes: 1
    parallel: {tp: 1, pp: 1, dp: 8}
    engine: {}
%s
%s`

func kvDeployment(t *testing.T, prefill, decode, tail string) *Deployment {
	t.Helper()
	var d Deployment
	src := strings.Replace(strings.Replace(strings.Replace(kvPoolsDoc, "%s", prefill, 1), "%s", decode, 1), "%s", tail, 1)
	if err := yaml.Unmarshal([]byte(src), &d); err != nil {
		t.Fatal(err)
	}
	return &d
}

const cpuTier = "    offload: {tiers: [{tier: cpu_dram, bytes: 1099511627776}]}\n"

func TestPoolOffload_PrefillOnlyValidatesAndDecodeHasNone(t *testing.T) {
	d := kvDeployment(t,
		cpuTier,
		"    pd_transfer: {connector: NixlConnector}\n",
		"pd_transfer: {connector: MultiConnector}\n")
	if p := d.Validate(); !p.OK() {
		t.Fatalf("a CPU tier on the prefill pool only is refused:\n%s", p.Error())
	}
	if o := d.PoolOffload(0); o == nil || len(o.Tiers) != 1 || o.Tiers[0].Device != "cpu_dram" {
		t.Errorf("prefill pool offload = %+v, want the cpu_dram tier", o)
	}
	if o := d.PoolOffload(1); o != nil {
		t.Errorf("decode pool offload = %+v, want none", o)
	}
	if c := d.PoolPDTransfer(0).Connector; c != "MultiConnector" {
		t.Errorf("prefill connector %q, want the deployment's MultiConnector", c)
	}
	if c := d.PoolPDTransfer(1).Connector; c != "NixlConnector" {
		t.Errorf("decode connector %q, want its own NixlConnector", c)
	}
}

// The deployment-level blocks mean what they always meant: every pool runs them.
func TestPoolOffload_DeploymentLevelBlockIsTheDefault(t *testing.T) {
	d := kvDeployment(t, "", "",
		"offload: {tiers: [{tier: cpu_dram, bytes: 1}]}\npd_transfer: {connector: MultiConnector}\n")
	if p := d.Validate(); !p.OK() {
		t.Fatal(p.Error())
	}
	for i := range d.Pools {
		if d.PoolOffload(i) != d.Offload || d.PoolPDTransfer(i) != d.PDTransfer {
			t.Errorf("pool %d does not inherit the deployment-level blocks", i)
		}
	}
}

// An empty offload block on a pool states no offload there, whatever the default says.
func TestPoolOffload_EmptyBlockOptsOutOfTheDefault(t *testing.T) {
	d := kvDeployment(t, "", "    offload: {tiers: []}\n",
		"offload: {tiers: [{tier: cpu_dram, bytes: 1}]}\npd_transfer: {connector: MultiConnector}\n")
	if p := d.Validate(); !p.OK() {
		t.Fatal(p.Error())
	}
	if d.PoolOffload(0) == nil || d.PoolOffload(1) != nil {
		t.Errorf("prefill %+v decode %+v: want the default on prefill, none on decode", d.PoolOffload(0), d.PoolOffload(1))
	}
}

func TestPoolOffload_AmbiguousPoolBlocksAreRefused(t *testing.T) {
	for _, c := range []struct{ name, prefill, decode, tail, want string }{
		{"settings over no tier", "    offload: {tiers: [], connector: OffloadingConnector}\n", "",
			"pd_transfer: {connector: NixlConnector}\n", "pools[0].offload.tiers: an offload block with no tiers offloads nowhere"},
		{"pool tier without bytes", "    offload: {tiers: [{tier: cpu_dram, bytes: 0}]}\n", "",
			"pd_transfer: {connector: NixlConnector}\n", "pools[0].offload.tiers[0].bytes"},
		{"pool connector missing", "    pd_transfer: {}\n", "", "pd_transfer: {connector: NixlConnector}\n",
			"pools[0].pd_transfer.connector"},
		{"no connector anywhere", "", "", "", "pd_transfer: required for a disaggregated deployment"},
		{"one pool states, none to inherit", "    pd_transfer: {connector: MultiConnector}\n", "", "",
			"pools[1].pd_transfer"},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := kvDeployment(t, c.prefill, c.decode, c.tail).Validate()
			if p.OK() || !strings.Contains(p.Error(), c.want) {
				t.Errorf("want a refusal containing %q, got:\n%s", c.want, p.Error())
			}
		})
	}
	var d Deployment
	if err := yaml.Unmarshal([]byte("kind: Deployment\nname: d\npools:\n  - role: colocated\n    nodes: 1\n    parallel: {tp: 1, pp: 1, dp: 1}\n    engine: {}\n    pd_transfer: {connector: NixlConnector}\n"), &d); err != nil {
		t.Fatal(err)
	}
	if p := d.Validate(); p.OK() || !strings.Contains(p.Error(), "a colocated pool hands no KV off") {
		t.Errorf("a colocated pool's pd_transfer: %v", p.Error())
	}
}

// A pool's tier draws from the cluster's storage inventory like the deployment's.
func TestPoolOffload_PoolTierMustBeInTheClusterInventory(t *testing.T) {
	d := kvDeployment(t, "    offload: {tiers: [{tier: tape, bytes: 1}]}\n", "", "pd_transfer: {connector: NixlConnector}\n")
	p := d.ValidateAgainstCluster(ClusterConstraints{Nodes: 2, GPUsPerNode: 8, Storage: []string{"cpu_dram"}})
	if p.OK() || !strings.Contains(p.Error(), `pools[0].offload.tiers[0].tier: "tape" is not in the cluster storage inventory`) {
		t.Errorf("got %v", p.Error())
	}
}

// Offloads and PDTransfers list each block with the path it is stated at, a pool's empty offload
// block (no offload) left out.
func TestPoolOffload_ListsEveryStatedBlockWithItsPath(t *testing.T) {
	d := kvDeployment(t, cpuTier, "    offload: {tiers: []}\n    pd_transfer: {connector: NixlConnector}\n",
		"offload: {tiers: [{tier: cpu_dram, bytes: 1}]}\npd_transfer: {connector: MultiConnector}\n")
	var got []string
	for _, o := range d.Offloads() {
		got = append(got, o.At)
	}
	for _, t2 := range d.PDTransfers() {
		got = append(got, t2.At)
	}
	if want := "offload pools[0].offload pd_transfer pools[1].pd_transfer"; strings.Join(got, " ") != want {
		t.Errorf("blocks %v, want %s", got, want)
	}
}

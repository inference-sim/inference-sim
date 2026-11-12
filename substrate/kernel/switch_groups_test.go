package latencykernel

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/load"
)

// pdKernel builds the kernel of a TP8 prefill pool (8 nodes) and decode pool (4) on 12 H200 nodes (8 GPUs each) behind
// ib-400g. groups is the scenario's cluster lines for switch groups, empty for none.
func pdKernel(t *testing.T, groups string) *Kernel {
	t.Helper()
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	body := fmt.Sprintf(`kind: Scenario
name: switch-groups
model: muse-spark-open-816b-a42b-nvfp4
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory, cost-model-fabric]
engine_version: "0.32.0"
cluster:
  hardware: h200
  fabric: ib-400g
  nodes: 12
  gpus_per_node: 8
%s---
kind: Deployment
name: pd
pd_transfer: {connector: nixl}
pools:
  - role: prefill
    nodes: 8
    parallel: {tp: 8, pp: 1, dp: 1}
  - role: decode
    nodes: 4
    parallel: {tp: 8, pp: 1, dp: 1}
`, groups)
	r, name := openRepos(t, d, body)
	r.Rules = func(string) rules.EngineRules { return inertRules{} }
	r.EngineSettings = func(*deployment.Deployment, int) (enginesettings.Settings, error) {
		return enginesettings.Settings{BlockSize: 16, MaxNumSeqs: 64, MaxNumBatchedTokens: 8192,
			MaxModelLen: 8192}, nil
	}
	in, err := OpenInputs(name, r, 0)
	if err != nil {
		t.Fatal(err)
	}
	k, err := New(in)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

// Three groups of four nodes at ratio 4 give each uplink n x gpusPerNode x nic / (r x rankGPUs)
// = 4 x 8 x nic / (4 x 8) = nic: a lone pull is bound by its NIC exactly as without groups,
// and two pulls through one uplink get half of it each.
const fourByThree = `  switch_groups: [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
  uplink_oversubscription: 4
`

func at(node int) kernel.Placement { return kernel.Placement{Node: node, Rack: node} }

// sharedTime is how long demand i takes while every demand in the set is in flight, when
// each link divides its rate equally among the demands that name it. It is exact for demands
// of equal payload that start together, which is all these tests pose.
func sharedTime(set []kernel.TransferDemand, i int) time.Duration {
	users := map[kernel.LinkID]int{}
	for _, d := range set {
		for _, l := range d.Links {
			users[l.Link]++
		}
	}
	d := set[i]
	rate := math.Inf(1)
	for _, l := range d.Links {
		rate = math.Min(rate, l.BytesPerSecond/float64(users[l.Link]))
	}
	if t := time.Duration(float64(d.Bytes) / rate * float64(time.Second)); t > d.Floor {
		return t
	}
	return d.Floor
}

const tokens = 16384

// TWO CROSS-GROUP PULLS FROM ONE GROUP TAKE TWICE AS LONG AS ONE.
func TestCrossGroupPullsFromOneGroupSlowEachOther(t *testing.T) {
	k := pdKernel(t, fourByThree)
	a := k.PDTransferDemand(tokens, at(0), at(4))
	b := k.PDTransferDemand(tokens, at(1), at(5))
	alone := a.AloneTime()
	both := sharedTime([]kernel.TransferDemand{a, b}, 0)
	if ratio := float64(both) / float64(alone); math.Abs(ratio-2) > 0.001 {
		t.Errorf("two cross-group pulls from one group take %v each against %v alone (x%.3f); want x2",
			both, alone, ratio)
	}
}

// TWO PULLS INSIDE ONE GROUP DO NOT SLOW EACH OTHER, AND NAME NO UPLINK.
func TestInGroupPullsDoNotSlowEachOther(t *testing.T) {
	k := pdKernel(t, fourByThree)
	a := k.PDTransferDemand(tokens, at(0), at(2))
	b := k.PDTransferDemand(tokens, at(1), at(3))
	if got, want := sharedTime([]kernel.TransferDemand{a, b}, 0), a.AloneTime(); got != want {
		t.Errorf("two in-group pulls take %v each, %v alone; want equal", got, want)
	}
	for _, l := range a.Links {
		if len(l.Link) >= 6 && l.Link[:6] == "uplink" {
			t.Errorf("an in-group pull names %q", l.Link)
		}
	}
}

// A PULL ALONE IS PRICED EXACTLY AS WITHOUT GROUPS, AT ANY PLACEMENT.
func TestLonePullPricesAsWithoutGroups(t *testing.T) {
	plain, grouped := pdKernel(t, ""), pdKernel(t, fourByThree)
	for _, to := range []int{2, 4, 11} {
		if p, g := plain.PDTransferTime(tokens, at(0), at(to)), grouped.PDTransferTime(tokens, at(0), at(to)); p != g {
			t.Errorf("pull 0 to %d: %v without groups, %v with", to, p, g)
		}
	}
}

// Pulls between different pairs of groups share no uplink, and two pulls out of one group
// to the same group share both.
func TestUplinksAreNamedPerGroupPair(t *testing.T) {
	k := pdKernel(t, fourByThree)
	a := k.PDTransferDemand(tokens, at(0), at(4)) // group 0 to 1
	b := k.PDTransferDemand(tokens, at(5), at(8)) // group 1 to 2
	if got := sharedTime([]kernel.TransferDemand{a, b}, 0); got != a.AloneTime() {
		t.Errorf("pulls 0 to 1 and 1 to 2 take %v, %v alone; they share no uplink", got, a.AloneTime())
	}
	c := k.PDTransferDemand(tokens, at(0), at(8)) // group 0 to 2
	if got := sharedTime([]kernel.TransferDemand{a, c}, 0); got <= a.AloneTime() {
		t.Errorf("pulls 0 to 1 and 0 to 2 leave one group and take %v, %v alone; want them to share its uplink",
			got, a.AloneTime())
	}
}

// A heavily oversubscribed uplink binds a lone pull: r = 16 gives 1/4 of the NIC rate.
func TestOversubscribedUplinkBindsALonePull(t *testing.T) {
	k := pdKernel(t, `  switch_groups: [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
  uplink_oversubscription: 16
`)
	plain := pdKernel(t, "")
	same, cross := plain.PDTransferTime(tokens, at(0), at(2)), k.PDTransferTime(tokens, at(0), at(4))
	if ratio := float64(cross) / float64(same); math.Abs(ratio-4) > 0.01 {
		t.Errorf("a lone cross-group pull at r=16 takes %v against %v on a NIC (x%.3f); want x4", cross, same, ratio)
	}
	// The group size matters: a smaller group has a smaller uplink.
	small := pdKernel(t, `  switch_groups: [[0,1],[2,3,4,5,6,7,8,9,10,11]]
  uplink_oversubscription: 4
`)
	if got, want := small.PDTransferTime(tokens, at(0), at(2)), plain.PDTransferTime(tokens, at(0), at(2)); got <= want {
		t.Errorf("a pull out of a 2-node group at r=4 takes %v, no more than %v on a NIC; want the 2-node uplink to bind", got, want)
	}
}

package latencykernel

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/conformance/kerneltest"
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
	k, err := pdPool(t, groups, "tp: 8, pp: 1, dp: 1", 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

// pdPool is the kernel of pool idx of the same deployment, whose decode pool has the layout
// decode, and the error New refuses it with. first is where the pool starts, nil if unstated.
func pdPool(t *testing.T, groups, decode string, idx int, first *int) (*Kernel, error) {
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
    parallel: {%s}
`, groups, decode)
	r, name := openRepos(t, d, body)
	r.Rules = func(string) rules.EngineRules { return inertRules{} }
	r.EngineSettings = func(*deployment.Deployment, int) (enginesettings.Settings, error) {
		return enginesettings.Settings{BlockSize: 16, MaxNumSeqs: 64, MaxNumBatchedTokens: 8192,
			MaxModelLen: 8192}, nil
	}
	in, err := OpenInputs(name, r, idx)
	if err != nil {
		t.Fatal(err)
	}
	in.PoolFirstNode = first
	return New(in)
}

// Three groups of four nodes at ratio 4 give each uplink n x gpusPerNode x nic / (r x rankGPUs)
// = 4 x 8 x nic / (4 x 8) = nic: a lone pull is bound by its NIC exactly as without groups,
// and two pulls through one uplink get half of it each.
const fourByThree = `  switch_groups: [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
  uplink_oversubscription: 4
`

func at(node int) kernel.Placement { return kernel.Placement{Node: node, Rack: node} }

// sharedTime is how long demand i takes while every demand in the set is in flight, when
// each link divides its rate equally among the streams that cross it. It is exact for demands
// of equal payload that start together, which is all these tests pose.
func sharedTime(set []kernel.TransferDemand, i int) time.Duration {
	users := map[kernel.LinkID]int{}
	for _, d := range set {
		for _, l := range d.Links {
			users[l.Link] += max(l.Streams, 1)
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

// A PHYSICAL UPLINK HAS ONE RATE IN EVERY KERNEL, WHATEVER THE POOL'S WIDTH.
// The prefill pool's instance is 8 GPUs and the decode pool's is 32 (tp 16 x dp 2 on four
// nodes); both name group 0's and group 1's uplinks, and must name them at the same rate.
func TestOneUplinkHasOneRateAcrossPoolsOfDifferentWidths(t *testing.T) {
	narrow, err := pdPool(t, fourByThree, "tp: 8, pp: 1, dp: 4", 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	wide, err := pdPool(t, fourByThree, "tp: 16, pp: 1, dp: 2", 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []kerneltest.PDCase{
		{Tokens: tokens, From: at(0), To: at(4)},
		{Tokens: tokens, From: at(5), To: at(8)},
	} {
		if err := kerneltest.CheckLinksAgree(narrow, wide, c); err != nil {
			t.Error(err)
		}
	}
	dn, dw := narrow.PDTransferDemand(tokens, at(0), at(4)), wide.PDTransferDemand(tokens, at(0), at(4))
	if dn.Links[2].Streams == dw.Links[2].Streams {
		t.Errorf("both pools state %d streams on %s; the instances pull with 8 and 32 ranks",
			dn.Links[2].Streams, dn.Links[2].Link)
	}
}

// PULLS SHARE AN UPLINK IN PROPORTION TO THE RANKS THEY PULL WITH.
// Two pulls leave group 0 together: one by an 8-rank instance and one by a 32-rank one. The
// uplink is shared by 40 rank flows, so each flow gets 1/40 of it: the narrow pull, bound alone
// by its NIC (the uplink gives its 8 flows a NIC each), takes 40/8 = 5 times as long, and the
// wide pull, bound alone by 1/32 of the uplink, takes 40/32 times as long.
func TestPullsShareAnUplinkInProportionToTheirRanks(t *testing.T) {
	narrow, _ := pdPool(t, fourByThree, "tp: 8, pp: 1, dp: 4", 0, nil)
	wide, err := pdPool(t, fourByThree, "tp: 16, pp: 1, dp: 2", 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	a := narrow.PDTransferDemand(tokens, at(0), at(4))
	b := wide.PDTransferDemand(tokens, at(1), at(5))
	set := []kernel.TransferDemand{a, b}
	for i, want := range []float64{40.0 / 8, 40.0 / 32} {
		got := float64(sharedTime(set, i)) / float64(set[i].AloneTime())
		if math.Abs(got-want) > 0.01 {
			t.Errorf("pull %d takes x%.3f its alone time sharing the uplink; want x%.3f", i, got, want)
		}
	}
}

// A POOL WHOSE COLLECTIVE WOULD CROSS A SWITCH GROUP IS REFUSED, NOT PRICED AT THE NODE RATE.
// The decode pool is four nodes of 32-GPU instances; pools take nodes in declaration order, so it
// starts at node 8 unless the caller says otherwise.
func TestAPoolWhoseCollectiveCrossesGroupsIsRefused(t *testing.T) {
	zero, two := 0, 2
	const wideDecode = "tp: 16, pp: 1, dp: 2"
	const split10 = `  switch_groups: [[0,1,2,3,4,5,6,7,8,9],[10,11]]
  uplink_oversubscription: 4
`
	for _, c := range []struct {
		name   string
		groups string
		first  *int
		want   string // the rule New refuses with, empty to accept
	}{
		{"four nodes at 8 to 11 inside one group", fourByThree, nil, ""},
		{"four nodes at 8 to 11 straddling groups 0-9 and 10-11", split10, nil, "switch_group_collective_crosses"},
		{"four nodes stated to start at 2, straddling two groups", fourByThree, &two, "switch_group_collective_crosses"},
		{"no groups: the collective stays on the node rate", "", &two, ""},
		{"one group holds the cluster", `  switch_groups: [[0,1,2,3,4,5,6,7,8,9,10,11]]
  uplink_oversubscription: 1
`, &zero, ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := pdPool(t, c.groups, wideDecode, 1, c.first)
			switch {
			case c.want == "" && err != nil:
				t.Errorf("refused: %v", err)
			case c.want != "" && (err == nil || !strings.Contains(err.Error(), c.want)):
				t.Errorf("err %v, want %s", err, c.want)
			}
		})
	}
}

// An instance that fits on a node has no collective to cross a boundary, wherever the pool
// sits and whether or not it states where.
func TestASingleNodeInstanceNeedsNoPlacement(t *testing.T) {
	if _, err := pdPool(t, fourByThree, "tp: 8, pp: 1, dp: 1", 1, nil); err != nil {
		t.Errorf("a TP8 pool on 8-GPU nodes was refused: %v", err)
	}
}

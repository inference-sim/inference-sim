package kerneltest

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
)

// fake prices a handoff the way a real kernel does: a floor plus a payload over the NIC
// across nodes, or over NVLink on one node. skew, when set, breaks the law by charging a
// different time from the demand it reports.
type fake struct {
	kernel.Kernel // unused methods panic, which these tests never reach
	skew          time.Duration
}

func (f fake) PDTransferDemand(tokens int, from, to kernel.Placement) kernel.TransferDemand {
	if tokens <= 0 {
		return kernel.TransferDemand{}
	}
	d := kernel.TransferDemand{Bytes: int64(tokens) * 70272, Floor: 20 * time.Microsecond}
	if from.Node == to.Node {
		d.Links = []kernel.LinkUse{{Link: kernel.LinkID(fmt.Sprintf("node%d/nvlink", from.Node)), BytesPerSecond: 450e9}}
		return d
	}
	d.Links = []kernel.LinkUse{
		{Link: kernel.LinkID(fmt.Sprintf("node%d/nic/egress", from.Node)), BytesPerSecond: 50e9},
		{Link: kernel.LinkID(fmt.Sprintf("node%d/nic/ingress", to.Node)), BytesPerSecond: 50e9},
	}
	return d
}

func (f fake) PDTransferTime(tokens int, from, to kernel.Placement) time.Duration {
	return f.PDTransferDemand(tokens, from, to).AloneTime() + f.skew
}

func TestLawHoldsForAConformingKernel(t *testing.T) {
	TestPDTransfer(t, fake{}, PDCases())
}

func TestLawFailsForADivergingKernel(t *testing.T) {
	err := CheckPDTransfer(fake{skew: time.Nanosecond},
		PDCase{Tokens: 4096, From: kernel.Placement{Pool: deployment.RolePrefill},
			To: kernel.Placement{Node: 1, Pool: deployment.RoleDecode}})
	if err == nil || !strings.Contains(err.Error(), "PDTransferTime") {
		t.Fatalf("a kernel charging 1ns more than its demand passed: %v", err)
	}
}

// relinked names its NIC differently, as a kernel would that derived IDs from something
// other than the placements.
type relinked struct{ fake }

func (r relinked) PDTransferDemand(tokens int, from, to kernel.Placement) kernel.TransferDemand {
	d := r.fake.PDTransferDemand(tokens, from, to)
	for i := range d.Links {
		d.Links[i].Link = "other/" + d.Links[i].Link
	}
	return d
}

func TestLinksAgree(t *testing.T) {
	for _, c := range PDCases() {
		if err := CheckLinksAgree(fake{}, fake{skew: 1}, c); err != nil {
			t.Errorf("two kernels over one fabric: %v", err)
		}
	}
	c := PDCase{Tokens: 4096, To: kernel.Placement{Node: 1}}
	if CheckLinksAgree(fake{}, relinked{}, c) == nil {
		t.Error("kernels naming different links passed")
	}
}

func TestCasesCoverEveryLinkClass(t *testing.T) {
	var sameNode, sameRack, crossRack bool
	for _, c := range PDCases() {
		switch {
		case c.From.Node == c.To.Node:
			sameNode = true
		case c.From.Rack == c.To.Rack:
			sameRack = true
		default:
			crossRack = true
		}
	}
	if !sameNode || !sameRack || !crossRack {
		t.Errorf("cases miss a link class: same node %v, same rack %v, cross rack %v",
			sameNode, sameRack, crossRack)
	}
}

// doubled names one link twice, which an arbiter would count twice.
type doubled struct{ fake }

func (d doubled) PDTransferDemand(tokens int, from, to kernel.Placement) kernel.TransferDemand {
	dem := d.fake.PDTransferDemand(tokens, from, to)
	if len(dem.Links) > 0 {
		dem.Links = append(dem.Links, dem.Links[0])
	}
	return dem
}

// drifting answers differently on every call, which no pure kernel may.
type drifting struct {
	fake
	n *int64
}

func (d drifting) PDTransferDemand(tokens int, from, to kernel.Placement) kernel.TransferDemand {
	*d.n++
	dem := d.fake.PDTransferDemand(tokens, from, to)
	dem.Bytes += *d.n
	return dem
}

func (d drifting) PDTransferTime(tokens int, from, to kernel.Placement) time.Duration {
	return d.PDTransferDemand(tokens, from, to).AloneTime()
}

// Each way the check can fail does fail, so a kernel that breaks the contract cannot pass
// conformance by a path the check never exercised.
func TestCheckFailsEachWay(t *testing.T) {
	c := PDCase{Tokens: 4096, From: kernel.Placement{Pool: deployment.RolePrefill},
		To: kernel.Placement{Node: 1, Pool: deployment.RoleDecode}}
	var n int64
	for name, k := range map[string]kernel.Kernel{
		"malformed demand":  doubled{},
		"different demands": drifting{n: &n},
		"PDTransferTime":    fake{skew: time.Nanosecond},
	} {
		if err := CheckPDTransfer(k, c); err == nil || !strings.Contains(err.Error(), name) {
			t.Errorf("%s: got %v", name, err)
		}
	}
}

// recorder captures a fatal call without stopping the test that checks for it.
type recorder struct {
	testing.TB
	fatal bool
}

func (r *recorder) Helper()               {}
func (r *recorder) Fatal(...any)          { r.fatal = true }
func (r *recorder) Error(...any)          {}
func (r *recorder) Errorf(string, ...any) {}

func TestNoCasesIsAFailure(t *testing.T) {
	r := &recorder{TB: t}
	TestPDTransfer(r, fake{}, nil)
	if !r.fatal {
		t.Error("TestPDTransfer with no cases passed, having checked nothing")
	}
}

// floorOnly puts the whole price in the floor and names no link, which leaves an arbiter
// nothing to share; it must not pass conformance.
type floorOnly struct{ fake }

func (f floorOnly) PDTransferDemand(tokens int, from, to kernel.Placement) kernel.TransferDemand {
	d := f.fake.PDTransferDemand(tokens, from, to)
	d.Floor, d.Links = d.AloneTime(), nil
	return d
}

func (f floorOnly) PDTransferTime(tokens int, from, to kernel.Placement) time.Duration {
	return f.PDTransferDemand(tokens, from, to).AloneTime()
}

// shrinking moves fewer bytes for more tokens past a threshold.
type shrinking struct{ fake }

func (s shrinking) PDTransferDemand(tokens int, from, to kernel.Placement) kernel.TransferDemand {
	if tokens > 4096 {
		tokens = 1
	}
	return s.fake.PDTransferDemand(tokens, from, to)
}

func (s shrinking) PDTransferTime(tokens int, from, to kernel.Placement) time.Duration {
	return s.PDTransferDemand(tokens, from, to).AloneTime()
}

func TestConformanceRefusesUnarbitrableAndShrinkingDemands(t *testing.T) {
	var malformed bool
	for _, c := range PDCases() {
		if err := CheckPDTransfer(floorOnly{}, c); err != nil && strings.Contains(err.Error(), "cross no link") {
			malformed = true
		}
	}
	if !malformed {
		t.Error("a kernel pricing everything as floor with no links passed")
	}
	if err := CheckPayloadGrows(shrinking{}, PDCases()); err == nil {
		t.Error("a kernel moving fewer bytes for more tokens passed")
	}
	if err := CheckPayloadGrows(fake{}, PDCases()); err != nil {
		t.Errorf("a conforming kernel failed: %v", err)
	}
	// CheckLinksAgree now refuses a malformed demand from either kernel.
	if err := CheckLinksAgree(fake{}, doubled{}, PDCase{Tokens: 4096, To: kernel.Placement{Node: 1}}); err == nil {
		t.Error("CheckLinksAgree accepted a demand naming a link twice")
	}
}

// Each way a pipeline split can break the StepEstimate contract is caught, and a conforming
// split and no split at all pass.
func TestCheckStagesFailsEachWay(t *testing.T) {
	us := time.Microsecond
	ok := kernel.StepEstimate{Overlap: 7 * us, NoOverlap: 10 * us,
		Stages: []kernel.StageEstimate{{Overlap: 3 * us, NoOverlap: 4 * us}, {Overlap: 4 * us, NoOverlap: 6 * us}}}
	if err := CheckStages(ok); err != nil {
		t.Errorf("a conforming split failed: %v", err)
	}
	if err := CheckStages(kernel.StepEstimate{Overlap: us, NoOverlap: us}); err != nil {
		t.Errorf("a step with no split failed: %v", err)
	}
	for name, e := range map[string]kernel.StepEstimate{
		"one stage":               {Overlap: us, NoOverlap: us, Stages: []kernel.StageEstimate{{Overlap: us, NoOverlap: us}}},
		"overlap does not sum":    {Overlap: 8 * us, NoOverlap: 10 * us, Stages: ok.Stages},
		"no-overlap does not sum": {Overlap: 7 * us, NoOverlap: 11 * us, Stages: ok.Stages},
		"a negative stage": {Overlap: 0, NoOverlap: 0,
			Stages: []kernel.StageEstimate{{Overlap: -us, NoOverlap: -us}, {Overlap: us, NoOverlap: us}}},
		"a stage that overlaps above its sum": {Overlap: 3 * us, NoOverlap: 3 * us,
			Stages: []kernel.StageEstimate{{Overlap: 2 * us, NoOverlap: us}, {Overlap: us, NoOverlap: 2 * us}}},
	} {
		if CheckStages(e) == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

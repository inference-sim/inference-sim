// Package kerneltest holds checks of the kernel.Kernel contract that any implementation
// can run against itself, in the way testing/fstest checks an fs.FS.
//
// A law stated only in a doc comment is kept by whoever reads it. Shipping it as code
// means every implementation, and every fake a simulator's tests build, is held to the
// same reading of it.
package kerneltest

import (
	"fmt"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// PDCase is one prefill-to-decode handoff to check.
type PDCase struct {
	Tokens   int
	From, To kernel.Placement
}

func (c PDCase) String() string {
	return fmt.Sprintf("%d tokens from %+v to %+v", c.Tokens, c.From, c.To)
}

// PDCases returns handoffs covering the three link classes a placement pair can select
// (same node, same rack on another node, another rack) at token counts from none to a
// long context. An implementation adds cases for any boundary its fabric model has
// beyond these.
func PDCases() []PDCase {
	from := kernel.Placement{Node: 0, Rack: 0, Pool: deployment.RolePrefill}
	tos := []kernel.Placement{
		{Node: 0, Rack: 0, Pool: deployment.RoleDecode},
		{Node: 1, Rack: 0, Pool: deployment.RoleDecode},
		{Node: 2, Rack: 1, Pool: deployment.RoleDecode},
	}
	var out []PDCase
	for _, to := range tos {
		for _, t := range []int{0, 1, 17, 4096, 131072} {
			out = append(out, PDCase{Tokens: t, From: from, To: to})
		}
	}
	return out
}

// CheckPDTransfer checks one case against the PDTransferDemand contract: the demand is
// well formed, asking for it twice gives the same answer, and alone it costs exactly
// what PDTransferTime charges.
func CheckPDTransfer(k kernel.Kernel, c PDCase) error {
	d := k.PDTransferDemand(c.Tokens, c.From, c.To)
	if err := d.Check(); err != nil {
		return fmt.Errorf("%v: malformed demand: %w", c, err)
	}
	if again := k.PDTransferDemand(c.Tokens, c.From, c.To); !reflect.DeepEqual(d, again) {
		return fmt.Errorf("%v: two calls gave different demands, %+v and %+v", c, d, again)
	}
	got, want := d.AloneTime(), k.PDTransferTime(c.Tokens, c.From, c.To)
	if got != want {
		return fmt.Errorf("%v: demand %+v costs %v alone but PDTransferTime is %v",
			c, d, got, want)
	}
	return nil
}

// CheckLinksAgree checks that two kernels built from one deployment, such as its prefill
// and decode kernels, describe one handoff alike: both demands are well formed and name
// the same links at the same rates. Without it a simulator could not tell that a
// transfer one kernel priced shares a link with one the other priced. It checks one
// handoff; that a physical link keeps its ID across different handoffs is the kernel's to
// honour, since only it knows which handoffs cross which links.
func CheckLinksAgree(a, b kernel.Kernel, c PDCase) error {
	da := a.PDTransferDemand(c.Tokens, c.From, c.To)
	db := b.PDTransferDemand(c.Tokens, c.From, c.To)
	for i, d := range []kernel.TransferDemand{da, db} {
		if err := d.Check(); err != nil {
			return fmt.Errorf("%v: kernel %d's demand is malformed: %w", c, i+1, err)
		}
	}
	key := func(ls []kernel.LinkUse) map[kernel.LinkID]float64 {
		m := make(map[kernel.LinkID]float64, len(ls))
		for _, l := range ls {
			m[l.Link] = l.BytesPerSecond
		}
		return m
	}
	if !reflect.DeepEqual(key(da.Links), key(db.Links)) {
		return fmt.Errorf("%v: the two kernels name different links, %v and %v", c, da.Links, db.Links)
	}
	return nil
}

// TestPDTransfer runs CheckPDTransfer over cases, reporting every failure. An empty list
// fails, because a conformance check that checked nothing would read as one that passed.
func TestPDTransfer(t testing.TB, k kernel.Kernel, cases []PDCase) {
	t.Helper()
	if len(cases) == 0 {
		t.Fatal("kerneltest.TestPDTransfer: no cases, so nothing was checked; pass PDCases()")
	}
	for _, c := range cases {
		if err := CheckPDTransfer(k, c); err != nil {
			t.Error(err)
		}
	}
	if err := CheckPayloadGrows(k, cases); err != nil {
		t.Error(err)
	}
}

// CheckPayloadGrows checks that, between one pair of placements, moving more tokens never
// moves fewer bytes: KV grows with the tokens it holds. It deliberately does not require
// the floor to stay fixed as tokens grow, because per-transfer overheads can grow with
// the number of pages a transfer moves.
func CheckPayloadGrows(k kernel.Kernel, cases []PDCase) error {
	type pair struct{ from, to kernel.Placement }
	byPair := map[pair][]PDCase{}
	for _, c := range cases {
		byPair[pair{c.From, c.To}] = append(byPair[pair{c.From, c.To}], c)
	}
	for _, cs := range byPair {
		sort.Slice(cs, func(i, j int) bool { return cs[i].Tokens < cs[j].Tokens })
		for i := 1; i < len(cs); i++ {
			lo := k.PDTransferDemand(cs[i-1].Tokens, cs[i-1].From, cs[i-1].To).Bytes
			hi := k.PDTransferDemand(cs[i].Tokens, cs[i].From, cs[i].To).Bytes
			if hi < lo {
				return fmt.Errorf("%v moves %d bytes, fewer than the %d of %v", cs[i], hi, lo, cs[i-1])
			}
		}
	}
	return nil
}

// CheckStages checks a step estimate's pipeline split against the StepEstimate contract: no
// split, or at least two stages whose Overlap and NoOverlap figures are each non-negative and
// sum exactly to the step's own. A single-stage step reports no split rather than a one-entry
// one, so a caller can tell a pipeline from a step by the length alone.
func CheckStages(e kernel.StepEstimate) error {
	if len(e.Stages) == 0 {
		return nil
	}
	if len(e.Stages) == 1 {
		return fmt.Errorf("a one-stage split %+v: a step with one stage reports none", e.Stages)
	}
	var overlap, noOverlap time.Duration
	for i, s := range e.Stages {
		if s.Overlap < 0 || s.NoOverlap < 0 {
			return fmt.Errorf("stage %d is negative: %+v", i, s)
		}
		if s.Overlap > s.NoOverlap {
			return fmt.Errorf("stage %d overlaps to more than it costs with no overlap: %+v", i, s)
		}
		overlap += s.Overlap
		noOverlap += s.NoOverlap
	}
	if overlap != e.Overlap || noOverlap != e.NoOverlap {
		return fmt.Errorf("stages sum to Overlap %v, NoOverlap %v; the step is %v, %v",
			overlap, noOverlap, e.Overlap, e.NoOverlap)
	}
	return nil
}

// CheckTrust checks a step estimate's Trust against the StepEstimate contract (design VAL-3).
// The zero value is allowed: an implementation that does not report trust. A reported trust
// has a rank from 1 to 7, names the method that sets it and the entry, and, when it says why
// the rank is not its method's own, says one of the two reasons the design allows. An
// extrapolated rank is never better than transferred's, and an unidentified one is assumed's.
func CheckTrust(e kernel.StepEstimate) error {
	t := e.Trust
	if t == (kernel.Trust{}) {
		return nil
	}
	if t.Rank < 1 || t.Rank > 7 {
		return fmt.Errorf("trust rank %d is outside 1..7", t.Rank)
	}
	m := vocab.Method(t.Method)
	own, ranked := m.Rank()
	if !ranked {
		return fmt.Errorf("trust names method %q, which has no rank", t.Method)
	}
	if t.Entry == "" {
		return fmt.Errorf("trust of rank %d names no entry", t.Rank)
	}
	switch t.Flag {
	case "":
		if t.Rank != own {
			return fmt.Errorf("trust rank %d is not method %s's own %d and gives no reason", t.Rank, t.Method, own)
		}
	case "extrapolated":
		if tr, _ := vocab.MethodTransferred.Rank(); t.Rank < tr {
			return fmt.Errorf("an extrapolated price has rank %d, better than transferred's %d", t.Rank, tr)
		}
	case "unidentified":
		if as, _ := vocab.MethodAssumed.Rank(); t.Rank != as {
			return fmt.Errorf("an unidentified entry has rank %d, not assumed's %d", t.Rank, as)
		}
	default:
		return fmt.Errorf("trust flag %q is neither extrapolated nor unidentified", t.Flag)
	}
	return nil
}

// TestTrust runs CheckTrust over every batch's estimate, and requires that pricing the same
// batch twice reports the same trust. An empty list fails.
func TestTrust(t testing.TB, k kernel.Kernel, batches []kernel.Batch) {
	t.Helper()
	if len(batches) == 0 {
		t.Fatal("kerneltest.TestTrust: no batches, so nothing was checked")
	}
	for i, b := range batches {
		first, second := k.StepTime(b), k.StepTime(b)
		if err := CheckTrust(first); err != nil {
			t.Errorf("batch %d: %v", i, err)
		}
		if first.Trust != second.Trust {
			t.Errorf("batch %d: the same batch reported trust %+v and then %+v", i, first.Trust, second.Trust)
		}
	}
}

// TestStages runs CheckStages over every batch's estimate. An empty list fails.
func TestStages(t testing.TB, k kernel.Kernel, batches []kernel.Batch) {
	t.Helper()
	if len(batches) == 0 {
		t.Fatal("kerneltest.TestStages: no batches, so nothing was checked")
	}
	for i, b := range batches {
		if err := CheckStages(k.StepTime(b)); err != nil {
			t.Errorf("batch %d: %v", i, err)
		}
	}
}

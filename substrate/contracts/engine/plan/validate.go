package plan

import (
	"fmt"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/units"
)

// Validate reports a plan no pool can have. It refuses what the cost model would otherwise
// price silently: a primitive outside the fixed list, a collective that does not say what it
// moves, a stage beyond the pipeline, a negative size, a shared figure on a basis it cannot
// have. Findings are sorted, so a failure reads the same on every run.
func (p PoolPlan) Validate() error {
	var bad []string
	add := func(format string, a ...any) { bad = append(bad, fmt.Sprintf(format, a...)) }

	if strings.TrimSpace(p.Hardware) == "" {
		add("Hardware is empty; a plan names its catalog chip")
	}
	l := p.Layout
	for name, w := range map[string]int{"Tensor": l.Tensor, "Pipeline": l.Pipeline, "Data": l.Data,
		"Expert": l.Expert, "Context": l.Context, "RanksPerNode": l.RanksPerNode, "Nodes": l.Nodes} {
		if w < 0 {
			add("Layout.%s is %d", name, w)
		}
	}
	if l.Nodes > 1 && p.Fabric == "" {
		add("Layout spans %d nodes but Fabric is empty; a plan across nodes names its network", l.Nodes)
	}
	if l.RanksPerNode > 0 && l.Nodes > 0 && l.RanksPerNode*l.Nodes < l.Ranks() {
		add("Layout places %d ranks on %d nodes of %d, fewer than its %d ranks", l.RanksPerNode*l.Nodes, l.Nodes, l.RanksPerNode, l.Ranks())
	}
	stages := l.Stages()
	if len(p.KVBudget) != 0 && len(p.KVBudget) != stages {
		add("KVBudget has %d values for %d pipeline stages; it has one per stage", len(p.KVBudget), stages)
	}
	for i, b := range p.KVBudget {
		if b.Value() < 0 {
			add("KVBudget[%d] is %d bytes", i, b.Value())
		}
	}
	layers := func(name string, ls []LoweredLayer) {
		for i, ll := range ls {
			at := fmt.Sprintf("%s[%d]", name, i)
			switch {
			case !ll.Primitive.Valid():
				add("%s: primitive %q is not in the fixed list", at, ll.Primitive)
			case ll.Primitive.IsCollective() && ll.Collective == nil:
				add("%s: %s is a collective and states no group, bytes or node crossing", at, ll.Primitive)
			case !ll.Primitive.IsCollective() && ll.Collective != nil:
				add("%s: %s is not a collective but states one", at, ll.Primitive)
			}
			if ll.Layer < 0 {
				add("%s: layer index %d", at, ll.Layer)
			}
			if ll.Scorer && ll.Primitive != PrimAttentionDecode && ll.Primitive != PrimAttentionPrefill {
				add("%s: Scorer is set on %s; only an attention primitive has a block-index scorer node", at, ll.Primitive)
			}
			if ll.Stage < 0 || ll.Stage >= stages {
				add("%s: stage %d of %d pipeline stages", at, ll.Stage, stages)
			}
			if strings.TrimSpace(ll.KernelImpl.Library) == "" && ll.Primitive.Valid() && ll.Primitive != PrimElementwise {
				add("%s: %s names no kernel library; the coefficient registry looks it up by implementation", at, ll.Primitive)
			}
			if c := ll.Collective; c != nil {
				if c.Group < 2 {
					add("%s: collective over a group of %d", at, c.Group)
				}
				if c.Bytes.Value() < 0 {
					add("%s: collective message of %d bytes", at, c.Bytes.Value())
				}
			}
			if err := ll.Shape.Format.Validate(); err != nil && ll.Shape.Format != (units.QuantFormat{}) {
				add("%s: %v", at, err)
			}
			if d := ll.Shape.Dtype; d != "" {
				switch {
				case !d.Valid():
					add("%s: Dtype %q is not a catalog dtype", at, d)
				case ll.Primitive != PrimAttentionDecode && ll.Primitive != PrimAttentionPrefill:
					add("%s: Dtype %q is set on %s; only an attention primitive states the element type of its cache", at, d, ll.Primitive)
				case ll.Shape.Format != (units.QuantFormat{}) && ll.Shape.Format.PayloadBits != d.Bits():
					add("%s: Dtype %q is %d bits and Format stores %d bits of payload; they describe different caches", at, d, d.Bits(), ll.Shape.Format.PayloadBits)
				}
			}
		}
	}
	layers("Layers", p.Layers)
	layers("DraftLayers", p.DraftLayers)
	if len(p.DraftLayers) > 0 && !p.Drafts.Enabled() {
		add("DraftLayers lists %d layers but Drafts configures no speculation", len(p.DraftLayers))
	}
	for i, g := range p.KV {
		if g.BytesPerPage <= 0 || g.TokensPerPage <= 0 {
			add("KV[%d]: %d bytes per page of %d tokens; both are positive", i, g.BytesPerPage, g.TokensPerPage)
		}
		if g.Window < 0 {
			add("KV[%d]: window %d", i, g.Window)
		}
	}
	for i, t := range p.Tiers {
		if strings.TrimSpace(t.Device) == "" {
			add("Tiers[%d]: no device", i)
		}
		if t.Bytes.Value() < 0 {
			add("Tiers[%d]: %d bytes", i, t.Bytes.Value())
		}
	}
	if !p.Graphs.Mode.Valid() {
		add("Graphs.Mode %q is not a known mode", p.Graphs.Mode)
	}
	if p.Graphs.Mode == GraphNone && len(p.Graphs.Buckets) > 0 {
		add("Graphs lists %d buckets but captures no graphs", len(p.Graphs.Buckets))
	}
	for i, b := range p.Graphs.Buckets {
		if b < 1 {
			add("Graphs.Buckets[%d] is %d", i, b)
		}
	}
	if p.Drafts.Chain < 0 || p.Drafts.TreeTopK < 0 || p.Drafts.TreeSteps < 0 {
		add("Drafts has a negative width: %+v", p.Drafts)
	}
	if (p.Drafts.TreeTopK == 0) != (p.Drafts.TreeSteps == 0) {
		add("Drafts tree states top-k %d and %d steps; a tree has both", p.Drafts.TreeTopK, p.Drafts.TreeSteps)
	}
	if p.Drafts.Chain > 0 && p.Drafts.TreeTopK > 0 {
		add("Drafts is both a chain of %d and a tree; it is one", p.Drafts.Chain)
	}
	if r := p.Drafts.FirstPassRows; r != "" && !r.Valid() {
		add("Drafts.FirstPassRows is %q; it is %q or %q, or empty for not stated", r, FirstPassPerRequest, FirstPassScheduled)
	}
	if r := p.Drafts.PassRequests; r != "" && !r.Valid() {
		add("Drafts.PassRequests is %q; it is %q or %q, or empty for not stated", r, PassRequestsSampling, PassRequestsAll)
	}
	for i, w := range p.Weights {
		if _, err := w.Bytes(); err != nil {
			add("Weights[%d]: %v", i, err)
		}
		if !w.Where.Valid() {
			add("Weights[%d]: residence %q", i, w.Where)
		}
		if w.Shared && w.Where != ResidentHost {
			add("Weights[%d]: %q is shared but resident in GPU memory; only a host figure is shared by the engines on a node", i, w.Name)
		}
	}
	for i, b := range p.Buffers {
		if !b.Basis.Valid() {
			add("Buffers[%d]: %q has basis %q; a figure states per rank, per node or for the group (SUB-16)", i, b.Name, b.Basis)
		}
		if b.Bytes < 0 {
			add("Buffers[%d]: %q is %d bytes", i, b.Name, b.Bytes)
		}
	}
	if p.InFlight < 0 {
		add("InFlight is %d", p.InFlight)
	}
	if len(bad) == 0 {
		return nil
	}
	sort.Strings(bad)
	return fmt.Errorf("plan: %s", strings.Join(bad, "; "))
}

// InFlightDepth is the most steps in flight, with the zero value read as one.
func (p PoolPlan) InFlightDepth() int {
	if p.InFlight < 1 {
		return 1
	}
	return p.InFlight
}

// Validate reports a step no engine can form. ranks is the number of data-parallel ranks in the
// lockstep group; the step describes every one of them (padding depends on all of them).
func (s StepShape) Validate(ranks, stages int) error {
	var bad []string
	add := func(format string, a ...any) { bad = append(bad, fmt.Sprintf(format, a...)) }
	if !s.Mode.Valid() {
		add("mode %q is not a known mode", s.Mode)
	}
	if ranks < 1 {
		ranks = 1
	}
	if stages < 1 {
		stages = 1
	}
	if s.Mode != StepIdle && len(s.Ranks) != ranks {
		add("%d rank entries for a lockstep group of %d; a step describes every rank", len(s.Ranks), ranks)
	}
	if len(s.Padded) != 0 && len(s.Padded) != len(s.Ranks) {
		add("%d padded shapes for %d ranks", len(s.Padded), len(s.Ranks))
	}
	if s.Stage < 0 || s.Stage >= stages {
		add("stage %d of %d pipeline stages", s.Stage, stages)
	}
	for r, rk := range s.Ranks {
		var scheduled units.Tokens
		for i, q := range rk.Requests {
			if q.Scheduled < 0 || q.Computed < 0 || q.Prompt < 0 || q.Cached < 0 {
				add("rank %d request %d has a negative token count: %+v", r, i, q)
			}
			scheduled += q.Scheduled
		}
		if r < len(s.Padded) {
			if !s.Padded[r].Launch.Valid() {
				add("rank %d: launch %q", r, s.Padded[r].Launch)
			}
			if s.Padded[r].Tokens < scheduled {
				add("rank %d launches %d tokens but schedules %d; padding never removes tokens", r, s.Padded[r].Tokens, scheduled)
			}
		}
	}
	if s.DecodeQueryLen < 0 {
		add("DecodeQueryLen is %d", s.DecodeQueryLen)
	}
	if s.Drafts.Drafted < 0 || s.Drafts.Verified < 0 {
		add("drafts %+v", s.Drafts)
	}
	if n := len(s.Drafts.Passes); n != 0 && n != len(s.Ranks) {
		add("%d drafter passes for %d ranks; a step states none or one for each rank", n, len(s.Ranks))
	}
	for r, d := range s.Drafts.Passes {
		if d.Padded < 0 {
			add("rank %d: drafter launches %d tokens", r, d.Padded)
		}
		if !d.Launch.Valid() {
			add("rank %d: drafter launch %q", r, d.Launch)
		}
	}
	if s.Mode != StepVerify && s.Drafts.Verified > 0 {
		add("%d verified draft tokens in a %s step; only a verify step verifies", s.Drafts.Verified, s.Mode)
	}
	if s.Mode == StepIdle || s.Mode == StepPrebuilt {
		for r, rk := range s.Ranks {
			if r < len(s.Padded) && s.Padded[r].Launch != LaunchEager {
				add("rank %d: a %s step launches nothing", r, s.Mode)
			}
			_ = rk
		}
	}
	if len(bad) == 0 {
		return nil
	}
	sort.Strings(bad)
	return fmt.Errorf("step shape: %s", strings.Join(bad, "; "))
}

// Validate reports a transfer no engine can start.
func (t Transfer) Validate() error {
	var bad []string
	// A pull from a peer may move nothing: a remote prefill whose KV is already here. A move
	// to or from a tier of nothing is a bug in the engine, and is refused.
	if t.Tokens < 0 || (t.Tokens == 0 && t.Kind != TransferPeer) {
		bad = append(bad, fmt.Sprintf("%d tokens", t.Tokens))
	}
	if !t.Kind.Valid() {
		bad = append(bad, fmt.Sprintf("kind %q", t.Kind))
	}
	if t.Kind == TransferPeer && t.Tier != "" {
		bad = append(bad, fmt.Sprintf("a move between pools names tier %q", t.Tier))
	}
	if t.Kind != TransferPeer && t.Tier == "" {
		bad = append(bad, "an offload move names no tier")
	}
	for _, e := range []Endpoint{t.From, t.To} {
		if e.Node < 0 || e.Rack < 0 {
			bad = append(bad, fmt.Sprintf("endpoint %+v", e))
		}
	}
	if len(bad) == 0 {
		return nil
	}
	sort.Strings(bad)
	return fmt.Errorf("transfer: %s", strings.Join(bad, "; "))
}

// CheckTransfer reports a transfer this pool cannot make: a move to or from a tier names a
// device the plan does not list in Tiers, or a node the layout does not have. A move between
// pools is checked only for its own fields, because its far end is in another pool's plan.
// An Environment refuses what CheckTransfer refuses (design §6.3).
func (p PoolPlan) CheckTransfer(t Transfer) error {
	if err := t.Validate(); err != nil {
		return err
	}
	if t.Kind == TransferPeer {
		return nil
	}
	for _, tier := range p.Tiers {
		if tier.Device == t.Tier {
			nodes := max(1, p.Layout.Nodes)
			for _, e := range []Endpoint{t.From, t.To} {
				if e.Node >= nodes {
					return fmt.Errorf("transfer: endpoint node %d of a pool with %d node(s)", e.Node, nodes)
				}
			}
			return nil
		}
	}
	return fmt.Errorf("transfer: tier %q is not a device of this pool's plan", t.Tier)
}

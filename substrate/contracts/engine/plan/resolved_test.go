package plan

import (
	"slices"
	"strings"
	"testing"
)

// Every resolved setting a plan cannot mean is refused by name, and a plan of meaningful values
// is accepted.
func TestValidateResolvedRefusesEachMeaninglessSettingByName(t *testing.T) {
	if err := (PoolPlan{}).ValidateResolved(); err != nil {
		t.Fatalf("the zero plan, which states nothing, is refused: %v", err)
	}
	ok := PoolPlan{Batch: BatchLimits{Requests: 64, Tokens: 8192}, MaxContext: 8192, PageTokens: 16,
		MemoryFraction: 0.9, Cache: CacheFormat{ElementBytes: 2, LatentCellBytes: 656, StateElementBytes: 4,
			Recurrent: RecurrentPageAligned}, MoE: MoEPlan{RedundantExperts: 16},
		DecodeContext: DecodeContextPlan{Combine: CombineAllToAll, Interleave: 16},
		Engram:        EngramPlacement{Host: true, Shards: 2},
		Graphs:        GraphPolicy{Decode: LaunchGraph, Other: LaunchPiecewise, Ceiling: 512},
		Drafts:        DraftPolicy{RetainedTokens: 2}, InFlight: 2,
		Launches: []LaunchChange{{FromTokens: 1, ToTokens: 128, PerLayer: -3}, {FromTokens: 129, ToTokens: 256, PerLayer: -2}}}
	if err := ok.ValidateResolved(); err != nil {
		t.Fatalf("a plan of meaningful values is refused: %v", err)
	}
	for _, c := range []struct {
		field string
		edit  func(*PoolPlan)
	}{
		{"Batch.Requests", func(p *PoolPlan) { p.Batch.Requests = -1 }},
		{"Batch.Tokens", func(p *PoolPlan) { p.Batch.Tokens = -1 }},
		{"MaxContext", func(p *PoolPlan) { p.MaxContext = -1 }},
		{"PageTokens", func(p *PoolPlan) { p.PageTokens = -1 }},
		{"MemoryFraction", func(p *PoolPlan) { p.MemoryFraction = 1.5 }},
		{"MemoryFraction", func(p *PoolPlan) { p.MemoryFraction = -0.1 }},
		{"Cache.Recurrent", func(p *PoolPlan) { p.Cache.Recurrent = "sometimes" }},
		{"Cache.TrackingSlots", func(p *PoolPlan) { p.Cache.Recurrent = RecurrentTracked }},
		{"Cache.TrackingSlots", func(p *PoolPlan) { p.Cache.TrackingSlots = 1 }},
		{"Cache.TrackingSlots", func(p *PoolPlan) { p.Cache.TrackingSlots = -1 }},
		{"Cache.ElementBytes", func(p *PoolPlan) { p.Cache.ElementBytes = -1 }},
		{"Cache.LatentCellBytes", func(p *PoolPlan) { p.Cache.LatentCellBytes = -1 }},
		{"Cache.StateElementBytes", func(p *PoolPlan) { p.Cache.StateElementBytes = -1 }},
		{"MoE.RedundantExperts", func(p *PoolPlan) { p.MoE.RedundantExperts = -1 }},
		{"MoE.Routing", func(p *PoolPlan) { p.MoE.Routing = "inline" }},
		{"DecodeContext.Combine", func(p *PoolPlan) { p.DecodeContext.Combine = "ring" }},
		{"DecodeContext.Interleave", func(p *PoolPlan) { p.DecodeContext.Interleave = -1 }},
		{"Engram.Shards", func(p *PoolPlan) { p.Engram.Shards = -1 }},
		{"Graphs.Decode", func(p *PoolPlan) { p.Graphs.Decode = "full" }},
		{"Graphs.Other", func(p *PoolPlan) { p.Graphs.Other = "full" }},
		{"Graphs.Ceiling", func(p *PoolPlan) { p.Graphs.Ceiling = -1 }},
		{"Drafts.RetainedTokens", func(p *PoolPlan) { p.Drafts.RetainedTokens = -1 }},
		{"InFlight", func(p *PoolPlan) { p.InFlight = -1 }},
		{"Launches[0] covers 0 to 128", func(p *PoolPlan) { p.Launches = []LaunchChange{{ToTokens: 128, PerLayer: -3}} }},
		{"Launches[0] covers 129 to 128", func(p *PoolPlan) { p.Launches = []LaunchChange{{FromTokens: 129, ToTokens: 128, PerLayer: -3}} }},
		{"Launches[0] changes no launch", func(p *PoolPlan) { p.Launches = []LaunchChange{{FromTokens: 1, ToTokens: 8}} }},
		{"Launches[1] (100 to 256 tokens) overlaps Launches[0]", func(p *PoolPlan) {
			p.Launches = []LaunchChange{{FromTokens: 1, ToTokens: 128, PerLayer: -3}, {FromTokens: 100, ToTokens: 256, PerLayer: -2}}
		}},
	} {
		p := ok
		p.Launches = slices.Clone(ok.Launches)
		c.edit(&p)
		err := p.ValidateResolved()
		if err == nil || !strings.Contains(err.Error(), c.field) {
			t.Errorf("%s: err %v, want a refusal naming the field", c.field, err)
		}
	}
}

// Original to blis-sglang.

package cache

import (
	"math"
	"slices"
	"strings"
	"testing"
)

// gdn is the linear-attention shape of the Qwen3.5 text model, a size small enough for a table.
func gdn() LinearAttnConfig {
	return LinearAttnConfig{NumHiddenLayers: 8, FullAttentionInterval: 4, KeyHeadDim: 128, NumKeyHeads: 16,
		ValueHeadDim: 128, NumValueHeads: 64, ConvKernel: 4}
}

// The state shape is built for the smallest widths it accepts and refused below them.
func TestLinearStateParamsBounds(t *testing.T) {
	ok := func(c LinearAttnConfig, tp int) bool {
		_, err := NewLinearStateParams(c, tp, BFloat16, Float32)
		return err == nil
	}
	if !ok(gdn(), 1) {
		t.Error("tensor-parallel width 1 was refused")
	}
	if c := gdn(); !ok(func() LinearAttnConfig { c.FullAttentionInterval = 1; return c }(), 1) {
		t.Error("a full-attention interval of 1 was refused")
	}
	if c := gdn(); !ok(func() LinearAttnConfig { c.ConvKernel = 2; return c }(), 1) {
		t.Error("a convolution kernel of 2 was refused")
	}
	for name, bad := range map[string]func() (LinearAttnConfig, int){
		"tp 0":          func() (LinearAttnConfig, int) { return gdn(), 0 },
		"tp -1":         func() (LinearAttnConfig, int) { return gdn(), -1 },
		"interval 0":    func() (LinearAttnConfig, int) { c := gdn(); c.FullAttentionInterval = 0; return c, 1 },
		"kernel 1":      func() (LinearAttnConfig, int) { c := gdn(); c.ConvKernel = 1; return c, 1 },
		"kernel 0":      func() (LinearAttnConfig, int) { c := gdn(); c.ConvKernel = 0; return c, 1 },
		"interval -1":   func() (LinearAttnConfig, int) { c := gdn(); c.FullAttentionInterval = -1; return c, 1 },
		"value heads 6": func() (LinearAttnConfig, int) { c := gdn(); c.NumValueHeads = 6; return c, 4 },
	} {
		c, tp := bad()
		func() {
			defer func() {
				if p := recover(); p != nil {
					t.Errorf("%s: panicked (%v), want an error", name, p)
				}
			}()
			if ok(c, tp) {
				t.Errorf("%s: accepted", name)
			}
		}()
	}
}

func poolParams(t *testing.T) MambaPoolParams {
	t.Helper()
	st, err := NewLinearStateParams(gdn(), 4, BFloat16, Float32)
	if err != nil {
		t.Fatal(err)
	}
	return MambaPoolParams{Size: 3, SpecStateSize: 2, State: st}
}

// The pool is sized for zero slots but not for fewer, nor for no layers.
func TestMambaPoolBytesBounds(t *testing.T) {
	p := poolParams(t)
	p.Size, p.SpecStateSize = 0, 0
	if _, err := MambaPoolBytes(p); err != nil {
		t.Errorf("a pool of size 0 with no spec state was refused: %v", err)
	}
	for name, edit := range map[string]func(*MambaPoolParams){
		"size -1":       func(p *MambaPoolParams) { p.Size = -1 },
		"spec state -1": func(p *MambaPoolParams) { p.SpecStateSize = -1 },
		"no layers":     func(p *MambaPoolParams) { p.State.Layers = nil },
	} {
		q := poolParams(t)
		edit(&q)
		if _, err := MambaPoolBytes(q); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// ReplaySSM's raw rings are added only for the speculative variant with a convolution dtype that is not
// float32: decode-only ReplaySSM adds the d, k and g rings and nothing else, whatever the state dtype.
func TestReplayRingsAreChargedPerVariant(t *testing.T) {
	base := poolParams(t)
	base.Size, base.SpecStateSize = 3, 3
	plain, err := MambaPoolBytes(base)
	if err != nil {
		t.Fatal(err)
	}
	st := base.State
	hv, v, k := int64(st.Temporal[0]), int64(st.Temporal[1]), int64(st.Temporal[2])
	hk, l := int64(st.NumKHeadsPerTP), int64(16)
	layers := int64(len(st.Layers))
	ssmB, convB := int64(st.TemporalDType.ItemSize()), int64(st.ConvDType.ItemSize())
	decode := base
	decode.ReplayDecode, decode.ReplayCacheLen = true, int(l)
	gotDecode, err := MambaPoolBytes(decode)
	if err != nil {
		t.Fatal(err)
	}
	slots := int64(base.Size) + 1
	wantDecode := layers * slots * (hv*l*v*ssmB + hk*l*k*ssmB + hv*l*4)
	if gotDecode-plain != wantDecode {
		t.Errorf("decode-only ReplaySSM adds %d bytes, want %d (d, k and g rings at the state dtype)", gotDecode-plain, wantDecode)
	}
	spec := base
	spec.ReplaySpec, spec.ReplayCacheLen = true, int(l)
	gotSpec, err := MambaPoolBytes(spec)
	if err != nil {
		t.Fatal(err)
	}
	specSlots := int64(base.SpecStateSize) + 1
	wantSpec := layers*specSlots*(hv*l*v*convB+hk*l*k*convB+hv*l*4) + layers*specSlots*(hv*l*v*convB+hk*l*k*convB)
	if gotSpec-plain != wantSpec {
		t.Errorf("speculative ReplaySSM with a bf16 convolution adds %d bytes, want %d (the raw rings too)", gotSpec-plain, wantSpec)
	}
	f32 := spec
	f32.State.ConvDType = Float32
	gotF32, err := MambaPoolBytes(f32)
	if err != nil {
		t.Fatal(err)
	}
	plainF32 := base
	plainF32.State.ConvDType = Float32
	pf, _ := MambaPoolBytes(plainF32)
	wantF32 := layers * specSlots * (hv*l*v*4 + hk*l*k*4 + hv*l*4)
	if gotF32-pf != wantF32 {
		t.Errorf("speculative ReplaySSM with a float32 convolution adds %d bytes, want %d (no raw rings)", gotF32-pf, wantF32)
	}
}

// The slots a group took ahead are handed out one at a time to single-slot allocations only, in order,
// and the unused ones go back when the group ends; once the group is used up, allocations come from the
// free list again.
func TestMambaSlotGroupAllocation(t *testing.T) {
	a := NewMambaSlotAllocator(6)
	a.AllocGroupBegin(3) // takes 1, 2, 3
	if got, ok := a.Alloc(2); !ok || !slices.Equal(got, []int64{4, 5}) {
		t.Errorf("a two-slot allocation inside a group took %v, want [4 5] from the free list", got)
	}
	for _, want := range []int64{1, 2, 3} {
		if got, ok := a.Alloc(1); !ok || !slices.Equal(got, []int64{want}) {
			t.Errorf("a single-slot allocation took %v, want [%d] from the group", got, want)
		}
	}
	if got, ok := a.Alloc(1); !ok || !slices.Equal(got, []int64{6}) {
		t.Errorf("with the group used up, a single-slot allocation took %v, want [6] from the free list", got)
	}
	a.AllocGroupEnd()
	if got := a.FreeSlots(); len(got) != 0 {
		t.Errorf("after the group ended the free list is %v, want empty", got)
	}

	b := NewMambaSlotAllocator(6)
	b.AllocGroupBegin(4)
	b.Alloc(1)
	b.AllocGroupEnd()
	if got := b.FreeSlots(); !slices.Equal(got, []int64{5, 6, 2, 3, 4}) {
		t.Errorf("free list %v after ending a group with three unused slots, want [5 6 2 3 4]", got)
	}
	if got, ok := b.Alloc(1); !ok || !slices.Equal(got, []int64{5}) {
		t.Errorf("an allocation after the group ended took %v, want [5]", got)
	}

	c := NewMambaSlotAllocator(2)
	c.AllocGroupBegin(3) // more than are free: no group
	if got, ok := c.Alloc(1); !ok || !slices.Equal(got, []int64{1}) {
		t.Errorf("after a group that did not fit, an allocation took %v, want [1]", got)
	}
	c.AllocGroupBegin(0)
	if got, ok := c.Alloc(1); !ok || !slices.Equal(got, []int64{2}) {
		t.Errorf("after an empty group, an allocation took %v, want [2]", got)
	}
}

// Free ping-pong entries: the entry to keep is 0, or 1 of a pool that tracks two slots; any other is
// refused.
func TestFreeMambaCacheKeepsOnlyAValidEntry(t *testing.T) {
	for _, tc := range []struct {
		overlap bool
		keep    int
		ok      bool
	}{{true, 0, true}, {true, 1, true}, {true, 2, false}, {true, -1, false}, {false, 0, true}, {false, 1, false}} {
		p := NewHybridRequestPool(2, 8, 16, true, false, tc.overlap)
		r := &Req{ID: "r"}
		if _, err := p.Alloc([]*Req{r}); err != nil {
			t.Fatal(err)
		}
		keep := tc.keep
		err := p.FreeMambaCache(*r.KV.ReqPoolIdx, r.KV.Mamba, &keep)
		if (err == nil) != tc.ok {
			t.Errorf("overlap %v, keep %d: error %v, want ok %v", tc.overlap, tc.keep, err, tc.ok)
		}
	}
}

// Python's float //, which floors toward minus infinity, for every sign of the operands.
func TestPyFloorDivSigns(t *testing.T) {
	for _, tc := range []struct{ a, b, want float64 }{
		{7, 2, 3}, {-7, 2, -4}, {7, -2, -4}, {-7, -2, 3}, {6, 2, 3}, {-6, 2, -3}, {6, -2, -3},
		{-1, 5, -1}, {1, -5, -1}, {0, 5, 0}, {5.5, 2, 2}, {-5.5, 2, -3},
	} {
		if got := pyFloorDiv(tc.a, tc.b); got != tc.want {
			t.Errorf("%v // %v = %v, want %v", tc.a, tc.b, got, tc.want)
		}
	}
	if got := pyFloorDiv(-0.0, 5); !math.Signbit(got) && got != 0 {
		t.Errorf("-0 // 5 = %v", got)
	}
}

func budgetFor(t *testing.T) MambaBudget {
	t.Helper()
	st, err := NewLinearStateParams(gdn(), 4, BFloat16, Float32)
	if err != nil {
		t.Fatal(err)
	}
	return MambaBudget{TotalRestGiB: 40, State: st, AttnDPSize: 1, FullMemoryRatio: 0.9, Strategy: MambaNoBuffer}
}

// The budget is refused for a data-parallel width below one, for no linear layers, and for a state of
// no bytes, each by its own message.
func TestMambaCacheSizeRefusals(t *testing.T) {
	b := budgetFor(t)
	if _, _, err := MambaCacheSize(b); err != nil {
		t.Fatalf("the base budget was refused: %v", err)
	}
	dp0 := b
	dp0.AttnDPSize = 0
	if _, _, err := MambaCacheSize(dp0); err == nil || !strings.Contains(err.Error(), "attention dp") {
		t.Errorf("a data-parallel width of 0: %v", err)
	}
	none := b
	none.State.Layers = nil
	if _, _, err := MambaCacheSize(none); err == nil || !strings.Contains(err.Error(), "linear layers") {
		t.Errorf("no linear layers: %v", err)
	}
	empty := b
	empty.State = LinearStateParams{Layers: []int{0}, ConvDType: BFloat16, TemporalDType: Float32}
	if _, _, err := MambaCacheSize(empty); err == nil || !strings.Contains(err.Error(), "no state per request") {
		t.Errorf("a state of no bytes: %v", err)
	}
}

// A request cap is refused when the state slots allow no request, and allowed at one.
func TestMaxNumReqsMambaBounds(t *testing.T) {
	if n, err := MaxNumReqsMamba(1000, 10, nil, 1, 3, 3); err != nil || n != 1 {
		t.Errorf("state slots for exactly one request: %d, %v", n, err)
	}
	if n, err := MaxNumReqsMamba(1000, 10, nil, 1, 2, 3); err == nil {
		t.Errorf("state slots for no request were accepted: %d", n)
	}
}

// A request that holds no mamba state is refused by name, not dereferenced.
func TestPrepareForCachingReqNeedsAMambaSlot(t *testing.T) {
	pool := NewHybridRequestPool(2, 6, 32, true, false, true)
	life, err := NewMambaLifecycle(MambaLifecycleParams{Pool: pool, ExtraBuffer: true, Evictor: &treeSlots{pool: pool}})
	if err != nil {
		t.Fatal(err)
	}
	for name, r := range map[string]*Req{
		"no mamba state":        {ID: "a"},
		"mamba state, no slot":  {ID: "b", KV: ReqKV{Mamba: &MambaKV{}}},
		"mamba state, no slot2": {ID: "c", KV: ReqKV{ReqPoolIdx: ptrInt(0), Mamba: &MambaKV{}}},
	} {
		if _, err := life.PrepareForCachingReq(r, &MambaInsert{}, 10, true); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func ptrInt(i int) *int { return &i }

// Original to blis-sglang.

package cache

import "testing"

// The smallest legal pool of each kind is accepted and one step below it is refused, whichever
// argument takes the step.
func TestKVPoolsAcceptTheirSmallestShapeAndRefuseOneBelow(t *testing.T) {
	ok := MHAParams{Size: 0, PageSize: 1, DType: BFloat16, HeadNum: 1, HeadDim: 1, LayerNum: 1}
	if _, err := NewMHAPool(ok); err != nil {
		t.Errorf("the smallest MHA pool: %v", err)
	}
	for name, edit := range map[string]func(*MHAParams){
		"page size": func(p *MHAParams) { p.PageSize = 0 }, "size": func(p *MHAParams) { p.Size = -1 },
		"layers": func(p *MHAParams) { p.LayerNum = 0 }, "heads": func(p *MHAParams) { p.HeadNum = 0 },
		"head dim": func(p *MHAParams) { p.HeadDim = 0 },
	} {
		p := ok
		edit(&p)
		if _, err := NewMHAPool(p); err == nil {
			t.Errorf("an MHA pool with %s one below its minimum was built", name)
		}
	}
	mla := MLAParams{Size: 0, PageSize: 1, DType: BFloat16, KVLoraRank: 1, QKRopeHeadDim: 0, LayerNum: 1}
	if _, err := NewMLAPool(mla); err != nil {
		t.Errorf("the smallest MLA pool: %v", err)
	}
	for name, edit := range map[string]func(*MLAParams){
		"page size": func(p *MLAParams) { p.PageSize = 0 }, "size": func(p *MLAParams) { p.Size = -1 },
		"layers": func(p *MLAParams) { p.LayerNum = 0 }, "rank": func(p *MLAParams) { p.KVLoraRank = 0 },
		"rope": func(p *MLAParams) { p.QKRopeHeadDim = -1 },
	} {
		p := mla
		edit(&p)
		if _, err := NewMLAPool(p); err == nil {
			t.Errorf("an MLA pool with %s one below its minimum was built", name)
		}
	}
}

// Only the dtypes a modelled pool stores parse, and uint8 is the storage type of FP8, not a KV dtype.
func TestOnlyStoredDTypesParse(t *testing.T) {
	for _, s := range []string{"bfloat16", "torch.float16", "float32", "float8_e4m3fn", "torch.float8_e5m2"} {
		if _, err := ParseDType(s); err != nil {
			t.Errorf("%s: %v", s, err)
		}
	}
	for _, s := range []string{"uint8", "torch.uint8", "int4", ""} {
		if _, err := ParseDType(s); err == nil {
			t.Errorf("%q parsed", s)
		}
	}
}

// A DSA model packs its FP8 latent rows with scales and a bf16 rope part, unless the model is not
// DSA, the attention backends read the raw layout, or the dtype is not FP8.
func TestMLALatentRowWidthPacksOnlyFP8DSA(t *testing.T) {
	for _, c := range []struct {
		dsa, raw bool
		dtype    DType
		want     int
	}{
		{true, false, Float8E4M3, 512 + 4*4 + 64*2}, {false, false, Float8E4M3, 576}, {true, true, Float8E4M3, 576},
		{true, false, BFloat16, 576}, {false, true, BFloat16, 576},
	} {
		got, err := MLAKVCacheDim(512, 64, c.dtype, c.dsa, c.raw)
		if err != nil || got != c.want {
			t.Errorf("dsa %v raw %v %s: %d, %v; want %d", c.dsa, c.raw, c.dtype, got, err, c.want)
		}
	}
	if _, err := MLAKVCacheDim(500, 64, Float8E4M3, true, false); err == nil {
		t.Error("a rank that is not a multiple of the quant block was packed")
	}
}

// The KV budget is the free memory less the runtime slack; exactly none left is refused, and the
// token capacity refuses a page size below one or a negative cell.
func TestKVBudgetAndTokenCapacityRefuseTheEmptyAndTheNegative(t *testing.T) {
	if _, err := KVBudgetBytes(Profile{AvailableGiB: 10, PreModelLoadGiB: 10, MemFractionStatic: 0}); err == nil {
		t.Error("a budget of exactly zero was granted")
	}
	if got, err := KVBudgetBytes(Profile{AvailableGiB: 11, PreModelLoadGiB: 10, MemFractionStatic: 0}); err != nil || got != 1<<30 {
		t.Errorf("one GiB left: %d, %v", got, err)
	}
	if _, err := TokenCapacity(1000, 10, 0, nil, 4096); err == nil {
		t.Error("page size 0")
	}
	if _, err := TokenCapacity(1000, -1, 1, nil, 4096); err == nil {
		t.Error("a negative cell")
	}
	if got, err := TokenCapacity(1000, 0, 1, nil, 4096); err != nil || got != 4096 {
		t.Errorf("no KV per token and no cap takes the context length: %d, %v", got, err)
	}
	five := 5
	if got, err := TokenCapacity(1000, 0, 1, &five, 4096); err != nil || got != 5 {
		t.Errorf("no KV per token with a cap takes the cap: %d, %v", got, err)
	}
}

// A request-pool row is writable up to the last row and the last position, and not outside them;
// a reused row must carry allocated KV; the last free rows can all be taken.
func TestRequestPoolWritesStayInsideAndRowsAreAllTakeable(t *testing.T) {
	p := NewRequestPool(2, 4)
	p.Write(0, 0, []int64{1})
	p.Write(2, 0, []int64{1, 2, 3, 4})
	p.Write(1, 3, []int64{9})
	for name, f := range map[string]func(){
		"row below": func() { p.Write(-1, 0, []int64{1}) }, "row above": func() { p.Write(3, 0, []int64{1}) },
		"start below": func() { p.Write(1, -1, []int64{1}) }, "past the end": func() { p.Write(1, 3, []int64{1, 2}) },
	} {
		mustPanic(t, name, f)
	}
	a, b := &Req{}, &Req{}
	rows, ok := p.Alloc([]*Req{a, b})
	if !ok || len(rows) != 2 {
		t.Fatalf("both rows: %v, %v", rows, ok)
	}
	if _, ok := p.Alloc([]*Req{{}}); ok {
		t.Error("a third row from a pool of two")
	}
	mustPanic(t, "a held row with no KV allocated", func() { p.Alloc([]*Req{a}) })
	a.KV.KVAllocatedLen = 1
	if _, ok := p.Alloc([]*Req{a}); !ok {
		t.Error("a held row with KV allocated was not reused")
	}
	if k := (Key{Bigram: true}).Len(); k != 0 {
		t.Errorf("an empty bigram key has length %d", k)
	}
	if k := (Key{Bigram: true, Tokens: []int64{1, 2, 3}}).Len(); k != 2 {
		t.Errorf("a bigram key of three tokens has length %d", k)
	}
}

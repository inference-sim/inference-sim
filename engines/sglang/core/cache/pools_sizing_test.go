// Original to blis-sglang.

package cache

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// sizingTrace is the sizing oracle/run_pools.py records beside a direct trace's calls: each
// pool's buffer bytes (get_kv_size_bytes), and the per-token cell SGLang's default pool
// configurator charges for each shape.
type sizingTrace struct {
	Calls  []Call `json:"calls"`
	Sizing []struct {
		Pool        string          `json:"pool"`
		Class       string          `json:"class"`
		KVSizeBytes json.RawMessage `json:"kv_size_bytes"`
	} `json:"sizing"`
	Cells []struct {
		Name     string `json:"name"`
		CellSize int64  `json:"cell_size"`
	} `json:"cells"`
}

func loadSizing(t *testing.T) map[string]sizingTrace {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(PoolsGoldenDir, "direct", "*.trace.json"))
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]sizingTrace{}
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		var s sizingTrace
		if err := json.Unmarshal(raw, &s); err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		out[strings.TrimSuffix(filepath.Base(p), ".trace.json")] = s
	}
	return out
}

// Each MHA, MLA and DSA pool SGLang built, rebuilt by the port from the same constructor
// arguments, holds the bytes SGLang's buffers hold: K and V for MHA, the latent rows for MLA,
// and the latent rows with the indexer key cache for DSA.
func TestPoolBufferBytesAreSGLangs(t *testing.T) {
	n := 0
	for name, s := range loadSizing(t) {
		r := newAllocReplay(t, Trace{Calls: s.Calls})
		pools := map[int]KVPool{}
		var inits []Call
		for _, c := range s.Calls {
			if c.Iface == KVPoolIface && c.Method == "__init__" {
				num := func(arg string) int {
					var v int
					r.arg(c, arg, &v)
					return v
				}
				p, _ := r.buildKVPool(c, num)
				pools[len(inits)] = p
				inits = append(inits, c)
			}
		}
		if len(inits) != len(s.Sizing) {
			t.Fatalf("%s: %d pools built, %d sized", name, len(inits), len(s.Sizing))
		}
		for i, want := range s.Sizing {
			if inits[i].Class != want.Class {
				t.Fatalf("%s: pool %s is a %s, built as a %s", name, want.Pool, want.Class, inits[i].Class)
			}
			var got any
			switch p := pools[i].(type) {
			case *MHAPool:
				k, v := p.KVSizeBytes()
				got = []int64{k, v}
			case *MLAPool:
				got = p.KVSizeBytes()
			case *DSAPool:
				got = p.KVSizeBytes()
			}
			g, _ := json.Marshal(got)
			if string(g) != strings.Join(strings.Fields(string(want.KVSizeBytes)), "") {
				t.Errorf("%s: %s (%s) holds %s bytes in the port, %s in SGLang", name, want.Pool, want.Class, g, want.KVSizeBytes)
			}
			n++
		}
	}
	if n == 0 {
		t.Fatal("no pool was sized")
	}
}

// The per-token cell, which divides the KV budget into the token capacity, is SGLang's
// DefaultPoolConfigurator._compute_cell_size for each shape of oracle/scenarios-pools: MHA heads
// over K and V, MLA's latent row, and DSA's latent row with one index-key row per layer.
func TestBytesPerTokenIsSGLangsCell(t *testing.T) {
	mustMHA := func(p MHAParams) KVPool {
		m, err := NewMHAPool(p)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
	mustMLA := func(p MLAParams) KVPool {
		m, err := NewMLAPool(p)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
	mustDSA := func(dt DType, layers int) KVPool {
		dim, err := MLAKVCacheDim(512, 64, dt, true, false)
		if err != nil {
			t.Fatal(err)
		}
		d, err := NewDSAPool(DSAParams{Size: 64, PageSize: 64, DType: dt, KVLoraRank: 512, QKRopeHeadDim: 64,
			LayerNum: layers, IndexHeadDim: 128, KVCacheDim: dim, IndexKPool: 1})
		if err != nil {
			t.Fatal(err)
		}
		return d
	}
	v64 := 64
	// The inputs of each `cells` entry of oracle/scenarios-pools/pool-sizing.json.
	shapes := map[string]KVPool{
		"mha":       mustMHA(MHAParams{Size: 1, PageSize: 1, DType: BFloat16, HeadNum: 2, HeadDim: 64, LayerNum: 24}),
		"mha_fp8_v": mustMHA(MHAParams{Size: 1, PageSize: 1, DType: Float8E4M3, HeadNum: 4, HeadDim: 128, LayerNum: 5, VHeadDim: &v64}),
		"mla":       mustMLA(MLAParams{Size: 1, PageSize: 1, DType: BFloat16, KVLoraRank: 512, QKRopeHeadDim: 64, LayerNum: 4}),
		"dsa_fp8":   mustDSA(Float8E4M3, 4),
		"dsa_bf16":  mustDSA(BFloat16, 6),
	}
	seen := 0
	for name, s := range loadSizing(t) {
		for _, c := range s.Cells {
			p, ok := shapes[c.Name]
			if !ok {
				t.Errorf("%s: SGLang sized a cell %q the test does not build", name, c.Name)
				continue
			}
			seen++
			if got := p.BytesPerToken(); got != c.CellSize {
				t.Errorf("%s: %s costs %d bytes a token in the port, %d in SGLang", name, c.Name, got, c.CellSize)
			}
		}
	}
	if seen != len(shapes) {
		t.Errorf("SGLang sized %d of the test's %d cells", seen, len(shapes))
	}
}

// The token budget follows SGLang's memory-fraction arithmetic. The recorded tiny-memory
// scenario gives the device 5 GiB, free before and after the (absent) weights; SGLang resolves
// mem_fraction_static for it to round((5120 - 3712) / 5120, 3) = 0.275
// (arg_groups/memory_hook.py:296-326@297e54fd, which config resolution ports), and sizes its pool
// from Qwen2.5-0.5B's cell. The port, from the same figures, must size the pool and the request
// pool as SGLang's constructors were called.
func TestTokenBudgetFromTheMemoryFractionIsSGLangs(t *testing.T) {
	tr, ok := loadGoldens(t)["recorded/tiny-memory"]
	if !ok {
		t.Fatal("no recorded/tiny-memory trace")
	}
	var scenario struct {
		DeviceMemoryGiB    float64 `json:"device_memory_gib"`
		PageSize           int     `json:"page_size"`
		MaxTotalTokens     *int    `json:"max_total_tokens"`
		MaxRunningRequests *int    `json:"max_running_requests"`
	}
	raw, err := os.ReadFile(filepath.Join("..", "..", "oracle", "scenarios", "tiny-memory.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &scenario); err != nil {
		t.Fatal(err)
	}
	r := newAllocReplay(t, tr)
	var kvInit, allocInit, reqInit *Call
	for i, c := range tr.Calls {
		if c.Method != "__init__" {
			continue
		}
		switch c.Iface {
		case KVPoolIface:
			kvInit = &tr.Calls[i]
		case AllocatorIface:
			allocInit = &tr.Calls[i]
		case ReqToTokenIface:
			reqInit = &tr.Calls[i]
		}
	}
	if kvInit == nil || allocInit == nil || reqInit == nil {
		t.Fatal("the trace does not build all three pools")
	}
	num := func(c Call) func(string) int {
		return func(arg string) int {
			var v int
			r.arg(c, arg, &v)
			return v
		}
	}
	pool, _ := r.buildKVPool(*kvInit, num(*kvInit))
	budget, err := KVBudgetBytes(Profile{AvailableGiB: scenario.DeviceMemoryGiB, PreModelLoadGiB: scenario.DeviceMemoryGiB,
		MemFractionStatic: 0.275})
	if err != nil {
		t.Fatal(err)
	}
	// The pool is first built at the capacity the budget gives, so its size is that capacity.
	// The model's context length: the request pool's rows hold it and 4 positions of headroom
	// when there is no speculation (mem_cache/allocation_sizing.py:96-105@297e54fd).
	contextLen := num(*reqInit)("max_context_len") - 4
	capacity, err := TokenCapacity(budget, pool.BytesPerToken(), scenario.PageSize, scenario.MaxTotalTokens, contextLen)
	if err != nil {
		t.Fatal(err)
	}
	if want := num(*allocInit)("size"); capacity != want || pool.Size() != want {
		t.Errorf("the port's token capacity is %d; SGLang sized its pool %d and its allocator %d", capacity, pool.Size(), want)
	}
	if got, want := MaxNumReqs(capacity, contextLen, scenario.MaxRunningRequests, 1), num(*reqInit)("size"); got != want {
		t.Errorf("the port gives the request pool %d rows, SGLang %d", got, want)
	}
}

// A user cap and a page size floor the capacity to whole pages; a budget that the weights
// exhaust is refused with SGLang's advice (kv_cache_configurator.py:3043-3058@297e54fd). The
// figures are worked from SGLang's formulas by hand; the oracle checks them on tiny-memory above.
func TestTokenCapacityCapsAndRefusals(t *testing.T) {
	cap100 := 100
	if got, _ := TokenCapacity(1<<20, 1000, 16, &cap100, 4096); got != 96 {
		t.Errorf("a 100-token cap on 16-token pages gives %d tokens, want 96", got)
	}
	if got, _ := TokenCapacity(1<<20, 1000, 16, nil, 4096); got != 1040 {
		t.Errorf("1 MiB over 1000-byte cells on 16-token pages gives %d tokens, want 1040", got)
	}
	_, err := KVBudgetBytes(Profile{AvailableGiB: 1, PreModelLoadGiB: 10, MemFractionStatic: 0.5})
	if err == nil || !strings.Contains(err.Error(), "--mem-fraction-static") {
		t.Errorf("weights that leave no memory: %v", err)
	}
	// A pool with no KV per token is sized by the cap, else the context length, in whole pages.
	if got, _ := TokenCapacity(1<<20, 0, 16, nil, 4100); got != 4096 {
		t.Errorf("no KV per token and no cap gives %d tokens, want the context length floored, 4096", got)
	}
	if got, _ := TokenCapacity(1<<20, 0, 16, &cap100, 4100); got != 96 {
		t.Errorf("no KV per token and a 100-token cap gives %d tokens, want 96", got)
	}
}

// Every allocator and pool the port does not model is refused before the run starts, by its
// SGLang class name, citing an open issue (SGL-6, X28); the modelled ones pass.
func TestUnmodelledAllocatorsAndPoolsAreRefusedByName(t *testing.T) {
	if err := CheckPoolSelection("PagedTokenToKVPoolAllocator", "DSATokenToKVPool", "ReqToTokenPool"); err != nil {
		t.Errorf("a modelled selection is refused: %v", err)
	}
	for _, c := range []struct {
		kind  PoolKind
		class string
	}{
		{AllocatorKind, "SWATokenToKVPoolAllocator"}, {AllocatorKind, "PageInterleavePoolAllocator"},
		{KVPoolKind, "DeepSeekV4TokenToKVPool"}, {KVPoolKind, "HybridLinearKVPool"}, {KVPoolKind, "MLATokenToKVPoolFP4"},
		{ReqPoolKind, "HybridReqToTokenPool"}, {ReqPoolKind, "MambaPool"},
	} {
		err := CheckPoolClass(c.kind, c.class)
		var ref *PoolRefusal
		if !errors.As(err, &ref) || !strings.Contains(err.Error(), c.class) || !strings.Contains(err.Error(), RefusedPoolsIssue) {
			t.Errorf("%s %s: %v; want a refusal naming it and citing %s", c.kind, c.class, err, RefusedPoolsIssue)
		}
	}
	// A class of the wrong kind, and a name that is no class of the release, are refused too.
	if err := CheckPoolClass(AllocatorKind, "MHATokenToKVPool"); err == nil {
		t.Error("a KV pool is accepted as an allocator")
	}
	if err := CheckPoolClass(KVPoolKind, "NoSuchPool"); err == nil || !strings.Contains(err.Error(), "NoSuchPool") {
		t.Errorf("an unknown pool: %v", err)
	}
	err := CheckPoolSelection("SWATokenToKVPoolAllocator", "SWAKVPool", "ReqToTokenPool")
	if err == nil || !strings.Contains(err.Error(), "SWATokenToKVPoolAllocator") || !strings.Contains(err.Error(), "SWAKVPool") {
		t.Errorf("a selection of two refused classes: %v; want both named", err)
	}
}

// Every refused class's refusal cites the open issue, and none is also modelled.
func TestEveryRefusalCitesTheOpenIssue(t *testing.T) {
	refused := RefusedPoolClasses()
	if len(refused) == 0 {
		t.Fatal("no class is refused")
	}
	for _, c := range refused {
		r := poolClasses[c]
		if err := CheckPoolClass(r.Kind, c); err == nil || !strings.Contains(err.Error(), RefusedPoolsIssue) ||
			!strings.Contains(err.Error(), c) {
			t.Errorf("%s: %v", c, err)
		}
	}
	// The selecting setting is named where it is known.
	if err := CheckPoolClass(KVPoolKind, "PageInterleaveMHATokenToKVPool"); err == nil ||
		!strings.Contains(err.Error(), "--enable-kv-cache-sharding") {
		t.Errorf("a page-interleave pool's refusal does not name its setting: %v", err)
	}
}

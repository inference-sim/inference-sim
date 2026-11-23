package model

import (
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// Properties of the helpers a cost model prices with, stated over many inputs rather than
// a few chosen ones, and metamorphic relations: transformations of a graph whose effect on
// validity and on priced quantities is known in advance.

var allDTypes = []DType{DTypeBF16, DTypeFP16, DTypeFP8, DTypeNVFP4, DTypeMXFP4, DTypeINT8, DTypeINT4, DTypeFP32}

// StoredBytes never reports less than the payload, grows with scale width, shrinks as more
// elements share a scale, and, for a format that defines its own layout, stating that
// layout explicitly changes nothing.
func TestStoredBytesProperties(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for i := 0; i < 2000; i++ {
		d := allDTypes[r.Intn(len(allDTypes))]
		s := WeightScale{Elements: 1 + r.Intn(1<<14), Bytes: 1 + r.Intn(4)}
		got := StoredBytes(d, &s)
		if got < d.Bytes() {
			t.Fatalf("StoredBytes(%s, %+v) = %v, below the %v payload", d, s, got, d.Bytes())
		}
		wider := s
		wider.Bytes++
		if StoredBytes(d, &wider) <= got {
			t.Fatalf("a wider scale did not raise StoredBytes(%s): %+v", d, s)
		}
		sparser := s
		sparser.Elements *= 2
		if StoredBytes(d, &sparser) >= got {
			t.Fatalf("sharing a scale more widely did not lower StoredBytes(%s): %+v", d, s)
		}
	}
	for _, d := range allDTypes {
		if is, ok := d.IntrinsicScale(); ok && StoredBytes(d, nil) != StoredBytes(d, &is) {
			t.Errorf("%s: its own layout stated explicitly changed the stored bytes", d)
		}
		if _, ok := d.IntrinsicScale(); !ok && StoredBytes(d, nil) != d.Bytes() {
			t.Errorf("%s: no layout stated and none intrinsic, yet scales were added", d)
		}
	}
}

// The global scale layout follows the global dtype and no other: for every pair of
// dtypes, a node stored at the global one carries it, and a node stored at another does
// not. A node that holds no weights has no bytes per parameter at all.
func TestWeightBytesPerParamScaleFollowsFormat(t *testing.T) {
	scale := &WeightScale{Elements: 32, Bytes: 1}
	for _, global := range allDTypes {
		for _, node := range allDTypes {
			g := &Graph{Global: GlobalShape{WeightDType: global, WeightScale: scale}}
			n := Node{Op: OpGEMM, WeightDType: node}
			want := StoredBytes(node, nil)
			if node == global {
				want = StoredBytes(node, scale)
			}
			if got := g.WeightBytesPerParam(n); got != want {
				t.Errorf("global %s, node %s: %v, want %v", global, node, got, want)
			}
		}
		g := &Graph{Global: GlobalShape{WeightDType: global, WeightScale: scale}}
		for _, op := range []Op{OpAttention, OpElementwise, OpGather, OpAllReduce, OpRecurrentUpdate} {
			if got := g.WeightBytesPerParam(Node{Op: op}); got != 0 {
				t.Errorf("%s holds no weights but has %v B per parameter", op, got)
			}
		}
	}
}

func snapshot(t *testing.T, g *Graph) string {
	t.Helper()
	b, err := yaml.Marshal(g)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// ForCapability leaves its receiver exactly as it was for every family, is idempotent
// (resolving a resolved graph changes nothing), gives a family with no override the base
// graph less its overrides, and gives a copy no later write reaches the base through.
func TestForCapabilityIsAPureOverlay(t *testing.T) {
	g := v41Graph()
	g.Head[0].Role = "fused_wqa_wkv" // the override now reaches a head node too
	g.Speculator.Head[0].Aliases = ""
	g.Speculator.Head[0].Role = "fused_wqa_wkv"
	g.LayerKinds[0].Nodes[1].Caches[1].BytesByCacheDType = map[string]int{"nvfp4_ds_mla": 288}
	g.LayerKinds[1].Nodes[0].Caches[1].BytesByCacheDType = map[string]int{"nvfp4_ds_mla": 288}
	g.ArchOverrides[0].Caches = append(g.ArchOverrides[0].Caches,
		CacheOverride{Name: "compressed", BytesByCacheDType: map[string]int{"fp8_ds_mla": 528}})
	if p := g.Validate(); !p.OK() {
		t.Fatalf("the base graph is invalid:\n%s", p.Error())
	}
	before := snapshot(t, g)

	base := *g
	base.ArchOverrides = nil
	for major := 1; major <= 12; major++ {
		r, err := g.ForCapability(major)
		if major != 9 && major != 10 {
			if err == nil {
				t.Fatalf("family %d is undescribed, yet it resolved", major)
			}
			if snapshot(t, g) != before {
				t.Fatalf("a refused ForCapability(%d) modified its receiver", major)
			}
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		if snapshot(t, g) != before {
			t.Fatalf("ForCapability(%d) modified its receiver", major)
		}
		again, err := r.ForCapability(major)
		if err != nil {
			t.Fatal(err)
		}
		if snapshot(t, again) != snapshot(t, r) {
			t.Fatalf("ForCapability(%d) is not idempotent", major)
		}
		if major == 9 && snapshot(t, r) != snapshot(t, &base) {
			t.Fatalf("family 9 is the base, yet the graph changed")
		}
		if !reflect.DeepEqual(r.BaseCapabilityFamilies, []int{major}) {
			t.Fatalf("family %d resolved to a graph describing %v", major, r.BaseCapabilityFamilies)
		}
		if _, err := r.ForCapability(19 - major); err == nil {
			t.Fatalf("a graph resolved for family %d resolved again for %d", major, 19-major)
		}
		if len(r.ArchOverrides) != 0 {
			t.Fatalf("family %d: overrides survive resolution", major)
		}
		if p := r.Validate(); !p.OK() {
			t.Fatalf("family %d resolved to an invalid graph:\n%s", major, p.Error())
		}
		// Writes through every pointer and map the copy shares a type with.
		r.Global.WeightScale.Bytes = 99
		r.Head[0].WeightDType = DTypeFP32
		r.Speculator.Head[0].WeightDType = DTypeFP32
		r.Speculator.Embedding[0].RowBytes = 1
		r.Speculator.SequentialHead[0].RowBytes = 1
		r.Embedding[0].RowBytes = 1
		r.LayerKinds[0].Nodes[1].Caches[1].BytesByCacheDType["nvfp4_ds_mla"] = 1
		r.LayerKinds[0].Nodes[5].Context.BytesPerEntry = 1
		r.Stack.Prologue[0] = "written"
		r.LayerKinds[0].Edges = append(r.LayerKinds[0].Edges[:0], [2]int{9, 9})
		r.Speculator.Stack.Pattern[0] = "written"
		r.Speculator.ContextStack.Pattern[0] = "written"
		r.BaseCapabilityFamilies[0] = 99
		if snapshot(t, g) != before {
			t.Fatalf("a write to the family-%d copy reached the base graph", major)
		}
	}
	r, _ := g.ForCapability(10)
	for _, n := range []Node{r.LayerKinds[0].Nodes[0], r.Head[0], r.Speculator.Head[0]} {
		if n.WeightDType != DTypeBF16 {
			t.Errorf("the role override missed a node: %+v", n)
		}
	}
	if got := r.LayerKinds[0].Nodes[1].Caches[1].BytesByCacheDType; got["nvfp4_ds_mla"] != 288 || got["fp8_ds_mla"] != 528 {
		t.Errorf("a cache override's records should merge over the base's: %v", got)
	}
	// Readers take the restated entry and per-dtype records, since they state what they
	// read, but never the owner's paging.
	reader := r.LayerKinds[1].Nodes[0].Caches[1]
	if reader.PageAlignment != 0 || reader.EntryBytes("fp8_ds_mla") != r.LayerKinds[0].Nodes[1].Caches[1].EntryBytes("fp8_ds_mla") {
		t.Errorf("the reader's record diverged from its owner's or took its paging: %+v", reader)
	}
}

// A chip with no compute capability yields major 0, which must not silently select the
// base graph's architecture when the graph distinguishes architectures.
func TestForCapabilityRefusesAnUnknownArchitecture(t *testing.T) {
	if _, err := v41Graph().ForCapability(0); err == nil {
		t.Error("a graph with overrides resolved for an unknown architecture")
	}
	g := v41Graph()
	g.ArchOverrides, g.BaseCapabilityFamilies = nil, nil
	for _, major := range []int{0, 9, 12} {
		if _, err := g.ForCapability(major); err != nil {
			t.Errorf("a graph that distinguishes no architectures is the same everywhere: %v", err)
		}
	}
	// Stating a base alone still limits the graph to the families it describes.
	g.BaseCapabilityFamilies = []int{9}
	if _, err := g.ForCapability(12); err == nil {
		t.Error("a graph whose base is SM90 resolved for SM120")
	}
}

func ownedPerToken(g *Graph, s Stack, dtype string) float64 {
	kinds := map[string]LayerKind{}
	for _, lk := range g.LayerKinds {
		kinds[lk.ID] = lk
	}
	var total float64
	for _, id := range s.Expand() {
		for _, n := range kinds[id].Nodes {
			for _, c := range n.Caches {
				total += c.BytesPerContextToken(dtype)
			}
		}
	}
	return total
}

// Cache sharing is by name and position, so renaming a cache consistently everywhere
// changes nothing, a second reader after an owner is still served by it, and a reader
// moved before every owner is not.
func TestCacheSharingMetamorphic(t *testing.T) {
	base := v41Graph()
	if p := base.Validate(); !p.OK() {
		t.Fatal(p.Error())
	}
	wantPerToken := ownedPerToken(base, base.Stack, "")

	renamed := v41Graph()
	for i := range renamed.LayerKinds {
		for j := range renamed.LayerKinds[i].Nodes {
			for k := range renamed.LayerKinds[i].Nodes[j].Caches {
				c := &renamed.LayerKinds[i].Nodes[j].Caches[k]
				c.Name = "renamed_" + c.Name
			}
		}
	}
	renamed.ArchOverrides[0].Caches[0].Name = "renamed_swa"
	if p := renamed.Validate(); !p.OK() {
		t.Errorf("a consistent rename broke validity:\n%s", p.Error())
	}
	if got := ownedPerToken(renamed, renamed.Stack, ""); got != wantPerToken {
		t.Errorf("a consistent rename changed residency: %v, want %v", got, wantPerToken)
	}

	more := v41Graph()
	more.Stack.Repeat = 7
	if p := more.Validate(); !p.OK() {
		t.Errorf("more readers after one owner broke validity:\n%s", p.Error())
	}
	if got := ownedPerToken(more, more.Stack, ""); got != wantPerToken {
		t.Errorf("readers added residency: %v, want %v", got, wantPerToken)
	}

	early := v41Graph()
	early.Stack = Stack{Prologue: []string{"reader", "source", "reader"}}
	if p := early.Validate(); p.OK() || !strings.Contains(p.Error(), "no earlier layer owns one") {
		t.Errorf("a reader before every owner validated:\n%s", p.Error())
	}
}

// Residency scales linearly with the entry size: doubling every owned entry doubles the
// bytes per context token, and a cache dtype a cache does not list leaves it unchanged.
func TestBytesPerContextTokenIsLinear(t *testing.T) {
	g := v41Graph()
	one := ownedPerToken(g, g.Stack, "fp8_ds_mla")
	for i := range g.LayerKinds {
		for j := range g.LayerKinds[i].Nodes {
			for k := range g.LayerKinds[i].Nodes[j].Caches {
				g.LayerKinds[i].Nodes[j].Caches[k].BytesPerEntry *= 2
			}
		}
	}
	if two := ownedPerToken(g, g.Stack, "fp8_ds_mla"); two != 2*one {
		t.Errorf("doubling every entry gave %v, want %v", two, 2*one)
	}
	if other := ownedPerToken(g, g.Stack, "not_listed"); other != 2*one {
		t.Errorf("an unlisted cache dtype should take bytes_per_entry: %v, want %v", other, 2*one)
	}
}

// The two defaults that have explicit spellings read the same either way.
func TestEffectiveDefaults(t *testing.T) {
	if Residency("").Effective() != ResidencyDevice || ResidencyHost.Effective() != ResidencyHost {
		t.Error("Residency.Effective")
	}
	if GatherShard("").Effective() != ShardReplicated || ShardHeads.Effective() != ShardHeads {
		t.Error("GatherShard.Effective")
	}
	// A default spelled out and left out validate alike.
	for _, n := range []Node{
		{Op: OpGather, Rows: 10, RowBytes: 2, RowsPerToken: 1, OutBytesPerToken: 2},
		{Op: OpGather, Rows: 10, RowBytes: 2, RowsPerToken: 1, OutBytesPerToken: 2,
			Residency: ResidencyDevice, Shard: ShardReplicated},
	} {
		g := validGraph()
		g.Embedding = []Node{n}
		if p := g.Validate(); !p.OK() {
			t.Errorf("%+v:\n%s", n, p.Error())
		}
	}
}

// DefaultWindow is the LongRoPE starting window where one is stated, else the window.
func TestDefaultWindow(t *testing.T) {
	if got := (GlobalShape{ContextWindow: 131072}).DefaultWindow(); got != 131072 {
		t.Errorf("got %d", got)
	}
	if got := (GlobalShape{ContextWindow: 131072, DefaultContextWindow: 4096}).DefaultWindow(); got != 4096 {
		t.Errorf("got %d", got)
	}
}

// A layer-kind id may contain dots; an alias splits at the last one.
func TestAliasToADottedLayerKind(t *testing.T) {
	g := v41Graph()
	g.LayerKinds[2].ID = "draft.v2"
	g.Speculator.Stack.Pattern = []string{"draft.v2"}
	g.LayerKinds[3].Nodes[0].Aliases = "draft.v2.fused_wqa_wkv"
	if p := g.Validate(); !p.OK() {
		t.Errorf("%s", p.Error())
	}
}

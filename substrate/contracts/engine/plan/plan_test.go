package plan

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/units"
)

func goodPlan() PoolPlan {
	return PoolPlan{
		Hardware: "h100", Fabric: "",
		Layout:   Layout{Tensor: 4, Pipeline: 1, Data: 1, RanksPerNode: 8, Nodes: 1},
		Layers:   []LoweredLayer{{Layer: 0, Primitive: PrimGEMM, KernelImpl: KernelImpl{Library: "lib", CodeHash: "abc"}, Shape: PrimitiveShape{M: 1, N: 4096, K: 4096}}},
		KV:       []KVGroup{{BytesPerPage: 1 << 20, TokensPerPage: 16}},
		KVBudget: []units.PerRank[units.Bytes]{units.Rank(units.Bytes(10 << 30))},
		Weights:  []ResidentTensor{{Name: "w", Elements: 1000, Format: units.QuantFormat{PayloadBits: 16}}},
		Buffers:  []Buffer{{Name: "ws", Bytes: 1 << 20, Basis: BasisRank}},
	}
}

func TestAGoodPlanValidates(t *testing.T) {
	if err := goodPlan().Validate(); err != nil {
		t.Fatal(err)
	}
	// The zero value of the optional parts is valid: nothing absent is an engine default.
	p := goodPlan()
	p.InFlight, p.Drafts, p.Graphs, p.Tiers, p.KVBudget = 0, DraftPolicy{}, GraphPolicy{}, nil, nil
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	if p.InFlightDepth() != 1 {
		t.Errorf("InFlightDepth of the zero value = %d, want 1", p.InFlightDepth())
	}
}

// X18: each rule has an input that breaks it.
func TestAPlanThatNoPoolCanHaveIsRefused(t *testing.T) {
	collective := func() LoweredLayer {
		return LoweredLayer{Primitive: PrimAllReduce, KernelImpl: KernelImpl{Library: "lib"}, Collective: &Collective{Group: 4, Bytes: units.Rank(units.Bytes(1024))}}
	}
	cases := []struct {
		name string
		edit func(*PoolPlan)
		want string
	}{
		{"no chip", func(p *PoolPlan) { p.Hardware = " " }, "Hardware is empty"},
		{"negative width", func(p *PoolPlan) { p.Layout.Tensor = -1 }, "Layout.Tensor is -1"},
		{"nodes without a fabric", func(p *PoolPlan) { p.Layout.Nodes = 2; p.Layout.RanksPerNode = 2 }, "Fabric is empty"},
		{"too few places", func(p *PoolPlan) { p.Layout.RanksPerNode = 1 }, "fewer than its 4 ranks"},
		{"one budget for two stages", func(p *PoolPlan) { p.Layout.Pipeline = 2 }, "one per stage"},
		{"negative budget", func(p *PoolPlan) { p.KVBudget[0] = units.Rank(units.Bytes(-1)) }, "KVBudget[0]"},
		{"unknown primitive", func(p *PoolPlan) { p.Layers[0].Primitive = "flash_attn_v3" }, "not in the fixed list"},
		{"collective without its description", func(p *PoolPlan) { l := collective(); l.Collective = nil; p.Layers = append(p.Layers, l) }, "states no group"},
		{"a non-collective that has one", func(p *PoolPlan) { p.Layers[0].Collective = &Collective{Group: 2} }, "is not a collective but states one"},
		{"collective over one rank", func(p *PoolPlan) { l := collective(); l.Collective.Group = 1; p.Layers = append(p.Layers, l) }, "group of 1"},
		{"stage beyond the pipeline", func(p *PoolPlan) { p.Layers[0].Stage = 1 }, "stage 1 of 1"},
		{"no kernel library", func(p *PoolPlan) { p.Layers[0].KernelImpl.Library = "" }, "names no kernel library"},
		{"draft layers without speculation", func(p *PoolPlan) { p.DraftLayers = p.Layers }, "configures no speculation"},
		{"a page of nothing", func(p *PoolPlan) { p.KV[0].TokensPerPage = 0 }, "both are positive"},
		{"a tier with no device", func(p *PoolPlan) { p.Tiers = []KVTier{{Bytes: units.Rank(units.Bytes(1))}} }, "no device"},
		{"unknown graph mode", func(p *PoolPlan) { p.Graphs.Mode = "magic" }, "not a known mode"},
		{"buckets without graphs", func(p *PoolPlan) { p.Graphs.Buckets = []int{1, 2} }, "captures no graphs"},
		{"a tree with no steps", func(p *PoolPlan) { p.Drafts = DraftPolicy{TreeTopK: 4} }, "a tree has both"},
		{"a chain and a tree", func(p *PoolPlan) { p.Drafts = DraftPolicy{Chain: 3, TreeTopK: 2, TreeSteps: 2} }, "is both a chain"},
		{"a malformed format", func(p *PoolPlan) { p.Weights[0].Format = units.QuantFormat{} }, "Weights[0]"},
		{"a GPU weight shared", func(p *PoolPlan) { p.Weights[0].Shared = true }, "only a host figure is shared"},
		{"a buffer with no basis", func(p *PoolPlan) { p.Buffers[0].Basis = "" }, "per rank, per node or for the group"},
		{"negative in-flight", func(p *PoolPlan) { p.InFlight = -1 }, "InFlight is -1"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := goodPlan()
			c.edit(&p)
			err := p.Validate()
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Errorf("error %v does not contain %q", err, c.want)
			}
		})
	}
}

func TestAStepThatNoEngineCanFormIsRefused(t *testing.T) {
	good := func() StepShape {
		return StepShape{
			Mode:   StepMixed,
			Ranks:  []RankStep{{Requests: []RequestStep{{Scheduled: 10, Prompt: 10}, {Scheduled: 1, Computed: 99, Prompt: 50}}}},
			Padded: []PaddedShape{{Tokens: 16, Launch: LaunchGraph}},
		}
	}
	if err := good().Validate(1, 1); err != nil {
		t.Fatal(err)
	}
	if err := (StepShape{Mode: StepIdle}).Validate(4, 1); err != nil {
		t.Errorf("an idle step names no ranks: %v", err)
	}
	cases := []struct {
		name string
		edit func(*StepShape)
		want string
	}{
		{"unknown mode", func(s *StepShape) { s.Mode = "fast" }, "not a known mode"},
		{"a rank missing from lockstep", func(s *StepShape) { s.Ranks = nil; s.Padded = nil }, "0 rank entries for a lockstep group of 1"},
		{"padding for the wrong ranks", func(s *StepShape) { s.Padded = append(s.Padded, PaddedShape{}) }, "2 padded shapes for 1 ranks"},
		{"stage beyond", func(s *StepShape) { s.Stage = 3 }, "stage 3 of 1"},
		{"a negative count", func(s *StepShape) { s.Ranks[0].Requests[0].Cached = -1 }, "negative token count"},
		{"padding that removes tokens", func(s *StepShape) { s.Padded[0].Tokens = 5 }, "padding never removes tokens"},
		{"unknown launch", func(s *StepShape) { s.Padded[0].Launch = "jit" }, `launch "jit"`},
		{"verified drafts outside a verify step", func(s *StepShape) { s.Drafts.Verified = 2 }, "only a verify step verifies"},
		{"a prebuilt step that launches", func(s *StepShape) { s.Mode = StepPrebuilt }, "launches nothing"},
		{"a negative decode threshold", func(s *StepShape) { s.DecodeQueryLen = -1 }, "DecodeQueryLen is -1"},
		{"a drafter pass for one rank of two", func(s *StepShape) {
			s.Ranks = append(s.Ranks, s.Ranks[0])
			s.Padded = append(s.Padded, s.Padded[0])
			s.Drafts.Passes = []DraftPass{{Padded: 4}}
		}, "1 drafter passes for 2 ranks"},
		{"too many drafter passes", func(s *StepShape) { s.Drafts.Passes = []DraftPass{{}, {}} }, "2 drafter passes for 1 ranks"},
		{"a negative drafter width", func(s *StepShape) { s.Drafts.Passes = []DraftPass{{Padded: -3}} }, "drafter launches -3 tokens"},
		{"an unknown drafter launch", func(s *StepShape) { s.Drafts.Passes = []DraftPass{{Padded: 4, Launch: "jit"}} }, `drafter launch "jit"`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := good()
			c.edit(&s)
			err := s.Validate(1, 1)
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Errorf("error %v does not contain %q", err, c.want)
			}
		})
	}
	// A lockstep group of two ranks needs both, since padding depends on every rank's tokens.
	if err := good().Validate(2, 1); err == nil {
		t.Error("a step describing one of two lockstep ranks was accepted")
	}
}

// A drafter pass for every rank, or none, is a step the contract allows; the zero value is none.
func TestADrafterPassForEveryRankIsAccepted(t *testing.T) {
	s := StepShape{
		Mode:   StepVerify,
		Ranks:  []RankStep{{Requests: []RequestStep{{Scheduled: 8, Prompt: 8}}}, {Requests: []RequestStep{{Scheduled: 8, Prompt: 8}}}},
		Padded: []PaddedShape{{Tokens: 8}, {Tokens: 8}},
		Drafts: DraftShape{Drafted: 7, Verified: 7, Passes: []DraftPass{{Padded: 16, Launch: LaunchGraph}, {Padded: 16, Launch: LaunchGraph}}},
	}
	if err := s.Validate(2, 1); err != nil {
		t.Errorf("a drafter pass on each rank was refused: %v", err)
	}
	s.Drafts.Passes = nil
	if err := s.Validate(2, 1); err != nil {
		t.Errorf("a step with no drafter pass was refused: %v", err)
	}
}

// DecodeQueryLen keeps a meaningful zero (design 6.2): unstated reads as 1, the least a decode is,
// and a stated value is returned as stated.
func TestTheDecodeThresholdKeepsAMeaningfulZero(t *testing.T) {
	for _, c := range []struct{ stated, want units.Tokens }{{0, 1}, {1, 1}, {4, 4}, {8, 8}} {
		if got := (StepShape{DecodeQueryLen: c.stated}).DecodeWidth(); got != c.want {
			t.Errorf("stated %d: width %d, want %d", c.stated, got, c.want)
		}
	}
	if err := (StepShape{Mode: StepIdle}).Validate(1, 1); err != nil {
		t.Errorf("the zero DecodeQueryLen was refused: %v", err)
	}
}

func TestATransferThatNoEngineCanStartIsRefused(t *testing.T) {
	good := Transfer{Tokens: 100, Kind: TransferToTier, Tier: "host-dram"}
	if err := good.Validate(); err != nil {
		t.Fatal(err)
	}
	if err := (Transfer{Tokens: 100, Kind: TransferPeer, From: Endpoint{Node: 0}, To: Endpoint{Node: 1}}).Validate(); err != nil {
		t.Fatal(err)
	}
	for name, tr := range map[string]Transfer{
		"no tokens":            {Kind: TransferToTier, Tier: "x"},
		"unknown kind":         {Tokens: 1, Kind: "teleport"},
		"offload without tier": {Tokens: 1, Kind: TransferFromTier},
		"peer with a tier":     {Tokens: 1, Kind: TransferPeer, Tier: "x"},
		"negative node":        {Tokens: 1, Kind: TransferPeer, From: Endpoint{Node: -1}},
	} {
		if err := tr.Validate(); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// SUB-14: GPU figures are per rank, host figures per node; a shared figure counts once.
func TestMemoryClosesWithBasis(t *testing.T) {
	p := goodPlan()
	p.Weights = []ResidentTensor{
		{Name: "gpu", Elements: 100, Format: units.QuantFormat{PayloadBits: 16}},                                            // 200 B per rank on GPU
		{Name: "host-private", Elements: 100, Format: units.QuantFormat{PayloadBits: 8}, Where: ResidentHost},               // 100 B per rank, host
		{Name: "host-shared", Elements: 1000, Format: units.QuantFormat{PayloadBits: 8}, Where: ResidentHost, Shared: true}, // 1000 B per node
	}
	p.Buffers = []Buffer{
		{Name: "ws", Bytes: 50, Basis: BasisRank},
		{Name: "pinned", Bytes: 7, Basis: BasisRank, Host: true},
		{Name: "pinned-node", Bytes: 30, Basis: BasisNode, Host: true},
	}
	m, err := MemoryOf(p, 4)
	if err != nil {
		t.Fatal(err)
	}
	if m.WeightsGPU.Value() != 200 || m.BuffersGPU.Value() != 50 {
		t.Errorf("GPU: weights %d buffers %d", m.WeightsGPU.Value(), m.BuffersGPU.Value())
	}
	// host: (100 + 7) per rank x 4 ranks + 1000 + 30 shared = 428 + 1030
	if want := units.Bytes(4*107 + 1030); m.Host.Value() != want {
		t.Errorf("host per node = %d, want %d", m.Host.Value(), want)
	}
	// One engine more on the node changes the unshared figure and leaves the shared one.
	m8, _ := MemoryOf(p, 8)
	if want := units.Bytes(8*107 + 1030); m8.Host.Value() != want {
		t.Errorf("host per node at 8 ranks = %d, want %d", m8.Host.Value(), want)
	}
	for name, edit := range map[string]func(*PoolPlan){
		"a host buffer for the group": func(p *PoolPlan) { p.Buffers = []Buffer{{Name: "x", Bytes: 1, Basis: BasisGroup, Host: true}} },
		"a GPU buffer per node":       func(p *PoolPlan) { p.Buffers = []Buffer{{Name: "x", Bytes: 1, Basis: BasisNode}} },
		"a buffer with no basis":      func(p *PoolPlan) { p.Buffers = []Buffer{{Name: "x", Bytes: 1}} },
	} {
		q := goodPlan()
		edit(&q)
		if _, err := MemoryOf(q, 1); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if _, err := MemoryOf(p, 0); err == nil {
		t.Error("zero ranks on a node accepted")
	}
}

func TestHostTierAndTheNodeRefusal(t *testing.T) {
	p := goodPlan()
	p.Tiers = []KVTier{
		{Device: "host-dram", Bytes: units.Rank(units.Bytes(100))},
		{Device: "host-dram", Bytes: units.Rank(units.Bytes(1000)), Shared: true},
		{Device: "nvme", Bytes: units.Rank(units.Bytes(1 << 40))},
	}
	got, err := HostTierBytes(p, 4, func(d string) bool { return d == "host-dram" })
	if err != nil || got.Value() != 4*100+1000 {
		t.Fatalf("host tier = %d, %v", got.Value(), err)
	}
	if err := CheckHostFits(got, units.Node(units.Bytes(1400))); err != nil {
		t.Errorf("exactly full was refused: %v", err)
	}
	if err := CheckHostFits(got, units.Node(units.Bytes(1399))); err == nil {
		t.Error("a node over its host memory was accepted")
	}
	if err := CheckHostFits(got, units.Node(units.Bytes(0))); err == nil {
		t.Error("an unstated host memory was accepted")
	}
}

// X2: a plan type names no engine, and a zero value is absent, never an engine default.
func TestPlanTypesNameNoEngine(t *testing.T) {
	banned := []string{"vllm", "sglang", "flashinfer", "flashmla", "trtllm", "cutlass", "deepep", "nixl", "mooncake"}
	seen := map[reflect.Type]bool{}
	var walk func(t reflect.Type, path string)
	walk = func(tp reflect.Type, path string) {
		for tp.Kind() == reflect.Pointer || tp.Kind() == reflect.Slice || tp.Kind() == reflect.Array {
			tp = tp.Elem()
		}
		if tp.Kind() != reflect.Struct || seen[tp] || !strings.Contains(tp.PkgPath(), "blis-substrate") {
			return
		}
		seen[tp] = true
		for i := 0; i < tp.NumField(); i++ {
			f := tp.Field(i)
			low := strings.ToLower(path + "." + f.Name)
			for _, b := range banned {
				if strings.Contains(low, b) {
					t.Errorf("%s names an engine or library (%q)", path+"."+f.Name, b)
				}
			}
			walk(f.Type, path+"."+f.Name)
		}
	}
	for _, v := range []any{PoolPlan{}, StepShape{}, Transfer{}} {
		walk(reflect.TypeOf(v), reflect.TypeOf(v).Name())
	}
	if len(seen) < 15 {
		t.Fatalf("walked %d types; the walk is not reaching the plan's parts", len(seen))
	}
	for _, v := range []any{PoolPlan{}, StepShape{}, Transfer{}} {
		if reflect.ValueOf(v).IsZero() != true {
			t.Errorf("%T: the zero value is not zero", v)
		}
	}
	// The zero value of every mode-like type is the neutral one.
	if !GraphNone.Valid() || !LaunchEager.Valid() || !ResidentGPU.Valid() || !FrontendNeutral() {
		t.Error("a zero enumeration is not valid")
	}
}

func FrontendNeutral() bool { return !(HostPlan{}).NativeFrontend }

func TestAPlanSurvivesJSON(t *testing.T) {
	p := goodPlan()
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	var back PoolPlan
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(p, back) {
		t.Errorf("round trip changed the plan:\n%+v\n%+v", p, back)
	}
}

// The implementations a pool runs are what the coefficient lookup matches against (VAL-8):
// each once, from the target's layers and the drafter's, in a fixed order.
func TestImplementationsAreTheDistinctOnesThePoolRuns(t *testing.T) {
	a := KernelImpl{Library: "nccl", Variant: "ncclAllReduce", CodeHash: "aaaaaaaaaaaa", Toolkit: "cuda13.0"}
	b := KernelImpl{Library: "flashinfer", Variant: "decode", CodeHash: "bbbbbbbbbbbb"}
	c := KernelImpl{Library: "nccl", Variant: "ncclAllGather", CodeHash: "aaaaaaaaaaaa"}
	p := PoolPlan{
		Layers:      []LoweredLayer{{KernelImpl: a}, {KernelImpl: b}, {KernelImpl: a}},
		DraftLayers: []LoweredLayer{{KernelImpl: c}, {KernelImpl: b}},
	}
	got := p.Implementations()
	want := []KernelImpl{b, c, a}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
	if len((PoolPlan{}).Implementations()) != 0 {
		t.Errorf("a plan with no layers runs no implementation")
	}
}

// The coefficient lookup binds to the implementations of the primitive it prices, so a pool that
// runs two kernels reports each under its own primitive.
func TestImplementationsByPrimitiveKeepsEachKernelWithItsPrimitive(t *testing.T) {
	a := KernelImpl{Library: "flashinfer", Variant: "decode", CodeHash: "aaaaaaaaaaaa"}
	b := KernelImpl{Library: "cublas", Variant: "gemm", CodeHash: "bbbbbbbbbbbb"}
	c := KernelImpl{Library: "cublas", Variant: "gemm", CodeHash: "cccccccccccc"}
	p := PoolPlan{
		Layers: []LoweredLayer{{Primitive: PrimAttentionDecode, KernelImpl: a}, {Primitive: PrimGEMM, KernelImpl: b},
			{Primitive: PrimAttentionDecode, KernelImpl: a}},
		DraftLayers: []LoweredLayer{{Primitive: PrimGEMM, KernelImpl: c}, {Primitive: PrimGEMM, KernelImpl: b}},
	}
	got := p.ImplementationsByPrimitive()
	if len(got) != 2 || len(got[PrimAttentionDecode]) != 1 || got[PrimAttentionDecode][0] != a {
		t.Fatalf("attention: %v", got)
	}
	if g := got[PrimGEMM]; len(g) != 2 || g[0] != b || g[1] != c {
		t.Fatalf("gemm: %v, want the two builds sorted", g)
	}
	if len((PoolPlan{}).ImplementationsByPrimitive()) != 0 {
		t.Errorf("a plan with no layers runs nothing")
	}
}

// The operations a pool lowers keep the layer, the primitive and which graph they belong to, so
// a coefficient can be bound to the kernel that consumes it, not to every kernel of its primitive.
func TestOperationsKeepWhichLayerAndGraphRunsWhichKernel(t *testing.T) {
	a := KernelImpl{Library: "lib", Variant: "a", CodeHash: "aaaaaaaaaaaa"}
	b := KernelImpl{Library: "lib", Variant: "b", CodeHash: "bbbbbbbbbbbb"}
	c := KernelImpl{Library: "lib", Variant: "c", CodeHash: "cccccccccccc"}
	p := PoolPlan{
		Layers: []LoweredLayer{
			{Layer: 0, Primitive: PrimAttentionDecode, KernelImpl: a},
			{Layer: 1, Primitive: PrimAttentionDecode, KernelImpl: b},
			{Layer: 1, Primitive: PrimGEMM, KernelImpl: a},
		},
		DraftLayers: []LoweredLayer{{Layer: 0, Primitive: PrimAttentionDecode, KernelImpl: c}},
	}
	want := []Operation{
		{Layer: 0, Primitive: PrimAttentionDecode, Impl: a},
		{Layer: 1, Primitive: PrimAttentionDecode, Impl: b},
		{Layer: 1, Primitive: PrimGEMM, Impl: a},
		{Draft: true, Layer: 0, Primitive: PrimAttentionDecode, Impl: c},
	}
	if got := p.Operations(); !reflect.DeepEqual(got, want) {
		t.Errorf("operations %+v, want %+v", got, want)
	}
	// The pooled form is the projection of the operations.
	by := p.ImplementationsByPrimitive()
	if !reflect.DeepEqual(by[PrimAttentionDecode], []KernelImpl{a, b, c}) || !reflect.DeepEqual(by[PrimGEMM], []KernelImpl{a}) {
		t.Errorf("by primitive %+v", by)
	}
	if !reflect.DeepEqual(ImplementationsOf(want), by) {
		t.Errorf("ImplementationsOf differs from ImplementationsByPrimitive")
	}
}

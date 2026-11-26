package cmd

import (
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/hardware"
	"github.com/inference-sim/inference-sim/substrate/contracts/vocab"
)

// runCatalogCLI runs `blis catalog ...` on the pinned catalog. It does not go through
// runKernelCLI, which also names a registry: a catalog command reads none.
func runCatalogCLI(t *testing.T, args ...string) string {
	t.Helper()
	_, catalog, _ := kernelRepos(t)
	full := append(args, "--catalog", catalog)
	cmd := exec.Command(os.Args[0], "-test.run=^TestKernelCLILeg$")
	cmd.Env = append(os.Environ(), kernelCLIArgsEnv+"="+strings.Join(full, "\x1f"))
	out, err := cmd.Output()
	if err != nil {
		stderr := ""
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			stderr = string(exit.Stderr)
		}
		t.Fatalf("blis %s: %v\n%s", strings.Join(args, " "), err, lastLines(stderr, 5))
	}
	return string(out)
}

// catalogListingOfCLI is `blis catalog list --json`'s document, parsed.
func catalogListingOfCLI(t *testing.T) catalogListing {
	t.Helper()
	var l catalogListing
	if err := json.Unmarshal([]byte(runCatalogCLI(t, "catalog", "list", "--json")), &l); err != nil {
		t.Fatalf("blis catalog list --json is not JSON: %v", err)
	}
	return l
}

// modelNamed is the listing's record for one model.
func modelNamed(t *testing.T, l catalogListing, name string) catalogModel {
	t.Helper()
	i := slices.IndexFunc(l.Models, func(m catalogModel) bool { return m.Name == name })
	if i < 0 {
		t.Fatalf("the listing has no model %q", name)
	}
	return l.Models[i]
}

// Every entity class is listed, sorted by name, and each record states the facts that make it
// usable: a model's stack shape, a chip's memory and compute, a fabric's bandwidth.
func TestCatalogListReportsEveryEntity(t *testing.T) {
	l := catalogListingOfCLI(t)
	if len(l.Models) == 0 || len(l.Chips) == 0 || len(l.Fabrics) == 0 {
		t.Fatalf("%d models, %d chips, %d fabrics, want each class listed", len(l.Models), len(l.Chips), len(l.Fabrics))
	}
	for _, names := range [][]string{
		modelNames(l.Models),
		chipNames(l.Chips),
		fabricNames(l.Fabrics),
	} {
		if !slices.IsSorted(names) {
			t.Errorf("a class is not sorted by name: %v", names)
		}
	}
	for _, m := range l.Models {
		if m.Name == "" || m.Layers <= 0 || m.HiddenSize <= 0 || m.VocabSize <= 0 {
			t.Errorf("model %q states %d layers, hidden %d, vocab %d", m.Name, m.Layers, m.HiddenSize, m.VocabSize)
		}
	}
	for _, c := range l.Chips {
		if c.Name == "" || c.MemoryGiB <= 0 || c.MemoryBandwidthTBs <= 0 || c.BF16TFlops <= 0 || c.SMCount <= 0 {
			t.Errorf("chip %q states %.2f GiB, %.2f TB/s, %.1f TFLOP/s bf16, %d SMs",
				c.Name, c.MemoryGiB, c.MemoryBandwidthTBs, c.BF16TFlops, c.SMCount)
		}
	}
	for _, f := range l.Fabrics {
		if f.Name == "" || f.InterNodeBwGBps <= 0 {
			t.Errorf("fabric %q states %.2f GB/s", f.Name, f.InterNodeBwGBps)
		}
	}
}

// A model's record is the graph's own shape: the attention heads and expert routing the layer
// kinds state, not a re-reading of the vendor configuration.
func TestCatalogListStatesTheGraphsShape(t *testing.T) {
	l := catalogListingOfCLI(t)

	// A dense grouped-query model: no experts, and the head shape its attention node states.
	llama := modelNamed(t, l, "codellama-34b-instruct-hf")
	if llama.Source == nil || llama.Source.Repo != "codellama/CodeLlama-34b-Instruct-hf" {
		t.Errorf("source %+v, want the entry's model.yaml repository", llama.Source)
	}
	if llama.Architecture != "LlamaForCausalLM" || llama.Layers != 48 || llama.HiddenSize != 8192 {
		t.Errorf("architecture %q, %d layers, hidden %d", llama.Architecture, llama.Layers, llama.HiddenSize)
	}
	if llama.Attention == nil {
		t.Fatal("a grouped-query model states no attention")
	}
	if *llama.Attention != (catalogAttention{Kind: "gqa", Heads: 64, KVHeads: 8, HeadDim: 128}) {
		t.Errorf("attention %+v", *llama.Attention)
	}
	if llama.Experts != nil {
		t.Errorf("a dense model states experts %+v", *llama.Experts)
	}

	// A routed mixture: the expert count and the number a token activates.
	moe := modelNamed(t, l, "deepseek-v2-lite")
	if moe.Experts == nil || *moe.Experts != (catalogExperts{Count: 64, TopK: 6}) {
		t.Errorf("experts %+v, want 64 routed with top-k 6", moe.Experts)
	}

	// A latent-attention model states the widths its layer states: the latent width, and the query
	// and KV head counts the node also states (64 query heads sharing one KV head).
	mla := modelNamed(t, l, "deepseek-v4.1-flash")
	if want := (catalogAttention{Kind: "sparse_mla", Heads: 64, KVHeads: 1, HeadDim: 512}); mla.Attention == nil || *mla.Attention != want {
		t.Errorf("attention %+v, want %+v", mla.Attention, want)
	}
	// The context window is the graph's, where it states one: this model's is 1M tokens.
	if mla.ContextWindow != 1048576 {
		t.Errorf("context_window %d, want the graph's 1048576", mla.ContextWindow)
	}
	// Its drafter is the entry's, as `catalog resolve` reports it.
	if len(mla.Drafters) != 1 || mla.Drafters[0].Name != "speculator" ||
		mla.Drafters[0].Method == nil || *mla.Drafters[0].Method != "dspark" ||
		mla.Drafters[0].NumSpecTokens == nil || *mla.Drafters[0].NumSpecTokens != 5 {
		t.Errorf("drafters %+v, want the entry's in-graph dspark speculator", mla.Drafters)
	}

	// A model whose graph states no context window reports none, rather than a figure another
	// source holds: the catalog does not state one, so the deployment does.
	if llama.ContextWindow != 0 {
		t.Errorf("a model whose graph states no context window reports %d", llama.ContextWindow)
	}
}

// A chip's record is its datasheet figures, and its provenance map is keyed by the names this
// record uses, so a reader can tell which figures came from another source than the file's own.
// A chip's quant peaks, packaging and rack tier are pinned by value, not by sign: a field swap
// among the peaks would pass a "> 0" check.
func TestCatalogListStatesChipDatasheetValues(t *testing.T) {
	l := catalogListingOfCLI(t)
	b200 := chipNamed(t, l, "b200")
	if got, want := [4]float64{b200.BF16TFlops, b200.FP8TFlops, b200.NVFP4TFlops, b200.MemoryBandwidthTBs},
		[4]float64{2250, 4500, 9000, 7.7}; got != want {
		t.Errorf("b200 bf16/fp8/nvfp4/bandwidth %v, want %v", got, want)
	}
	// b200 states no packaging: the node's GPU count is the deployment's, so both are null rather
	// than a claimed zero, in this command and in `catalog resolve` alike.
	if b200.Packaging.GPUsPerNode != nil || b200.Packaging.GPUsPerRack != nil {
		t.Errorf("b200 packaging %+v, want both counts unstated", b200.Packaging)
	}

	// gb300 is a rack-scale part: it fixes both counts and states its within-rack tier.
	gb300 := chipNamed(t, l, "gb300")
	if gb300.Packaging.GPUsPerNode == nil || *gb300.Packaging.GPUsPerNode != 4 ||
		gb300.Packaging.GPUsPerRack == nil || *gb300.Packaging.GPUsPerRack != 72 {
		t.Errorf("gb300 packaging %+v, want 4 per node and 72 per rack", gb300.Packaging)
	}
	if gb300.IntraRackBwGBps != 900 {
		t.Errorf("gb300 intra_rack_bw_gbps %v, want the catalog's 900", gb300.IntraRackBwGBps)
	}
}

// A fabric's record is its nominal bandwidth and whether it keeps the CPU out of the data path;
// both classes are pinned by value.
func TestCatalogListStatesTheFabricContract(t *testing.T) {
	l := catalogListingOfCLI(t)
	byName := map[string]catalogFabric{}
	for _, f := range l.Fabrics {
		byName[f.Name] = f
	}
	eth, ok := byName["ethernet-100gbe"]
	if !ok {
		t.Fatal("the listing has no ethernet-100gbe")
	}
	if eth.InterNodeBwGBps != 12.5 || eth.RDMA {
		t.Errorf("ethernet-100gbe is %v GB/s with rdma %t, want 12.5 and no RDMA", eth.InterNodeBwGBps, eth.RDMA)
	}
	if ib, ok := byName["ib-400g"]; !ok || ib.InterNodeBwGBps != 50 {
		t.Errorf("ib-400g %+v, want the catalog's 50 GB/s", ib)
	}
}

func TestCatalogListStatesChipProvenance(t *testing.T) {
	l := catalogListingOfCLI(t)
	i := slices.IndexFunc(l.Chips, func(c catalogChip) bool { return c.Name == "h200" })
	if i < 0 {
		t.Fatal("the listing has no h200")
	}
	c := l.Chips[i]
	if got, want := c.MemoryGiB, 139.80072021; got != want {
		t.Errorf("memory_gib %v, want the catalog's %v", got, want)
	}
	if c.FieldProvenance["memory_gib"] == "" {
		t.Errorf("field_provenance %v does not name memory_gib's source", c.FieldProvenance)
	}
	if _, unmapped := c.FieldProvenance["MemoryGiB"]; unmapped {
		t.Errorf("field_provenance is keyed by the catalog's field names: %v", c.FieldProvenance)
	}
	if c.SMCount != 132 || c.ComputeCapability != "9.0" || c.HostCPU != "x86" {
		t.Errorf("sm_count %d, compute %q, host %q", c.SMCount, c.ComputeCapability, c.HostCPU)
	}
	// Board power is a per-GPU figure the entry states, and it needs no field_provenance
	// entry because the file's own provenance covers it.
	if c.TDPWatts != 700 {
		t.Errorf("tdp_watts %v, want the catalog's 700", c.TDPWatts)
	}
}

// A chip's board power is a packaging maximum a client prices electricity with. It is reported
// under its own key, and a chip whose entry states none leaves it out so a client reads the
// omission as unavailable rather than as a zero-watt part. A field_provenance entry naming the
// catalog's TDPWatts is rekeyed like every other fact, so a client can read its source.
func TestCatalogChipReportsStatedBoardPower(t *testing.T) {
	stated := catalogChip{Name: "h200", chipFacts: chipFactsOf(&hardware.Chip{
		Name: "h200", TDPWatts: 700,
		FieldProvenance: map[string]string{"TDPWatts": "vendor_spec"},
	})}
	var fields map[string]any
	raw, err := json.Marshal(stated)
	if err != nil {
		t.Fatalf("marshalling a chip: %v", err)
	}
	if err := json.Unmarshal(raw, &fields); err != nil {
		t.Fatalf("unmarshalling a chip: %v", err)
	}
	if fields["tdp_watts"] != 700.0 {
		t.Errorf("tdp_watts = %v, want 700 under its own key: %s", fields["tdp_watts"], raw)
	}
	// The provenance map is keyed by this record's field names, not the catalog's.
	prov, ok := fields["field_provenance"].(map[string]any)
	if !ok || prov["tdp_watts"] != "vendor_spec" {
		t.Errorf("field_provenance = %v, want tdp_watts keyed by this record's name", fields["field_provenance"])
	}
	if _, unmapped := prov["TDPWatts"]; unmapped {
		t.Errorf("field_provenance is keyed by the catalog's field names: %v", prov)
	}

	unstated, err := json.Marshal(catalogChip{Name: "x", chipFacts: chipFactsOf(&hardware.Chip{Name: "x"})})
	if err != nil {
		t.Fatalf("marshalling a chip: %v", err)
	}
	if strings.Contains(string(unstated), "tdp_watts") {
		t.Errorf("a chip stating no board power reports one: %s", unstated)
	}
}

// Every shipped chip states a board power: each catalog hardware entry names the vendor's maximum,
// so a chip reporting none is an omission to fix rather than a part with no rating.
func TestCatalogListStatesBoardPowerForEveryChip(t *testing.T) {
	l := catalogListingOfCLI(t)
	if len(l.Chips) == 0 {
		t.Fatal("the listing has no chips")
	}
	for _, c := range l.Chips {
		if c.TDPWatts <= 0 {
			t.Errorf("%s reports no tdp_watts, want the board power its entry states", c.Name)
		}
	}
}

// The text form names every entity, so the command is useful without --json.
func TestCatalogListPrintsInText(t *testing.T) {
	out := runCatalogCLI(t, "catalog", "list")
	for _, want := range []string{"models: ", "chips: ", "fabrics: ", "codellama-34b-instruct-hf", "h200", "ib-400g", "700 W"} {
		if !strings.Contains(out, want) {
			t.Errorf("the text listing does not contain %q:\n%s", want, out)
		}
	}
}

func modelNames(models []catalogModel) []string {
	out := make([]string, len(models))
	for i, m := range models {
		out[i] = m.Name
	}
	return out
}

func chipNames(chips []catalogChip) []string {
	out := make([]string, len(chips))
	for i, c := range chips {
		out[i] = c.Name
	}
	return out
}

func fabricNames(fabrics []catalogFabric) []string {
	out := make([]string, len(fabrics))
	for i, f := range fabrics {
		out[i] = f.Name
	}
	return out
}

// chipNamed is the listing's record for one chip.
func chipNamed(t *testing.T, l catalogListing, name string) catalogChip {
	t.Helper()
	i := slices.IndexFunc(l.Chips, func(c catalogChip) bool { return c.Name == name })
	if i < 0 {
		t.Fatalf("the listing has no chip %q", name)
	}
	return l.Chips[i]
}

// chipFactsOf maps every catalog field once, to its own name: distinct values catch a swap, and a
// count the entry does not state stays null in both commands' shared projection.
func TestChipFactsMapEachCatalogFieldOnce(t *testing.T) {
	c := &hardware.Chip{
		Name:               "test",
		Provenance:         vocab.ProvenanceVendorSpec,
		FieldProvenance:    map[string]string{"TFlopsFP8": "measured:test"},
		BF16Peak:           1,
		FP8Peak:            2,
		NVFP4Peak:          3,
		MemoryGiB:          4,
		MemoryBandwidthTBs: 5,
		IntraNodeBwGBps:    6,
		IntraRackBwGBps:    7,
		SMCount:            8,
		ComputeCapability:  "9.0",
		HostCPU:            "x86",
	}
	f := chipFactsOf(c)
	if got, want := [7]float64{f.BF16TFlops, f.FP8TFlops, f.NVFP4TFlops, f.MemoryGiB,
		f.MemoryBandwidthTBs, f.IntraNodeBwGBps, f.IntraRackBwGBps}, [7]float64{1, 2, 3, 4, 5, 6, 7}; got != want {
		t.Errorf("figures %v, want each catalog field under its own name %v", got, want)
	}
	if f.SMCount != 8 || f.ComputeCapability != "9.0" || f.HostCPU != "x86" ||
		f.Provenance != string(vocab.ProvenanceVendorSpec) {
		t.Errorf("identity fields %+v", f)
	}
	if f.FieldProvenance["fp8_tflops"] != "measured:test" {
		t.Errorf("field_provenance %v, want fp8_tflops's source", f.FieldProvenance)
	}
	if f.Packaging.GPUsPerNode != nil || f.Packaging.GPUsPerRack != nil {
		t.Errorf("an entry stating no packaging reports %+v, want both null", f.Packaging)
	}
	c.GPUsPerNode, c.GPUsPerRack = 4, 72
	f = chipFactsOf(c)
	if f.Packaging.GPUsPerNode == nil || *f.Packaging.GPUsPerNode != 4 ||
		f.Packaging.GPUsPerRack == nil || *f.Packaging.GPUsPerRack != 72 {
		t.Errorf("packaging %+v, want 4 and 72", f.Packaging)
	}
}

// The fabric projection carries RDMA both ways and rekeys its per-field provenance, as a chip's.
func TestFabricFactsCarryRDMAAndProvenance(t *testing.T) {
	rdma := catalogFabricOf(&hardware.Fabric{
		Name: "ib-test", Provenance: vocab.ProvenanceVendorSpec, InterNodeBwGBps: 50, RDMA: true,
		FieldProvenance: map[string]string{"InterNodeBwGBps": "published"},
	})
	if !rdma.RDMA || rdma.InterNodeBwGBps != 50 {
		t.Errorf("an RDMA fabric reports rdma %t and %v GB/s", rdma.RDMA, rdma.InterNodeBwGBps)
	}
	if rdma.FieldProvenance["inter_node_bw_gbps"] != "published" {
		t.Errorf("field_provenance %v, want inter_node_bw_gbps's source", rdma.FieldProvenance)
	}
	if _, unmapped := rdma.FieldProvenance["InterNodeBwGBps"]; unmapped {
		t.Errorf("field_provenance is keyed by the catalog's field names: %v", rdma.FieldProvenance)
	}
	if tcp := catalogFabricOf(&hardware.Fabric{Name: "eth-test", InterNodeBwGBps: 12.5}); tcp.RDMA {
		t.Error("a fabric that does not state RDMA reports it")
	}
}

// A drafter the catalog ships no graph for is named without facts, rather than with an empty method
// and a zero draft length that read as stated values.
func TestDrafterInfoOmitsFactsTheCatalogDoesNotState(t *testing.T) {
	got, err := json.Marshal(drafterInfo{Name: "some-drafter"})
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != `{"name":"some-drafter"}` {
		t.Errorf("a factless drafter marshals as %s", got)
	}
}

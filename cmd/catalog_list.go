package cmd

import (
	"fmt"
	"io"

	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/hardware"
	specmodel "github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
	"github.com/inference-sim/inference-sim/substrate/load"
)

// catalogListCmd prints the catalog this build reads: every model it can price, every chip and
// every inter-node fabric. A client that needs the support facts a scenario may name -- a picker's
// model list with the shape facts it shows beside each name, a chip's datasheet figures, the
// fabrics a cluster may state -- reads them here rather than walking the catalog's file layout,
// which is a substrate detail and not part of any contract.
//
// It reports a catalog fact only. A figure the catalog does not state is reported as zero or
// omitted, never filled in from another source: what a deployment can do with a model on a chip is
// the engine's to say (`blis version`), and a number the catalog leaves out is not inferred here.
var catalogListCmd = &cobra.Command{
	Use:   "list",
	Short: "List the models, chips and fabrics of the catalog in use",
	Long: "List every model, chip and inter-node fabric in the catalog this build reads: the one " +
		"embedded in the substrate release it pins, or the directory --catalog or BLIS_CATALOG " +
		"names. A model is listed when the catalog ships its derived graph, that is, when this " +
		"build can price it, and its record states what the graph and the entry's model.yaml " +
		"state: the source repository and revision, the architecture, the context window the " +
		"engine admits, the stack's layer count and the global shapes, the attention heads and " +
		"expert routing of the layer kinds that state them (omitted where the model has none), " +
		"and the drafters the entry lists. A chip's record states its datasheet figures " +
		"(compute, memory, bandwidth, and the board power where the entry states one), its " +
		"packaging (gpus_per_node and gpus_per_rack, each null where the entry leaves the " +
		"count to the scenario), compute capability and host CPU, with its provenance and any " +
		"per-field provenance. A fabric's record states its nominal inter-node bandwidth, " +
		"whether it keeps the CPU out of the data path, and its provenance. With --json the " +
		"output is {\"models\": [...], \"chips\": [...], \"fabrics\": [...]}, each sorted by name.",
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, _ []string) error {
		data, err := catalogData()
		if err != nil {
			return err
		}
		listing, err := catalogListingOf(data)
		if err != nil {
			return err
		}
		if substrateJSON {
			return writeJSON(cmd.OutOrStdout(), listing)
		}
		return printCatalogListing(cmd.OutOrStdout(), listing)
	},
}

// catalogListing is catalog list's document: the catalog's three entity classes, each sorted by
// name.
type catalogListing struct {
	Models  []catalogModel  `json:"models"`
	Chips   []catalogChip   `json:"chips"`
	Fabrics []catalogFabric `json:"fabrics"`
}

// catalogModel is one priceable model: the entry's identity, and the shape facts its graph states.
//
// The numbers are the graph's, not the vendor configuration's: the graph is what this build prices,
// so a client that shows a model's head count or context window beside a prediction shows the
// figures the prediction used. ContextWindow is the longest context the engine admits where the
// deployment states no other limit, which is not always the configuration's
// max_position_embeddings (a RoPE factor scales it, a sliding window caps it).
type catalogModel struct {
	Name string `json:"name"`
	// Source is the vendor repository the entry was derived from, and its pinned revision; absent
	// for an entry whose model.yaml states none.
	Source *catalogModelSource `json:"source,omitempty"`
	// Architecture is the model class the engine serves the checkpoint with, empty where the graph
	// states none.
	Architecture  string `json:"architecture,omitempty"`
	ContextWindow int    `json:"context_window,omitempty"`
	Layers        int    `json:"layers"`
	HiddenSize    int    `json:"hidden_size"`
	VocabSize     int    `json:"vocab_size"`
	WeightDType   string `json:"weight_dtype,omitempty"`
	// Attention is the KV-holding attention layer kind's head shape, absent for a model with no
	// attention layer at all (a pure state-space stack).
	Attention *catalogAttention `json:"attention,omitempty"`
	// Experts is the routed experts of the first layer kind that states any, absent for a dense
	// model.
	Experts *catalogExperts `json:"experts,omitempty"`
	// Drafters is what the entry lists as able to draft for this model, as `blis catalog resolve`
	// reports it: omitted for a model that lists none.
	Drafters []drafterInfo `json:"drafters,omitempty"`
}

// catalogModelSource is a model's vendor origin.
type catalogModelSource struct {
	Provider string `json:"provider"`
	Repo     string `json:"repo"`
	Revision string `json:"revision,omitempty"`
}

// catalogAttention is the head shape of the attention layer kind that reads the engine's KV cache:
// the head counts and head width its node states. A zero field is one the graph does not state.
//
// A latent-attention node states the same fields, not a different set: an MLA layer's n_q query
// heads share one KV head (n_kv 1) whose latent width is d_h, so the record reports all three
// rather than implying the counts are a grouped-query-only view.
type catalogAttention struct {
	Kind    string `json:"kind,omitempty"`
	Heads   int    `json:"heads,omitempty"`
	KVHeads int    `json:"kv_heads,omitempty"`
	HeadDim int    `json:"head_dim,omitempty"`
}

// catalogExperts is a routed mixture of experts: how many experts a token chooses among (Count)
// and how many it activates (TopK). Shared is the shared_experts count the routed node itself
// states, experts every token passes through inside that node; it is not a count of separate
// shared-expert nodes, which this record does not include. No shipped catalog graph states
// shared_experts, so it is 0 there, and a model whose shared experts are separate nodes (a
// GroupedGEMM with shared true) also reports 0: the value is what the routed node states, not a
// claim that the model has none.
type catalogExperts struct {
	Count  int `json:"count"`
	TopK   int `json:"top_k"`
	Shared int `json:"shared,omitempty"`
}

// catalogChip is one chip's record in the listing: the shared chip facts, under the chip's name.
type catalogChip struct {
	Name string `json:"name"`
	chipFacts
}

// catalogFabric is one inter-node network class. A fabric is a property of the cluster a deployment
// runs on, not of a chip, so it is listed on its own.
type catalogFabric struct {
	Name       string `json:"name"`
	Provenance string `json:"provenance,omitempty"`
	// FieldProvenance names the fields whose source differs from Provenance, keyed by this
	// record's own field names, as a chip's is.
	FieldProvenance map[string]string `json:"field_provenance,omitempty"`
	// InterNodeBwGBps is per-GPU unidirectional and nominal: a line rate, not a measurement.
	InterNodeBwGBps float64 `json:"inter_node_bw_gbps"`
	// RDMA reports whether the fabric keeps the CPU out of the data path.
	RDMA bool `json:"rdma"`
}

// catalogListingOf builds the document from the catalog in use.
func catalogListingOf(data *load.Data) (catalogListing, error) {
	out := catalogListing{
		Models:  []catalogModel{},
		Chips:   []catalogChip{},
		Fabrics: []catalogFabric{},
	}
	for _, name := range data.Models() {
		m, err := catalogModelOf(data, name)
		if err != nil {
			return catalogListing{}, err
		}
		out.Models = append(out.Models, m)
	}
	for _, name := range data.Chips() {
		c, err := data.Chip(name)
		if err != nil {
			return catalogListing{}, err
		}
		out.Chips = append(out.Chips, catalogChip{Name: c.Name, chipFacts: chipFactsOf(c)})
	}
	for _, name := range data.Fabrics() {
		f, err := data.Fabric(name)
		if err != nil {
			return catalogListing{}, err
		}
		out.Fabrics = append(out.Fabrics, catalogFabricOf(f))
	}
	return out, nil
}

// catalogModelOf is one model's record: its entry's identity, its graph's global shapes, and the
// head and expert shapes of the layer kinds that state them.
func catalogModelOf(data *load.Data, name string) (catalogModel, error) {
	g, err := data.Model(name)
	if err != nil {
		return catalogModel{}, err
	}
	drafters, err := draftersOf(data, name)
	if err != nil {
		return catalogModel{}, err
	}
	m := catalogModel{
		Name:          name,
		Architecture:  g.Global.Architecture,
		ContextWindow: g.Global.ContextWindow,
		Layers:        g.Stack.Layers(),
		HiddenSize:    g.Global.HiddenSize,
		VocabSize:     g.Global.VocabSize,
		WeightDType:   string(g.Global.WeightDType),
		Drafters:      drafters,
	}
	id, err := data.ModelIdentity(name)
	if err != nil {
		return catalogModel{}, err
	}
	if id.Source != nil {
		m.Source = &catalogModelSource{Provider: id.Source.Provider, Repo: id.Source.Repo, Revision: id.Source.Revision}
	}
	m.Attention, m.Experts = layerFactsOf(g)
	return m, nil
}

// layerFactsOf is the attention and expert shapes a graph's layer kinds state: the first kind's
// KV-holding attention node, and the first kind's routed experts. Absent where the model has none.
func layerFactsOf(g *specmodel.Graph) (*catalogAttention, *catalogExperts) {
	var attn *catalogAttention
	var experts *catalogExperts
	for _, kind := range g.LayerKinds {
		for _, n := range kind.Nodes {
			if attn == nil && n.ReadsEngineKV() {
				attn = &catalogAttention{
					Kind:    string(n.AttentionKind),
					Heads:   n.NumQHeads,
					KVHeads: n.NumKVHeads,
					HeadDim: n.HeadDim,
				}
			}
			if experts == nil && n.Op == specmodel.OpGroupedGEMM && !n.IsSharedExpert() {
				experts = &catalogExperts{Count: n.Experts, TopK: n.TopK, Shared: n.SharedExperts}
			}
		}
	}
	return attn, experts
}

// catalogFabricOf is one fabric's record: its provenance, rekeyed as a chip's is, its nominal
// inter-node bandwidth, and whether it keeps the CPU out of the data path.
func catalogFabricOf(f *hardware.Fabric) catalogFabric {
	out := catalogFabric{
		Name:            f.Name,
		Provenance:      string(f.Provenance),
		InterNodeBwGBps: f.InterNodeBwGBps,
		RDMA:            f.RDMA,
	}
	if len(f.FieldProvenance) > 0 {
		out.FieldProvenance = make(map[string]string, len(f.FieldProvenance))
		for k, v := range f.FieldProvenance {
			out.FieldProvenance[fabricFieldName(k)] = v
		}
	}
	return out
}

// fabricFieldName maps a catalog fabric field's wire name to the name this record reports it
// under; an unmapped field keeps its own name, as chipFieldName does.
func fabricFieldName(field string) string {
	if name, ok := fabricFieldNames[field]; ok {
		return name
	}
	return field
}

var fabricFieldNames = map[string]string{
	"InterNodeBwGBps": "inter_node_bw_gbps",
	"RDMA":            "rdma",
}

// printCatalogListing writes the text form: a count and one line a record, so the command is
// useful without --json.
func printCatalogListing(w io.Writer, l catalogListing) error {
	if _, err := fmt.Fprintf(w, "models: %d\n", len(l.Models)); err != nil {
		return err
	}
	for _, m := range l.Models {
		if _, err := fmt.Fprintf(w, "  %s  %s  %d layers, %d context, %s\n",
			m.Name, m.Architecture, m.Layers, m.ContextWindow, m.WeightDType); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "chips: %d\n", len(l.Chips)); err != nil {
		return err
	}
	for _, c := range l.Chips {
		// Board power is printed only where the entry states it: an unstated figure is a
		// property of the part the catalog leaves out, not a zero-watt part.
		power := ""
		if c.TDPWatts > 0 {
			power = fmt.Sprintf(", %g W", c.TDPWatts)
		}
		if _, err := fmt.Fprintf(w, "  %s  %.2f GiB, %.2f TB/s, %.1f TFLOP/s bf16, %d SMs%s\n",
			c.Name, c.MemoryGiB, c.MemoryBandwidthTBs, c.BF16TFlops, c.SMCount, power); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "fabrics: %d\n", len(l.Fabrics)); err != nil {
		return err
	}
	for _, f := range l.Fabrics {
		rdma := "no RDMA"
		if f.RDMA {
			rdma = "RDMA"
		}
		if _, err := fmt.Fprintf(w, "  %s  %.2f GB/s, %s\n", f.Name, f.InterNodeBwGBps, rdma); err != nil {
			return err
		}
	}
	return nil
}

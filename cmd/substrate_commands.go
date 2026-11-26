package cmd

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/substrate/load"
)

// The commands the applications read substrate data through (design §5, accuracy-dashboard
// card; the brief's command table). Each prints JSON with --json and text without it, reads
// only what the pinned substrate publishes, and reports an error on standard error with a
// non-zero exit. None writes a file.

var substrateJSON bool

var catalogCmd = &cobra.Command{
	Use:   "catalog",
	Short: "Query the catalog this build reads",
}

var (
	resolveModel string
	resolveChip  string
)

var catalogResolveCmd = &cobra.Command{
	Use:   "resolve",
	Short: "Map a model or chip name to its catalog identity, or report that none exists",
	Long: "Resolve --model or --chip against the catalog in use: the one embedded in the substrate " +
		"release this build pins, or the directory --catalog or BLIS_CATALOG names. A model " +
		"resolves when the catalog has an entry of that name, or exactly one entry whose " +
		"model.yaml source repository is that name (compared without regard to letter case); a " +
		"chip resolves when hardware/<name>.yaml exists. The output is {\"query\": ..., " +
		"\"identity\": <catalog name> | null, \"reason\": ...}; a resolved chip adds " +
		"\"packaging\": {\"gpus_per_node\", \"gpus_per_rack\"}, each the value its entry fixes or null " +
		"where the scenario states it; a resolved model adds \"drafters\": [{\"name\", \"method\", " +
		"\"num_spec_tokens\"}], the latter two omitted where the catalog has no graph for the " +
		"drafter, what the entry lists as able to draft for it (\"speculator\" for its " +
		"own in-graph one, else a drafter checkpoint's catalog name; the --drafter of `blis " +
		"acceptance`), omitted for a model that lists none. A name two entries claim is an error, never a choice.",
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, _ []string) error {
		if (resolveModel == "") == (resolveChip == "") {
			return fmt.Errorf("catalog resolve: give exactly one of --model and --chip")
		}
		data, err := catalogData()
		if err != nil {
			return err
		}
		var r catalogResolution
		if resolveModel != "" {
			r, err = resolveCatalogModel(data, resolveModel)
		} else {
			r = resolveCatalogChip(data, resolveChip)
		}
		if err != nil {
			return err
		}
		if r.Identity != nil && resolveModel != "" {
			drafters, err := draftersOf(data, *r.Identity)
			if err != nil {
				return err
			}
			r.Drafters = drafters
		}
		if substrateJSON {
			return writeJSON(cmd.OutOrStdout(), r)
		}
		id := "none"
		if r.Identity != nil {
			id = *r.Identity
		}
		_, err = fmt.Fprintf(cmd.OutOrStdout(), "%s: %s (%s)\n", r.Query, id, r.Reason)
		return err
	},
}

// catalogResolution is catalog resolve's output.
type catalogResolution struct {
	Query    string  `json:"query"`
	Identity *string `json:"identity"`
	Reason   string  `json:"reason"`
	// Packaging is a resolved chip's packaging as its catalog entry states it: the GPUs per
	// node and per rack the product fixes, each null where the entry leaves it to the
	// scenario. Absent for a model, or a chip that does not resolve. A generator writes
	// neither field for a value stated here, and the scenario's own for a null one
	// (the substrate scenario.Cluster.Packaging, design SUB-3).
	Packaging *chipPackaging `json:"packaging,omitempty"`
	// Drafters is a resolved model's drafters as its entry lists them (model.Identity.Drafters,
	// design D8), in the entry's order: the in-graph speculator and each separate drafter
	// checkpoint, with the method and draft length the drafter's graph defaults to. Absent for a
	// chip, a model that does not resolve, and a model that lists none. `blis acceptance
	// --drafter` takes the name (inference-sim#2141).
	Drafters []drafterInfo `json:"drafters,omitempty"`
}

// catalogData opens the catalog a command names, with the embedded registry: a catalog
// command reads no coefficient, but the loader checks the two together.
func catalogData() (*load.Data, error) {
	o := dataOptions()
	o.Registry, o.RegistryOverlays = "", nil
	return openEngineData(o)
}

func resolveCatalogModel(data *load.Data, query string) (catalogResolution, error) {
	if _, err := data.Model(query); err == nil {
		return catalogResolution{Query: query, Identity: &query, Reason: "a catalog entry has this name"}, nil
	}
	var hits []string
	for _, name := range data.Models() {
		id, err := data.ModelIdentity(name)
		if err != nil {
			return catalogResolution{}, err
		}
		if id.Source != nil && strings.EqualFold(id.Source.Repo, query) {
			hits = append(hits, id.Name)
		}
	}
	sort.Strings(hits)
	switch len(hits) {
	case 0:
		return catalogResolution{Query: query,
			Reason: "no catalog entry has this name, and none has it as its source repository"}, nil
	case 1:
		return catalogResolution{Query: query, Identity: &hits[0],
			Reason: "the entry's model.yaml names this source repository"}, nil
	}
	return catalogResolution{}, fmt.Errorf("model %q is the source repository of %d catalog entries (%s); "+
		"name the entry instead", query, len(hits), strings.Join(hits, ", "))
}

func resolveCatalogChip(data *load.Data, query string) catalogResolution {
	if chip, err := data.Chip(query); err == nil {
		// The shared projection is the one `catalog list` reports the same chip through, so the
		// two commands cannot encode an unstated count differently.
		facts := chipFactsOf(chip)
		return catalogResolution{Query: query, Identity: &query, Reason: "the catalog has hardware/" + query + ".yaml",
			Packaging: &facts.Packaging}
	}
	return catalogResolution{Query: query, Reason: "the catalog has no hardware entry of this name"}
}

func writeJSON(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	return enc.Encode(v)
}

func init() {
	for _, c := range []*cobra.Command{catalogResolveCmd, catalogListCmd} {
		c.Flags().BoolVar(&substrateJSON, "json", false, "print JSON")
	}
	catalogResolveCmd.Flags().StringVar(&resolveModel, "model", "", "a model name: a catalog entry name or a source repository such as meta-llama/Llama-3.1-70B-Instruct")
	catalogResolveCmd.Flags().StringVar(&resolveChip, "chip", "", "a chip name, such as h200")
	registerCatalogFlag(catalogResolveCmd)
	registerCatalogFlag(catalogListCmd)
	catalogCmd.AddCommand(catalogResolveCmd)
	catalogCmd.AddCommand(catalogListCmd)
	rootCmd.AddCommand(catalogCmd)
}

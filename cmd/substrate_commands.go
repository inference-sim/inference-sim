package cmd

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
	"github.com/inference-sim/blis-substrate/load"
)

// The commands the applications read substrate data through (design §5, accuracy-dashboard
// card; the brief's command table). Each prints JSON with --json and text without it, reads
// only what the pinned substrate publishes, and reports an error on standard error with a
// non-zero exit. None writes a file.

var substrateJSON bool

var invariantsCmd = &cobra.Command{
	Use:   "invariants",
	Short: "Print the invariant registry of the blis-substrate version this build pins",
	Long: "Print every invariant of the registry embedded in blis-substrate's contracts/invariants, " +
		"at the version this build selects (stack.json names it). With --json the output is " +
		"{\"invariants\": [...]}, each record with the registry's own keys; without it, one line " +
		"per invariant: identifier, status, title.",
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, _ []string) error {
		return printInvariants(cmd.OutOrStdout(), substrateJSON)
	},
}

func printInvariants(w io.Writer, asJSON bool) error {
	reg, err := invariants.Embedded()
	if err != nil {
		return fmt.Errorf("the embedded invariant registry: %w", err)
	}
	if !asJSON {
		for _, inv := range reg.Invariants {
			title := inv.Title
			if title == "" {
				title = firstSentence(inv.Statement)
			}
			if _, err := fmt.Fprintf(w, "%-8s %-8s %s\n", inv.ID, inv.Status, title); err != nil {
				return err
			}
		}
		return nil
	}
	// The records carry yaml tags only; their JSON keys are the registry's own, so each record
	// is converted through its YAML form rather than given a second set of names here.
	records := make([]any, 0, len(reg.Invariants))
	for _, inv := range reg.Invariants {
		b, err := yaml.Marshal(inv)
		if err != nil {
			return err
		}
		var m map[string]any
		if err := yaml.Unmarshal(b, &m); err != nil {
			return err
		}
		records = append(records, m)
	}
	return writeJSON(w, map[string]any{"invariants": records})
}

func firstSentence(s string) string {
	if i := strings.Index(s, ". "); i >= 0 {
		return s[:i+1]
	}
	return s
}

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
	Long: "Resolve --model or --chip against the catalog in use: the one embedded in the blis-substrate " +
		"release this build pins, or the directory --catalog or BLIS_CATALOG names. A model " +
		"resolves when the catalog has an entry of that name, or exactly one entry whose " +
		"model.yaml source repository is that name (compared without regard to letter case); a " +
		"chip resolves when hardware/<name>.yaml exists. The output is {\"query\": ..., " +
		"\"identity\": <catalog name> | null, \"reason\": ...}. A name two entries claim is an " +
		"error, never a choice.",
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
	if _, err := data.Chip(query); err == nil {
		return catalogResolution{Query: query, Identity: &query, Reason: "the catalog has hardware/" + query + ".yaml"}
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
	for _, c := range []*cobra.Command{invariantsCmd, catalogResolveCmd} {
		c.Flags().BoolVar(&substrateJSON, "json", false, "print JSON")
	}
	catalogResolveCmd.Flags().StringVar(&resolveModel, "model", "", "a model name: a catalog entry name or a source repository such as meta-llama/Llama-3.1-70B-Instruct")
	catalogResolveCmd.Flags().StringVar(&resolveChip, "chip", "", "a chip name, such as h200")
	registerCatalogFlag(catalogResolveCmd)
	catalogCmd.AddCommand(catalogResolveCmd)
	rootCmd.AddCommand(invariantsCmd, catalogCmd)
}

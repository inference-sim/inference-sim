package cmd

import (
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"

	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"

	substrate "github.com/inference-sim/blis-substrate"
	"github.com/inference-sim/blis-substrate/ledger"
)

// `blis ledger` prints what the blis-substrate release this build pins publishes about its data
// ledger (design section 10.2, D7, COV-2). It reads only the catalog in use, the embedded copy
// or the directory --catalog names, and holds no list of its own.
//
//	blis ledger --list fit|anchor|report   one of the three published role lists
//	blis ledger units --catalog <checkout> the unit list, with each unit's identifier
//
// The published lists live in catalog/ledger/published/ and join the substrate's embedded data
// when a ledger owner has created the seed (data.go). A release without them has none to print,
// and the command says so and fails; it never draws a list itself. The unit lists are not
// embedded in any release (the loader ignores catalog/ledger/ except published/), so `units`
// reads a substrate checkout and refuses to run without one.

var ledgerList string

const publishedDir = "ledger/published"

var ledgerCmd = &cobra.Command{
	Use:   "ledger --list fit|anchor|report",
	Short: "Print a published role list of the pinned substrate release",
	Long: "Print the fit, anchor or report list the blis-substrate release this build pins publishes " +
		"(design section 10.2): the YAML file as the release carries it, after the substrate's own check " +
		"(ledger.LoadPublished). COV-2 scores only the anchor list. With --catalog or BLIS_CATALOG naming " +
		"a blis-substrate checkout's catalog, the lists are checked against ledger.yaml and the unit lists: " +
		"the seed and partition they were drawn under, all three lists present, every unit in a source, " +
		"no unit on two lists. The embedded copy carries neither ledger.yaml nor the unit lists, so it is " +
		"checked for what the lists say of themselves: kind, role, unit format and order, one seed and " +
		"partition across the three, one role per unit. The command never draws a list: a release that " +
		"has not published them has none to print, and the command fails saying so.",
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, _ []string) error {
		if ledgerList == "" {
			return fmt.Errorf("ledger: give --list fit, anchor or report, or the subcommand units")
		}
		dir := pathFrom("catalog", catalogPath, catalogEnvVar, os.Getenv(catalogEnvVar))
		return printPublishedList(cmd.OutOrStdout(), dir, substrate.Catalog(), ledgerList)
	},
}

var ledgerUnitsCmd = &cobra.Command{
	Use:   "units --catalog <substrate checkout>/catalog",
	Short: "Print the ledger's unit list with each unit's identifier",
	Long: "Print every unit of every source in catalog/ledger/sources.yaml, one line each: the " +
		"unit_id, a tab and the source. No release embeds the unit lists, so --catalog must name a " +
		"blis-substrate checkout's catalog directory; without one the command fails.",
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, _ []string) error {
		dir := pathFrom("catalog", catalogPath, catalogEnvVar, os.Getenv(catalogEnvVar))
		if dir == "" {
			return fmt.Errorf("ledger units: the unit lists are not embedded in any substrate release; " +
				"name a blis-substrate checkout's catalog directory with --catalog")
		}
		return printUnits(cmd.OutOrStdout(), filepath.Join(dir, "ledger"))
	},
}

// publishedRoles are the three roles a release publishes, in the order the substrate checks them.
var publishedRoles = []ledger.Role{ledger.RoleFit, ledger.RoleAnchor, ledger.RoleReport}

// printPublishedList writes the role's published list after the substrate's checks. dir is a
// substrate checkout's catalog directory ("" for the embedded copy, cat). The rules are
// ledger.LoadPublished's; this function only supplies what it reads.
func printPublishedList(w io.Writer, dir string, cat fs.FS, role string) error {
	if !ledger.Role(role).Published() {
		return fmt.Errorf("ledger: --list %q: the published lists are fit, anchor and report", role)
	}
	var (
		lists  map[ledger.Role]*ledger.Published
		listed string // the directory holding published/
		err    error
	)
	if dir != "" {
		listed = filepath.Join(dir, "ledger")
		lists, err = loadCheckoutLists(listed)
	} else {
		listed, lists, err = loadEmbeddedLists(cat)
		if listed != "" {
			defer func() { _ = os.RemoveAll(listed) }()
		}
	}
	if err != nil {
		return fmt.Errorf("ledger: %w", err)
	}
	if lists[ledger.Role(role)] == nil {
		return fmt.Errorf("ledger: the catalog in use has no %s list (%s): a release publishes the lists "+
			"once a ledger owner has created the seed (blis-substrate#31), and this one has not",
			role, path.Join(publishedDir, role+".yaml"))
	}
	raw, err := os.ReadFile(filepath.Join(listed, ledger.DirPublished, role+".yaml"))
	if err != nil {
		return fmt.Errorf("ledger: %w", err)
	}
	_, err = w.Write(raw)
	return err
}

// loadCheckoutLists reads a ledger directory of a substrate checkout the way the substrate's
// release job does: ledger.yaml, every source's unit list, then ledger.LoadPublished.
func loadCheckoutLists(ledgerDir string) (map[ledger.Role]*ledger.Published, error) {
	cfg, err := ledger.LoadConfig(ledgerDir)
	if err != nil {
		return nil, fmt.Errorf("%s is not a substrate ledger directory (the lists are checked against ledger.yaml): %w", ledgerDir, err)
	}
	_, units, err := sourceUnits(ledgerDir)
	if err != nil {
		return nil, err
	}
	return ledger.LoadPublished(ledgerDir, cfg, units)
}

// loadEmbeddedLists checks the lists an embedded catalog carries. LoadPublished reads a
// directory and needs the ledger's configuration and unit lists, neither of which a release
// embeds, so the lists are copied to a temporary directory and the configuration is the seed
// and partition the first list states: the lists must agree on them, which is all an embedded
// copy can be checked for. It returns the directory (the caller removes it).
func loadEmbeddedLists(cat fs.FS) (string, map[ledger.Role]*ledger.Published, error) {
	tmp, err := os.MkdirTemp("", "blis-ledger-")
	if err != nil {
		return "", nil, err
	}
	if err := os.MkdirAll(filepath.Join(tmp, ledger.DirPublished), 0o755); err != nil {
		return tmp, nil, err
	}
	cfg := &ledger.Config{}
	known := []ledger.Unit{}
	found := 0
	for _, r := range publishedRoles {
		raw, err := fs.ReadFile(cat, path.Join(publishedDir, string(r)+".yaml"))
		if err != nil {
			continue
		}
		found++
		if err := os.WriteFile(filepath.Join(tmp, ledger.DirPublished, string(r)+".yaml"), raw, 0o644); err != nil {
			return tmp, nil, err
		}
		var p ledger.Published
		if err := yaml.Unmarshal(raw, &p); err != nil {
			continue // LoadPublished reports the malformed file
		}
		if found == 1 {
			cfg.PartitionVersion, cfg.Seed.SHA256 = p.PartitionVersion, p.SeedSHA256
		}
		for _, id := range p.Units {
			known = append(known, ledger.Unit{UnitID: id})
		}
	}
	if found == 0 {
		return tmp, map[ledger.Role]*ledger.Published{}, nil
	}
	lists, err := ledger.LoadPublished(tmp, cfg, map[string][]ledger.Unit{"embedded": known})
	return tmp, lists, err
}

// sourceUnits reads sources.yaml and the unit list of every source that has one.
func sourceUnits(ledgerDir string) (*ledger.Sources, map[string][]ledger.Unit, error) {
	srcs, err := ledger.LoadSources(ledgerDir)
	if err != nil {
		return nil, nil, err
	}
	units := map[string][]ledger.Unit{}
	for _, s := range srcs.Sources {
		if s.Units == "" {
			continue
		}
		us, _, err := ledger.ReadUnitList(filepath.Join(ledgerDir, filepath.FromSlash(s.Units)))
		if err != nil {
			return nil, nil, fmt.Errorf("source %s: %w", s.ID, err)
		}
		units[s.ID] = us
	}
	return srcs, units, nil
}

// printUnits writes "<unit_id>\t<source>" for every unit of every source in dir's sources.yaml,
// sources in file order and units in unit_id order, read with the ledger package's own reader.
func printUnits(w io.Writer, dir string) error {
	srcs, units, err := sourceUnits(dir)
	if err != nil {
		return fmt.Errorf("ledger units: %w", err)
	}
	n := 0
	for _, s := range srcs.Sources {
		ids := make([]string, len(units[s.ID]))
		for i, u := range units[s.ID] {
			ids[i] = u.UnitID
		}
		sort.Strings(ids)
		for _, id := range ids {
			if _, err := fmt.Fprintf(w, "%s\t%s\n", id, s.ID); err != nil {
				return err
			}
			n++
		}
	}
	if n == 0 {
		return fmt.Errorf("ledger units: %s lists no unit", filepath.Join(dir, "sources.yaml"))
	}
	return nil
}

func init() {
	ledgerCmd.Flags().StringVar(&ledgerList, "list", "", "The published role list to print: fit, anchor or report")
	ledgerCmd.PersistentFlags().StringVar(&catalogPath, "catalog", "", "A directory with the layout of blis-substrate's catalog/ that replaces the embedded catalog (or "+catalogEnvVar+"; the flag wins). Without either the command reads the catalog embedded in the blis-substrate release this build pins")
	ledgerCmd.AddCommand(ledgerUnitsCmd)
	rootCmd.AddCommand(ledgerCmd)
}

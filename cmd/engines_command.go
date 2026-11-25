package cmd

import (
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
)

// engineReport is one engine kind as this build runs it. It is the record `blis engines` prints.
type engineReport struct {
	Kind string `json:"kind"`
	// ModeledVersion is the upstream release whose serving behaviour this build prices.
	ModeledVersion string `json:"modeled_version"`
	// ModeledCommit is the full commit ModeledVersion names.
	ModeledCommit string `json:"modeled_commit"`
	// BuildVersion is the upstream tree the engine's rules were ported from. Provenance only:
	// it is not the release a price is reported against.
	BuildVersion string `json:"build_version"`
	// BuildCommit is the full commit BuildVersion names.
	BuildCommit string `json:"build_commit"`
	// Roles are the deployment roles this build can run the kind in: a colocated pool, or one
	// side of a prefill/decode deployment.
	Roles []deployment.Role `json:"roles"`
}

// enginesCmd reports, for every engine kind this build runs, the release whose serving behaviour
// it models, the upstream tree its rules were ported from, and the deployment roles it can run
// (design LIF-6, X23). The build fixes the release and a scenario states none, so a caller that
// needs to know which release a price belongs to reads it here and keeps no table of its own.
//
// The two releases are different facts and are reported separately:
//
//   - modeled_version is the release a user can name and match benchmarked figures against. A
//     vLLM Result's kernel assumptions cite it.
//   - build_version is the upstream tree the engine's rules were generated from. It is
//     provenance: it may be a commit with no release tag, and it is never the release a price is
//     reported against.
//
// The command holds no version and no role of its own: each engine's build site states them
// (vllmEngineModel in cmd/engine_build.go, sglangEngineModel in cmd/sglang_engine.go), and this
// file prints what they return.
//
// With --json the output is {"engines": [...]}, one record per kind, sorted by kind. Without it,
// one line per kind.
var enginesCmd = &cobra.Command{
	Use:   "engines",
	Short: "Print the release each engine kind models and the roles it can run",
	Long: "Print every engine kind this build runs: the upstream release whose serving behaviour it " +
		"models, the commit of that release, the upstream tree its rules were ported from, and the " +
		"deployment roles it can run (colocated, prefill, decode). The build fixes the release — a " +
		"scenario states none — so this is where a caller reads which release a price belongs to. " +
		"With --json the output is {\"engines\": [...]}, one record per kind, sorted by kind.",
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, _ []string) error {
		return printEngines(cmd.OutOrStdout(), engineModels(), enginesJSON)
	},
}

// engineModels is every engine kind this build runs, sorted by kind. Each engine's build site
// supplies its own record, so this function names no engine.
func engineModels() []engineReport {
	models := []engineReport{sglangEngineModel(), vllmEngineModel()}
	sort.Slice(models, func(i, j int) bool { return models[i].Kind < models[j].Kind })
	return models
}

// printEngines writes models, as JSON or as one line each.
func printEngines(w io.Writer, models []engineReport, asJSON bool) error {
	if asJSON {
		return writeJSON(w, map[string]any{"engines": models})
	}
	for _, m := range models {
		roles := make([]string, len(m.Roles))
		for i, r := range m.Roles {
			roles[i] = string(r)
		}
		if _, err := fmt.Fprintf(w, "%s: models %s (%s); rules from %s (%s); roles %s\n",
			m.Kind, m.ModeledVersion, m.ModeledCommit, m.BuildVersion, m.BuildCommit,
			strings.Join(roles, ",")); err != nil {
			return err
		}
	}
	return nil
}

var enginesJSON bool

func init() {
	enginesCmd.Flags().BoolVar(&enginesJSON, "json", false, "print JSON")
	rootCmd.AddCommand(enginesCmd)
}

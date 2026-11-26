package cmd

import (
	"fmt"
	"io"
	"runtime/debug"
	"sort"
	"strings"

	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/internal/stack"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"
)

// bundledStack is the repository's stack.json compiled into the binary (main.go embeds it):
// the upstream commits this build runs (LIF-6).
var bundledStack []byte

// SetBundledStack installs the compiled-in stack.json. Called once by main.
func SetBundledStack(b []byte) { bundledStack = b }

// buildStack is the compiled-in stack.json, or nil when the build has none (a test binary).
func buildStack() *stack.Stack {
	if bundledStack == nil {
		return nil
	}
	s, err := stack.Parse(bundledStack)
	if err != nil {
		return nil
	}
	return s
}

// llmdCommit is the llm-d commit the router fork is based on, from the compiled-in stack.json,
// or "" for a build that has none.
func llmdCommit() string {
	if s := buildStack(); s != nil {
		return s.Router.Commit
	}
	return ""
}

// engineReport is one engine kind as this build runs it: the release whose serving behaviour it
// models, the upstream tree its rules were ported from, and the deployment roles it can run.
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

// versionDocument is `blis version --json`'s document. It is the build's identity plus the
// engines it runs: the same facts the text form prints, addressed by name rather than by line.
type versionDocument struct {
	InferenceSim string `json:"inference_sim"`
	Substrate    string `json:"substrate"`
	// Engines is one record per engine kind this build runs, sorted by kind.
	Engines []engineReport `json:"engines"`
	// LLMD is the llm-d release the router fork is based on, and that fork's commit. Absent for
	// a build with no compiled-in stack, which reports the simulator and substrate alone.
	LLMD *versionLLMD `json:"llm_d,omitempty"`
}

// versionLLMD is the llm-d release a build's router fork is based on.
type versionLLMD struct {
	Version    string `json:"version"`
	Commit     string `json:"commit"`
	RouterFork string `json:"router_fork"`
}

var versionJSON bool

var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print the commits this build runs and the release each engine models",
	Long: "Print the simulator build, the substrate, the llm-d commit the router is based on, and, " +
		"for every engine kind, the upstream release whose serving behaviour this build prices " +
		"beside the upstream tree its rules were ported from and the deployment roles it can run. " +
		"A scenario states none of them: the build fixes them (stack.json) and every Result records " +
		"them (LIF-6). The two releases are different facts: the modelled release is the one a " +
		"benchmarked figure is comparable with and the one a Result's kernel assumptions cite; the " +
		"rules tree is provenance and may be a commit with no release tag. A user who needs another " +
		"upstream commit uses the inference-sim release that runs it. With --json the output is " +
		"{\"inference_sim\", \"substrate\", \"engines\": [...], \"llm_d\"}, with one engine record " +
		"per kind, sorted by kind.",
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, _ []string) error {
		info, _ := debug.ReadBuildInfo()
		rev := revisionsFrom(info)
		engines := engineModels()
		if versionJSON {
			return writeJSON(cmd.OutOrStdout(), versionDocumentOf(rev, engines))
		}
		return printVersion(cmd.OutOrStdout(), rev, engines)
	},
}

// versionDocumentOf is the build's identity and engines as `version --json` reports them.
func versionDocumentOf(rev result.Revisions, engines []engineReport) versionDocument {
	out := versionDocument{
		InferenceSim: rev.Simulator,
		Substrate:    rev.Substrate,
		Engines:      engines,
	}
	router := rev.Router
	if router == "" {
		router = "unknown"
	}
	if s := buildStack(); s != nil {
		out.LLMD = &versionLLMD{Version: s.Router.Version, Commit: s.Router.Commit, RouterFork: router}
	}
	return out
}

// printVersion writes the text form. Each engine line keeps the `engine <kind> <release> <commit>`
// shape and appends the rules tree and roles in parentheses, so a caller that reads the release
// positionally still finds it there.
func printVersion(w io.Writer, rev result.Revisions, engines []engineReport) error {
	var b strings.Builder
	fmt.Fprintf(&b, "inference-sim %s\n", rev.Simulator)
	fmt.Fprintf(&b, "substrate %s\n", rev.Substrate)
	for _, e := range engines {
		roles := make([]string, len(e.Roles))
		for i, r := range e.Roles {
			roles[i] = string(r)
		}
		fmt.Fprintf(&b, "engine %s %s %s (rules from %s %s; roles %s)\n",
			e.Kind, e.ModeledVersion, e.ModeledCommit, e.BuildVersion, e.BuildCommit,
			strings.Join(roles, ","))
	}
	router := rev.Router
	if router == "" {
		router = "unknown"
	}
	if s := buildStack(); s != nil {
		fmt.Fprintf(&b, "llm-d %s %s (router fork %s)\n", s.Router.Version, s.Router.Commit, router)
	} else {
		fmt.Fprintf(&b, "llm-d unknown (router fork %s)\n", router)
	}
	_, err := io.WriteString(w, b.String())
	return err
}

// engineModels is every engine kind this build runs, sorted by kind. Each engine's build site
// supplies its own record (vllmEngineModel in cmd/engine_build.go, sglangEngineModel in
// cmd/sglang_engine.go), so this function names no engine.
func engineModels() []engineReport {
	models := []engineReport{sglangEngineModel(), vllmEngineModel()}
	sort.Slice(models, func(i, j int) bool { return models[i].Kind < models[j].Kind })
	return models
}

func init() {
	versionCmd.Flags().BoolVar(&versionJSON, "json", false, "print JSON")
	rootCmd.AddCommand(versionCmd)
}

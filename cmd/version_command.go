package cmd

import (
	"fmt"
	"io"
	"runtime/debug"
	"sort"
	"strings"

	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/internal/stack"
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

var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print the commits this build runs",
	Long: "Print the simulator build, the substrate, the upstream commit of each engine this build " +
		"runs, and the llm-d commit the router is based on. A scenario states none of them: the " +
		"build fixes them (stack.json) and every Result records them (LIF-6). A user who needs " +
		"another upstream commit uses the inference-sim release that runs it.",
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, _ []string) error {
		info, _ := debug.ReadBuildInfo()
		rev := revisionsFrom(info)
		var w strings.Builder
		fmt.Fprintf(&w, "inference-sim %s\n", rev.Simulator)
		fmt.Fprintf(&w, "substrate %s\n", rev.Substrate)
		engines := EngineReleases()
		sort.Slice(engines, func(i, j int) bool { return engines[i].Kind < engines[j].Kind })
		for _, e := range engines {
			fmt.Fprintf(&w, "engine %s %s %s\n", e.Kind, e.Version, e.Commit)
		}
		router := rev.Router
		if router == "" {
			router = "unknown"
		}
		if s := buildStack(); s != nil {
			fmt.Fprintf(&w, "llm-d %s %s (router fork %s)\n", s.Router.Version, s.Router.Commit, router)
		} else {
			fmt.Fprintf(&w, "llm-d unknown (router fork %s)\n", router)
		}
		_, err := io.WriteString(cmd.OutOrStdout(), w.String())
		return err
	},
}

func init() { rootCmd.AddCommand(versionCmd) }

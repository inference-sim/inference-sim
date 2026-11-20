// Idiomatic entrypoint for Cobra CLI that deletes handling to the Cobra root command in cmd/root.go

package main

import (
	_ "embed"
	"os"
	"runtime/debug"

	"github.com/inference-sim/inference-sim/cmd"
)

// defaultsYAML is the repository's defaults.yaml, compiled in so a run outside a checkout
// has the shipped defaults (cmd.SetBundledDefaults).
//
//go:embed defaults.yaml
var defaultsYAML []byte

// stackJSON is the repository's stack.json, compiled in: the upstream commits this build runs
// (LIF-6), which `blis version` prints and every Result records (cmd.SetBundledStack).
//
//go:embed stack.json
var stackJSON []byte

func main() {
	cmd.SetBundledDefaults(defaultsYAML)
	cmd.SetBundledStack(stackJSON)
	if os.Getenv("GOGC") == "" {
		debug.SetGCPercent(400) // the run is allocation-heavy and short; GC was 10-20% of CPU
	}
	cmd.Execute()
}

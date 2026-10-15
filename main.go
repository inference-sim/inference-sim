// Idiomatic entrypoint for Cobra CLI that deletes handling to the Cobra root command in cmd/root.go

package main

import (
	_ "embed"

	"github.com/inference-sim/inference-sim/cmd"
)

// defaultsYAML is the repository's defaults.yaml, compiled in so a run outside a checkout
// has the shipped defaults (cmd.SetBundledDefaults).
//
//go:embed defaults.yaml
var defaultsYAML []byte

func main() {
	cmd.SetBundledDefaults(defaultsYAML)
	cmd.Execute()
}

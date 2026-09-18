// Idiomatic entrypoint for Cobra CLI that deletes handling to the Cobra root command in cmd/root.go

package main

import (
	"github.com/inference-sim/inference-sim/cmd"
)

func main() {
	cmd.Execute()
}

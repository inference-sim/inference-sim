// Idiomatic entrypoint for Cobra CLI that deletes handling to the Cobra root command in cmd/root.go

package main

import (
	"blis/cmd"
)

func main() {
	cmd.Execute()
}

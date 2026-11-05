// Command stackgen writes stack.json from the build it runs in (design SIM-3). Run it from the
// repository root after changing go.mod or guides.lock:
//
//	go run ./internal/stack/cmd/stackgen
//
// With -check it writes nothing and exits 1 when the committed file differs.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/inference-sim/inference-sim/cmd"
	"github.com/inference-sim/inference-sim/internal/stack"
)

func main() {
	root := flag.String("root", ".", "the inference-sim repository root")
	check := flag.Bool("check", false, "compare with the committed stack.json instead of writing it")
	flag.Parse()
	if err := run(*root, *check); err != nil {
		fmt.Fprintln(os.Stderr, "stackgen:", err)
		os.Exit(1)
	}
}

func run(root string, check bool) error {
	s, err := stack.Build(stack.Inputs{Root: root, Engines: cmd.EngineReleases()})
	if err != nil {
		return err
	}
	want, err := s.Marshal()
	if err != nil {
		return err
	}
	path := filepath.Join(root, "stack.json")
	if !check {
		return os.WriteFile(path, want, 0o644)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if !bytes.Equal(got, want) {
		return fmt.Errorf("%s is stale; run go run ./internal/stack/cmd/stackgen", path)
	}
	return nil
}

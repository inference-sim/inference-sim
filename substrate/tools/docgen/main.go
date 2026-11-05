// Command docgen writes the parts of the documentation that are facts about the code: the
// members of every closed vocabulary, and each registered rules pack's rules and accepted
// values. Prose is written by hand; a list that the code already states is generated, so
// the two cannot disagree.
//
// Run it from the repository root after changing a vocabulary or a rules pack:
//
//	go run ./tools/docgen
//
// TestGeneratedDocsAreCurrent fails when a committed file differs from what this command
// would write, so a stale page is caught by `go test ./...` rather than by a reader.
package main

import (
	"fmt"
	"os"
	"path/filepath"
)

func main() {
	files, err := generate(".")
	if err != nil {
		fmt.Fprintln(os.Stderr, "docgen:", err)
		os.Exit(1)
	}
	// Start from an empty directory, so a file docgen no longer writes does not linger.
	if err := os.RemoveAll(outDir); err != nil {
		fmt.Fprintln(os.Stderr, "docgen:", err)
		os.Exit(1)
	}
	for _, f := range files {
		path := filepath.Join(".", f.path)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			fmt.Fprintln(os.Stderr, "docgen:", err)
			os.Exit(1)
		}
		if err := os.WriteFile(path, f.content, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "docgen:", err)
			os.Exit(1)
		}
	}
	fmt.Printf("docgen: wrote %d file(s) under %s\n", len(files), outDir)
}

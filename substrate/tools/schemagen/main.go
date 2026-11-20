// Command schemagen writes the JSON Schema of each BLIS document (SUB-19) from its Go type
// into contracts/schema, which the substrate release publishes and embeds.
//
// Run it from the repository root after changing a document type:
//
//	go run ./tools/schemagen
//
// TestPublishedSchemasMatchTheTypes fails when a committed schema differs from what this
// command would write.
package main

import (
	"fmt"
	"os"
	"path/filepath"
)

type file struct {
	path    string
	content []byte
}

func main() {
	files, err := generate(".")
	if err != nil {
		fmt.Fprintln(os.Stderr, "schemagen:", err)
		os.Exit(1)
	}
	for _, f := range files {
		if err := os.WriteFile(filepath.FromSlash(f.path), f.content, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "schemagen:", err)
			os.Exit(1)
		}
	}
	fmt.Printf("schemagen: wrote %d schema(s) under %s\n", len(files), outDir)
}

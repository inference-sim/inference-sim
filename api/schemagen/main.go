// Command schemagen writes the committed JSON Schema of the BLIS declarative envelope,
// derived from the Go types in package api.
//
// It is invoked by the go:generate directive in api/doc.go, so the canonical way to run it
// is from the repository root:
//
//	go generate ./api/...
//
// api.TestCommittedSchemaIsCurrent compares the committed file against a fresh
// api.GenerateSchema, so CI fails if the envelope types changed without a regeneration.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/inference-sim/inference-sim/api"
)

func main() {
	out := flag.String("out", api.SchemaPath,
		"path to write the schema to, relative to the working directory (go generate runs in api/)")
	flag.Parse()

	if err := run(*out); err != nil {
		// A generator is a command, not library code: a failure aborts with a diagnostic on
		// stderr and a non-zero status (engineering principles: CLI errors terminate).
		fmt.Fprintf(os.Stderr, "schemagen: %v\n", err)
		os.Exit(1)
	}
}

func run(out string) error {
	schema, err := api.GenerateSchema()
	if err != nil {
		return err
	}
	if dir := filepath.Dir(out); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("creating %s: %w", dir, err)
		}
	}
	if err := os.WriteFile(out, schema, 0o644); err != nil {
		return fmt.Errorf("writing %s: %w", out, err)
	}
	return nil
}

// Package schema publishes the JSON Schema of every document BLIS reads and writes across
// a repository boundary (SUB-19): scenario, deployment, result, point-record, workload
// and trace. Each is generated from its Go type in contracts/spec by Generate, committed
// under this directory, and embedded, so `blis schema <document>` prints the schema the
// build's substrate release carries.
//
// A published schema serves editors and structural checks: types, required keys, closed
// vocabularies and the refusal of unknown keys. The Go validators remain the definition;
// `blis schema validate` runs them, and a document the schema accepts can still be
// refused by a validator rule a schema cannot state.
//
// Regenerate after changing a document type:
//
//	go run ./tools/schemagen
package schema

import (
	"embed"
	"fmt"
	"sort"
)

//go:embed *.schema.json
var published embed.FS

// Documents names each published document and its file.
var Documents = map[string]string{
	"scenario":     "scenario.schema.json",
	"deployment":   "deployment.schema.json",
	"result":       "result.schema.json",
	"point-record": "point-record.schema.json",
	"workload":     "workload.schema.json",
	"trace":        "trace.schema.json",
}

// Names returns the document names, sorted.
func Names() []string {
	out := make([]string, 0, len(Documents))
	for n := range Documents {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// Published returns the committed JSON Schema of a document, as this release embeds it.
func Published(document string) ([]byte, error) {
	f, ok := Documents[document]
	if !ok {
		return nil, fmt.Errorf("no schema for document %q; documents are %v", document, Names())
	}
	return published.ReadFile(f)
}

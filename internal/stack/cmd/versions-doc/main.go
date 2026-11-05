// Command versions-doc renders the generated version blocks in the documents that state a
// version (design X23) from the committed stack.json. Run it from the repository root after
// stackgen:
//
//	go run ./internal/stack/cmd/versions-doc
//
// With -check it writes nothing and exits 1 when a document is stale.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/inference-sim/inference-sim/internal/stack"
)

func main() {
	root := flag.String("root", ".", "the inference-sim repository root")
	check := flag.Bool("check", false, "report stale documents instead of rewriting them")
	flag.Parse()
	s, err := stack.Read(filepath.Join(*root, "stack.json"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "versions-doc:", err)
		os.Exit(1)
	}
	stale, err := stack.RenderDocs(*root, s, !*check)
	if err != nil {
		fmt.Fprintln(os.Stderr, "versions-doc:", err)
		os.Exit(1)
	}
	if *check && len(stale) > 0 {
		fmt.Fprintf(os.Stderr, "versions-doc: stale, run go run ./internal/stack/cmd/versions-doc:\n  %s\n", strings.Join(stale, "\n  "))
		os.Exit(1)
	}
}

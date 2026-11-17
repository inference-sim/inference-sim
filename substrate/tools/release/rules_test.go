package main

import (
	"os"
	"testing"
)

// The scan must not find the markers it looks for in its own definition (blis-substrate#84),
// and must still find a real one of every kind.
func TestTheDeprecationScanDoesNotMatchItsOwnDefinition(t *testing.T) {
	for _, f := range []string{"rules.go", "release.go"} {
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if got := FindDeprecations(f, src); len(got) != 0 {
			t.Errorf("%s: the scan reports its own text as deprecations: %v", f, got)
		}
	}
	for name, line := range map[string]string{
		"comment":     "// Deprecated: use x.",
		"cobra field": "\tDeprecated: \"use y\",",
		"pflag":       "\tfs.MarkDeprecated(\"old\", \"use new\")",
		"flag usage":  "\t\"the old flag (DEPRECATED: use --new)\"",
	} {
		if got := FindDeprecations("a.go", []byte(line)); len(got) != 1 {
			t.Errorf("%s: a real marker %q gave %d findings, want 1", name, line, len(got))
		}
	}
}

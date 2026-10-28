// Moved from blis-latency-kernel/examples/docs_examples_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	"path/filepath"

	"os"
	"regexp"
	"strings"
	"testing"
)

// The tests that tie the documentation site to the code. The Go examples themselves are run by
// example_test.go and the generated files are checked by internal/docgen; these cover what the
// pages say beside them. docs/contributing/documentation.md describes all of it.

// exampleOutputs maps each snippet section of example_test.go to the // Output: of the example
// that contains it.
func exampleOutputs(t *testing.T) map[string]string {
	t.Helper()
	src, err := os.ReadFile("example_test.go")
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]string{}
	section := regexp.MustCompile(`--8<-- \[start:([a-z]+)\]`)
	funcs := strings.Split(string(src), "\nfunc ")
	for _, f := range funcs {
		i := strings.Index(f, "\t// Output:\n")
		if i < 0 {
			continue
		}
		var lines []string
		for _, line := range strings.Split(f[i+len("\t// Output:\n"):], "\n") {
			if !strings.HasPrefix(line, "\t//") {
				break
			}
			lines = append(lines, strings.TrimPrefix(strings.TrimPrefix(line, "\t//"), " "))
		}
		for _, m := range section.FindAllStringSubmatch(f, -1) {
			out[m[1]] = strings.Join(lines, "\n")
		}
	}
	return out
}

// TestDocOutputsMatchTheirExamples checks that the output a page shows beside a Go snippet is
// the output of the example the snippet comes from. Go checks the example against its own
// // Output: comment; this checks the copy on the page against that, so the page cannot show a
// number the code no longer computes.
func TestDocOutputsMatchTheirExamples(t *testing.T) {
	outputs := exampleOutputs(t)
	// A page names its example in a comment above the code block, and shows the output below it.
	snippet := regexp.MustCompile("(?s)<!-- example_test.go:([a-z]+): [^>]*-->\n```go\n.*?\n```\n\n```text\n(.*?)\n```")
	seen := 0
	for path, page := range docPages(t) {
		for _, m := range snippet.FindAllStringSubmatch(page, -1) {
			seen++
			want, ok := outputs[m[1]]
			if !ok {
				t.Errorf("%s shows output for snippet %q, which is in no example with an "+
					"// Output: comment", path, m[1])
				continue
			}
			if m[2] != want {
				t.Errorf("%s shows this output for snippet %q:\n%s\nbut the example prints:\n%s",
					path, m[1], m[2], want)
			}
		}
	}
	if seen == 0 {
		t.Error("no page shows an example's output; the pattern this test matches is stale")
	}
}

// docPages returns every Markdown page under docs/, keyed by path, excluding generated files.
func docPages(t *testing.T) map[string]string {
	t.Helper()
	pages := map[string]string{}
	err := filepath.WalkDir(filepath.Join("..", "docs"), func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && path == filepath.Join("..", "docs", "generated") {
			return filepath.SkipDir
		}
		if !d.IsDir() && strings.HasSuffix(path, ".md") {
			b, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			pages[path] = string(b)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(pages) == 0 {
		t.Fatal("no pages under docs/")
	}
	return pages
}

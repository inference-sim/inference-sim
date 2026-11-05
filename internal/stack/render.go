package stack

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Rendered documents (X23). A document states a version only inside a generated block:
//
//	<!-- BEGIN GENERATED from stack.json: <kind> -->
//	...
//	<!-- END GENERATED -->
//
// The text between the markers is Block(kind) of the committed stack.json, and Render rewrites
// it. The documents carrying blocks are listed in RenderedDocs.

// RenderedDocs are the documents, relative to the repository root, that carry generated
// blocks. A test checks that each has at least one, so a block cannot be deleted unnoticed.
var RenderedDocs = []string{
	"CLAUDE.md",
	"README.md",
	"docs/index.md",
	"docs/getting-started/installation.md",
	"docs/getting-started/quickstart.md",
	"docs/getting-started/tutorial.md",
	"docs/guide/latency-models.md",
	"docs/reference/configuration.md",
}

var blockRE = regexp.MustCompile(`(?s)<!-- BEGIN GENERATED from stack\.json: ([a-z-]+) -->\n(.*?)<!-- END GENERATED -->`)

const endMarker = "<!-- END GENERATED -->"

// substrateRef is what a checkout names: the tag, or the full commit of a pseudo-version.
func (s *Stack) substrateRef() (string, error) {
	m, ok := s.Module(SubstrateModule)
	if !ok {
		return "", fmt.Errorf("stack.json names no %s", SubstrateModule)
	}
	if m.Commit != "" {
		return m.Commit, nil
	}
	return m.Version, nil
}

// Block is the generated text of one block kind, ending in a newline.
func (s *Stack) Block(kind string) (string, error) {
	ref, err := s.substrateRef()
	if err != nil {
		return "", err
	}
	sub, _ := s.Module(SubstrateModule)
	switch kind {
	case "data-checkout":
		return "```bash\n" +
			"git clone https://github.com/inference-sim/blis-substrate.git\n" +
			"git -C blis-substrate checkout " + ref + "\n" +
			"export BLIS_CATALOG=$PWD/blis-substrate/catalog   # or pass --catalog on every command\n" +
			"```\n", nil
	case "toolchain":
		return "Requires Go " + strings.TrimPrefix(s.Toolchain, "go") +
			", the toolchain blis-substrate's contracts/toolchain names at the pinned release.\n", nil
	case "versions":
		var b strings.Builder
		b.WriteString("| Component | Version | Commit |\n|---|---|---|\n")
		mods := append([]Module(nil), s.Modules...)
		sort.Slice(mods, func(i, j int) bool { return mods[i].Path < mods[j].Path })
		for _, m := range mods {
			fmt.Fprintf(&b, "| `%s` | `%s` | %s |\n", m.Path, m.Version, short(m.Commit))
		}
		for _, e := range s.Engines {
			fmt.Fprintf(&b, "| %s (engine) | `%s` | %s |\n", e.Kind, e.Version, short(e.Commit))
		}
		fmt.Fprintf(&b, "| %s (router) | `%s` | %s |\n", s.Router.Kind, s.Router.Version, short(s.Router.Commit))
		fmt.Fprintf(&b, "| llm-d guides | | %s |\n", short(s.GuidesCommit))
		fmt.Fprintf(&b, "| Go toolchain | `%s` | |\n", s.Toolchain)
		_ = sub
		return b.String(), nil
	}
	return "", fmt.Errorf("unknown generated block %q", kind)
}

func short(c string) string {
	if len(c) > 12 {
		return "`" + c[:12] + "`"
	}
	if c == "" {
		return ""
	}
	return "`" + c + "`"
}

// Render returns src with every generated block rewritten from s, and the number of blocks.
func (s *Stack) Render(src []byte) ([]byte, int, error) {
	var err error
	n := 0
	out := blockRE.ReplaceAllFunc(src, func(m []byte) []byte {
		n++
		kind := string(blockRE.FindSubmatch(m)[1])
		body, e := s.Block(kind)
		if e != nil {
			err = e
			return m
		}
		return []byte("<!-- BEGIN GENERATED from stack.json: " + kind + " -->\n" + body + endMarker)
	})
	if err != nil {
		return nil, 0, err
	}
	if c := bytes.Count(src, []byte("<!-- BEGIN GENERATED from stack.json:")); c != n {
		return nil, 0, fmt.Errorf("%d block openings, %d well-formed blocks", c, n)
	}
	return out, n, nil
}

// RenderDocs renders every document in RenderedDocs under root. With write false it changes
// nothing and returns the documents that differ from their rendering.
func RenderDocs(root string, s *Stack, write bool) ([]string, error) {
	var stale []string
	for _, rel := range RenderedDocs {
		path := filepath.Join(root, rel)
		src, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		out, n, err := s.Render(src)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", rel, err)
		}
		if n == 0 {
			return nil, fmt.Errorf("%s carries no generated block", rel)
		}
		if bytes.Equal(out, src) {
			continue
		}
		stale = append(stale, rel)
		if write {
			if err := os.WriteFile(path, out, 0o644); err != nil {
				return nil, err
			}
		}
	}
	return stale, nil
}

// OutsideBlocks returns src with every generated block removed, for the check that no version
// is stated outside one.
func OutsideBlocks(src []byte) []byte { return blockRE.ReplaceAll(src, nil) }

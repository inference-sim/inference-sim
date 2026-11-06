package main

import (
	"fmt"
	"regexp"
	"strings"
)

// The documentation reference check (X26). In a Markdown or HTML document, every source
// path, every Go symbol and every issue or pull-request reference is checked against the
// tree it is in:
//
//   - a path in backticks, or a relative link target, that names a file or directory of
//     the repository must exist in the tree (an absolute URL is not checked);
//   - a reference path:line or path:a-b must name lines the file has;
//   - a Go symbol written `pkg.Name` where pkg is a directory of this repository must be
//     declared in that package (checked by the caller, which has the parser);
//   - a table row may not appear twice in one table (blis-schemas #96).
//
// A path under a directory the docs build generates (generatedDocs) is not checked, and
// neither is a link holding a template placeholder ("[[").
//
// Statements that a feature is supported are generated from surface.json or stack.json
// in the repositories that publish them; they are outside this repository's check.

var (
	codeSpan   = regexp.MustCompile("`([^`\\s]+)`")
	mdLink     = regexp.MustCompile(`\]\(([^)\s#]+)(?:#[^)]*)?\)`)
	lineRef    = regexp.MustCompile(`^(.+?):(\d+)(?:-(\d+))?$`)
	pathish    = regexp.MustCompile(`^[\w.-]+(?:/[\w.*-]+)+/?$|^[\w-]+\.(?:go|py|sh|ya?ml|json|md|txt|toml)$`)
	tableRowRe = regexp.MustCompile(`^\s*\|.*\|\s*$`)
)

// Tree is what the reference check asks of the repository at the pinned tree.
type Tree interface {
	// Lines returns the number of lines of a file, and whether the path exists (a
	// directory exists with zero lines).
	Lines(path string) (int, bool)
}

// DocFindings returns the X26 findings for one document at path doc.
func DocFindings(doc string, src []byte, tree Tree, roots map[string]bool) []string {
	var bad []string
	dir := ""
	if i := strings.LastIndex(doc, "/"); i >= 0 {
		dir = doc[:i+1]
	}
	lines := strings.Split(string(src), "\n")
	inFence := false
	var table map[string]int
	for i, line := range lines {
		ln := i + 1
		if strings.HasPrefix(strings.TrimSpace(line), "```") {
			inFence = !inFence
			continue
		}
		if inFence {
			continue
		}
		if tableRowRe.MatchString(line) && !strings.Contains(line, "---") {
			if table == nil {
				table = map[string]int{}
			}
			key := strings.Join(strings.Fields(line), " ")
			if prev, dup := table[key]; dup {
				bad = append(bad, fmt.Sprintf("%s:%d: the table row at line %d is repeated (X26)", doc, ln, prev))
			} else {
				table[key] = ln
			}
		} else {
			table = nil
		}
		for _, m := range codeSpan.FindAllStringSubmatch(line, -1) {
			ref := m[1]
			if f := checkRef(ref, "", tree, roots); f != "" {
				bad = append(bad, fmt.Sprintf("%s:%d: %s (X26)", doc, ln, f))
			}
		}
		for _, m := range mdLink.FindAllStringSubmatch(line, -1) {
			ref := m[1]
			if strings.Contains(ref, "://") || strings.HasPrefix(ref, "mailto:") {
				continue
			}
			if f := checkRef(ref, dir, tree, nil); f != "" {
				bad = append(bad, fmt.Sprintf("%s:%d: %s (X26)", doc, ln, f))
			}
		}
	}
	return bad
}

// checkRef checks one reference. A code span is checked only when its first segment is a
// top-level directory of the repository (roots), so `config.json` in prose is not taken
// for a path; a link is resolved against the document's directory.
// generatedDocs are the directories the docs build writes, which are not in the tree:
// docs/hooks/catalog_pages.py writes one page per model under catalog/reference/models.
var generatedDocs = []string{"docs/catalog/reference/models/", "catalog/reference/models/"}

func checkRef(ref, dir string, tree Tree, roots map[string]bool) string {
	if strings.Contains(ref, "[[") {
		return ""
	}
	path, from, to := ref, 0, 0
	if m := lineRef.FindStringSubmatch(ref); m != nil {
		path = m[1]
		fmt.Sscan(m[2], &from)
		to = from
		if m[3] != "" {
			fmt.Sscan(m[3], &to)
		}
	}
	if roots != nil {
		first := strings.SplitN(path, "/", 2)[0]
		if !roots[first] || !strings.Contains(path, "/") || !pathish.MatchString(path) || strings.ContainsAny(path, "*<>{}") {
			return ""
		}
	} else {
		path = clean(dir + path)
	}
	for _, g := range generatedDocs {
		if strings.HasPrefix(path+"/", g) {
			return ""
		}
	}
	n, ok := tree.Lines(strings.TrimSuffix(path, "/"))
	if !ok {
		return fmt.Sprintf("%s does not exist in this tree", path)
	}
	if from > 0 && (to < from || to > n) {
		return fmt.Sprintf("%s has %d lines; the reference names lines %d-%d", path, n, from, to)
	}
	return ""
}

func clean(p string) string {
	var out []string
	for _, s := range strings.Split(p, "/") {
		switch s {
		case "", ".":
		case "..":
			if len(out) > 0 {
				out = out[:len(out)-1]
			}
		default:
			out = append(out, s)
		}
	}
	return strings.Join(out, "/")
}

package kernelimpls

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"regexp"
	"slices"
	"strings"

	check "github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

// Load reads and validates the table at path.
func Load(path string) (*Table, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(path, data)
}

// Parse reads a table strictly and validates it. An engine writes the table as JSON (Marshal);
// it is decoded as every BLIS document is (check.DecodeStrict, and JSON is YAML): an unknown
// key, a key that appears twice, a second document or a number that does not convert is
// refused. name is the file or source, for messages.
func Parse(name string, data []byte) (*Table, error) {
	var t Table
	if err := check.DecodeStrict(name, data, &t); err != nil {
		return nil, err
	}
	if err := t.Validate(); err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	return &t, nil
}

// Marshal writes the table's canonical bytes: JSON indented by two spaces, keys in the declared
// order, no HTML escaping, and a final newline. An engine repository checks its committed file
// by comparing it with Marshal of what Parse read.
func (t *Table) Marshal() ([]byte, error) {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(t); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

var (
	commitRE = regexp.MustCompile(`^[0-9a-f]{40}$`)
	issueRE  = regexp.MustCompile(`^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#[1-9][0-9]*$`)
	archRE   = regexp.MustCompile(`^sm_[0-9]{2,3}[af]?$`)
	citeRE   = regexp.MustCompile(`^[^\s:@]+:[1-9][0-9]*@([0-9a-f]{8})$`)
)

// checkCite records a problem unless s cites path:line@commit8 at the release's commit.
func (t *Table) checkCite(p *check.Problems, at, s string) {
	m := citeRE.FindStringSubmatch(s)
	switch {
	case m == nil:
		p.Field(at, "%q must cite path:line@commit8", s)
	case len(t.Engine.Commit) >= 8 && m[1] != t.Engine.Commit[:8]:
		p.Field(at, "%q cites commit %s, not the release's %s", s, m[1], t.Engine.Commit[:8])
	}
}

// oneLine records a problem when a value that CodeHash or a key writes holds a line break.
func oneLine(p *check.Problems, at, s string) {
	if strings.ContainsAny(s, "\n\r") {
		p.Field(at, "holds a line break")
	}
}

// Validate checks the table and returns every problem found, or nil.
func (t *Table) Validate() error {
	var p check.Problems
	if t.Kind != Kind {
		p.Field("kind", "must be %q, got %q", Kind, t.Kind)
	}
	t.validateEngine(&p)
	covered := t.validatePrimitives(&p)
	t.validateGaps(&p, covered)
	archs := t.validateArchs(&p)
	libs := t.validateLibraries(&p)
	t.validateImplementations(&p, covered, libs, archs)
	if len(t.Primitives) > 0 && len(t.Implementations) == 0 {
		p.Field("implementations", "required: a table that covers primitives lists their builds")
	}
	used := map[libKey]bool{}
	for _, r := range t.Implementations {
		used[libKey{r.Library, r.LibraryVersion}] = true
	}
	for i, l := range t.Libraries {
		if !used[libKey{l.Name, l.Version}] {
			p.Field(fmt.Sprintf("libraries[%d]", i), "library %s %s is named by no row", l.Name, l.Version)
		}
	}
	if !p.OK() {
		return errors.New(p.Error())
	}
	return nil
}

func (t *Table) validateEngine(p *check.Problems) {
	if strings.TrimSpace(t.Engine.Kind) == "" {
		p.Field("engine.kind", "required")
	}
	if strings.TrimSpace(t.Engine.Version) == "" {
		p.Field("engine.version", "required")
	}
	if !commitRE.MatchString(t.Engine.Commit) {
		p.Field("engine.commit", "must be the release's commit, 40 lower-case hexadecimal digits, got %q", t.Engine.Commit)
	}
}

func (t *Table) validatePrimitives(p *check.Problems) map[string]bool {
	covered := map[string]bool{}
	if len(t.Primitives) == 0 {
		p.Field("primitives", "required: the primitives whose implementations the table lists completely")
	}
	for i, w := range t.Primitives {
		at := fmt.Sprintf("primitives[%d]", i)
		if !plan.Primitive(w).Valid() {
			p.Field(at, "%q is not a lowered primitive", w)
		}
		if covered[w] {
			p.Field(at, "%q is listed twice", w)
		}
		if i > 0 && t.Primitives[i-1] > w {
			p.Field(at, "%q is out of order: primitives are sorted", w)
		}
		covered[w] = true
	}
	return covered
}

func (t *Table) validateArchs(p *check.Problems) map[string]bool {
	archs := map[string]bool{}
	if len(t.Archs) == 0 {
		p.Field("archs", "required: the device architectures the table lists builds for")
	}
	for i, a := range t.Archs {
		at := fmt.Sprintf("archs[%d]", i)
		if !archRE.MatchString(a) {
			p.Field(at, "%q is not a GPU architecture as the compiler names it (sm_90a, sm_100a)", a)
		}
		if i > 0 && t.Archs[i-1] >= a {
			p.Field(at, "%q is out of order or repeated: archs are sorted, each once", a)
		}
		archs[a] = true
	}
	return archs
}

func (t *Table) validateGaps(p *check.Problems, covered map[string]bool) {
	for i, g := range t.Gaps {
		at := fmt.Sprintf("gaps[%d]", i)
		if !covered[g.Primitive] {
			p.Field(at+".primitive", "%q is not one of the table's primitives", g.Primitive)
		}
		if strings.TrimSpace(g.Selection) == "" {
			p.Field(at+".selection", "required: the upstream choice whose kernels are not listed")
		}
		if !issueRE.MatchString(g.Issue) {
			p.Field(at+".issue", "must name the issue as <owner>/<repo>#<n>, got %q", g.Issue)
		}
		if (g.Library == "") != (g.Variant == "") {
			p.Field(at, "a gap names both the library and the variant of the one kernel it covers, or neither")
		}
		oneLine(p, at+".selection", g.Selection)
		if i > 0 {
			prev := t.Gaps[i-1]
			if slices.Compare([]string{prev.Primitive, prev.Selection, prev.Library, prev.Variant},
				[]string{g.Primitive, g.Selection, g.Library, g.Variant}) >= 0 {
				p.Field(at, "gap %s %q is out of order or repeated: gaps are sorted by primitive, selection, library and variant", g.Primitive, g.Selection)
			}
		}
	}
}

type libKey struct{ name, version string }

func (t *Table) validateLibraries(p *check.Problems) map[libKey]bool {
	libs := map[libKey]bool{}
	pinned := map[string]int{}
	if len(t.Libraries) == 0 {
		p.Field("libraries", "required: the libraries the rows name")
	}
	for i, l := range t.Libraries {
		oneLine(p, fmt.Sprintf("libraries[%d].name", i), l.Name)
		if l.Pinned && l.Version != t.Engine.Commit && strings.TrimSpace(l.Pin) != "" {
			t.checkCite(p, fmt.Sprintf("libraries[%d].pin", i), l.Pin)
		}
		at := fmt.Sprintf("libraries[%d]", i)
		if strings.TrimSpace(l.Name) == "" {
			p.Field(at+".name", "required")
		}
		if strings.TrimSpace(l.Version) == "" {
			p.Field(at+".version", "required")
		}
		if strings.TrimSpace(l.Pin) == "" {
			p.Field(at+".pin", "required: where the version is stated")
		}
		if strings.TrimSpace(l.Origin) == "" {
			p.Field(at+".origin", "required: the immutable identity of what was hashed")
		}
		k := libKey{l.Name, l.Version}
		if libs[k] {
			p.Field(at, "library %s %s is listed twice", l.Name, l.Version)
		}
		if i > 0 {
			prev := t.Libraries[i-1]
			if prev.Name > l.Name || (prev.Name == l.Name && prev.Version > l.Version) {
				p.Field(at, "library %s %s is out of order: libraries are sorted by name, then version", l.Name, l.Version)
			}
		}
		libs[k] = true
		if l.Pinned {
			pinned[l.Name]++
		} else if _, ok := pinned[l.Name]; !ok {
			pinned[l.Name] = 0
		}
	}
	for _, name := range sortedKeys(pinned) {
		if n := pinned[name]; n != 1 {
			p.Field("libraries", "library %s has %d pinned versions; the release pins exactly one", name, n)
		}
	}
	return libs
}

type rowKey struct{ primitive, selection, library, version, variant, arch string }

func keyOf(r Implementation) rowKey {
	return rowKey{r.Primitive, r.Selection, r.Library, r.LibraryVersion, r.Variant, r.TargetArch}
}

func (k rowKey) less(o rowKey) bool {
	return slices.Compare([]string{k.primitive, k.selection, k.library, k.version, k.variant, k.arch},
		[]string{o.primitive, o.selection, o.library, o.version, o.variant, o.arch}) < 0
}

type buildKey struct{ library, version, variant, arch string }

func (t *Table) validateImplementations(p *check.Problems, covered map[string]bool, libs map[libKey]bool, archs map[string]bool) {
	seen := map[rowKey]bool{}
	hashOfBuild := map[buildKey]string{}
	for i, r := range t.Implementations {
		at := fmt.Sprintf("implementations[%d]", i)
		k := keyOf(r)
		if strings.TrimSpace(r.Selection) == "" {
			p.Field(at+".selection", "required: the upstream choice that runs the kernel")
		}
		oneLine(p, at+".selection", r.Selection)
		oneLine(p, at+".library", r.Library)
		oneLine(p, at+".variant", r.Variant)
		if r.Primitive != "" && !covered[r.Primitive] {
			p.Field(at+".primitive", "%q is not one of the table's primitives", r.Primitive)
		}
		if r.Primitive == "" {
			p.Field(at+".primitive", "required")
		}
		if !libs[libKey{r.Library, r.LibraryVersion}] {
			p.Field(at+".library", "library %s %s is not in libraries", r.Library, r.LibraryVersion)
		}
		if strings.TrimSpace(r.Variant) == "" {
			p.Field(at+".variant", "required: the library's own exported kernel name")
		}
		if !archs[r.TargetArch] {
			p.Field(at+".target_arch", "%q is not one of the table's archs", r.TargetArch)
		}
		if r.HashBasis != HashSource && r.HashBasis != HashBinary {
			p.Field(at+".hash_basis", "must be %q or %q, got %q", HashSource, HashBinary, r.HashBasis)
		}
		validateSortedUnique(p, at+".hash_inputs", r.HashInputs, "the content identities the hash covers")
		if strings.TrimSpace(r.Toolkit) == "" {
			p.Field(at+".toolkit", "required: the compiler or CUDA toolkit version that builds the kernel")
		}
		validateSortedUnique(p, at+".selected_by", r.SelectedBy, "the upstream code that can select the kernel")
		for j, c := range r.SelectedBy {
			t.checkCite(p, fmt.Sprintf("%s.selected_by[%d]", at, j), c)
		}
		// The hash is recomputed from the row, so a hash copied onto another kernel, library or
		// architecture is refused: one hash identifies one kernel build.
		if !check.IsSHA256(r.CodeHash) {
			p.Field(at+".code_hash", "must be a SHA-256, 64 lower-case hexadecimal digits, got %q", r.CodeHash)
		} else if want := CodeHash(r.HashBasis, r.Library, r.Variant, r.TargetArch, r.HashInputs); want != r.CodeHash {
			p.Field(at+".code_hash", "is not the hash of the row's basis, library, variant, architecture and inputs (CodeHash gives %s)", want)
		}
		if seen[k] {
			p.Field(at, "row %s %q %s %s %s %s is listed twice", r.Primitive, r.Selection, r.Library, r.LibraryVersion, r.Variant, r.TargetArch)
		}
		if i > 0 {
			if k.less(keyOf(t.Implementations[i-1])) {
				p.Field(at, "row is out of order: rows are sorted by primitive, selection, library, library version, variant and target architecture")
			}
		}
		seen[k] = true
		// One build of a kernel has one hash, whichever primitive it serves.
		b := buildKey{r.Library, r.LibraryVersion, r.Variant, r.TargetArch}
		if h, ok := hashOfBuild[b]; ok && h != r.CodeHash {
			p.Field(at+".code_hash", "%s %s %s for %s has two hashes, %s and %s; one build has one hash", r.Library,
				r.LibraryVersion, r.Variant, r.TargetArch, h, r.CodeHash)
		}
		hashOfBuild[b] = r.CodeHash
	}
}

func validateSortedUnique(p *check.Problems, at string, list []string, what string) {
	if len(list) == 0 {
		p.Field(at, "required: %s", what)
	}
	for i, s := range list {
		switch {
		case strings.TrimSpace(s) == "":
			p.Field(fmt.Sprintf("%s[%d]", at, i), "is empty")
		case strings.ContainsAny(s, "\n\r"):
			p.Field(fmt.Sprintf("%s[%d]", at, i), "holds a line break")
		}
		if i > 0 && list[i-1] >= s {
			p.Field(fmt.Sprintf("%s[%d]", at, i), "%q is out of order or repeated: the list is sorted, each once", s)
		}
	}
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	slices.Sort(out)
	return out
}

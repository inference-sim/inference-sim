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
	archRE   = regexp.MustCompile(`^sm_[0-9]{2,3}[af]?$`)
)

// Validate checks the table and returns every problem found, or nil.
func (t *Table) Validate() error {
	var p check.Problems
	if t.Kind != Kind {
		p.Field("kind", "must be %q, got %q", Kind, t.Kind)
	}
	t.validateEngine(&p)
	covered := t.validatePrimitives(&p)
	libs := t.validateLibraries(&p)
	t.validateImplementations(&p, covered, libs)
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

type libKey struct{ name, version string }

func (t *Table) validateLibraries(p *check.Problems) map[libKey]bool {
	libs := map[libKey]bool{}
	pinned := map[string]int{}
	for i, l := range t.Libraries {
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

type rowKey struct{ primitive, library, version, variant, arch string }

func (k rowKey) less(o rowKey) bool {
	return slices.Compare([]string{k.primitive, k.library, k.version, k.variant, k.arch},
		[]string{o.primitive, o.library, o.version, o.variant, o.arch}) < 0
}

type buildKey struct{ library, version, variant, arch string }

func (t *Table) validateImplementations(p *check.Problems, covered map[string]bool, libs map[libKey]bool) {
	seen := map[rowKey]bool{}
	hashOfBuild := map[buildKey]string{}
	for i, r := range t.Implementations {
		at := fmt.Sprintf("implementations[%d]", i)
		k := rowKey{r.Primitive, r.Library, r.LibraryVersion, r.Variant, r.TargetArch}
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
		if !archRE.MatchString(r.TargetArch) {
			p.Field(at+".target_arch", "%q is not a GPU architecture as the compiler names it (sm_90a, sm_100a)", r.TargetArch)
		}
		if r.HashBasis != HashSource && r.HashBasis != HashBinary {
			p.Field(at+".hash_basis", "must be %q or %q, got %q", HashSource, HashBinary, r.HashBasis)
		}
		validateSortedUnique(p, at+".hash_inputs", r.HashInputs, "the content identities the hash covers")
		if strings.TrimSpace(r.Toolkit) == "" {
			p.Field(at+".toolkit", "required: the CUDA toolkit version of the release's build")
		}
		validateSortedUnique(p, at+".selected_by", r.SelectedBy, "the upstream code that can select the kernel")
		// The hash is recomputed from the row, so a hash copied onto another kernel, library or
		// architecture is refused: one hash identifies one kernel build.
		if !check.IsSHA256(r.CodeHash) {
			p.Field(at+".code_hash", "must be a SHA-256, 64 lower-case hexadecimal digits, got %q", r.CodeHash)
		} else if want := CodeHash(r.HashBasis, r.Library, r.Variant, r.TargetArch, r.HashInputs); want != r.CodeHash {
			p.Field(at+".code_hash", "is not the hash of the row's basis, library, variant, architecture and inputs (CodeHash gives %s)", want)
		}
		if seen[k] {
			p.Field(at, "row %s %s %s %s %s is listed twice", r.Primitive, r.Library, r.LibraryVersion, r.Variant, r.TargetArch)
		}
		if i > 0 {
			prev := t.Implementations[i-1]
			if k.less(rowKey{prev.Primitive, prev.Library, prev.LibraryVersion, prev.Variant, prev.TargetArch}) {
				p.Field(at, "row is out of order: rows are sorted by primitive, library, library version, variant and target architecture")
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

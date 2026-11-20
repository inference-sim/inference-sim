package validate

import (
	"strings"
	"testing"
)

// SUB-12 at its root: DecodeStrict refuses each malformed document its caller lists, and
// accepts the valid one. Every loader that ends here inherits the rules.
func TestDecodeStrictRefusesTheMalformedSet(t *testing.T) {
	type doc struct {
		Kind string  `yaml:"kind"`
		N    int     `yaml:"n"`
		F    float64 `yaml:"f"`
		List []int   `yaml:"list"`
	}
	var ok doc
	if err := DecodeStrict("ok", []byte("kind: a\nn: 3\nf: 1.5\nlist: [1, 2]\n"), &ok); err != nil || ok.N != 3 || len(ok.List) != 2 {
		t.Fatalf("a valid document was refused or misread: %v %+v", err, ok)
	}
	for name, c := range map[string]struct{ in, want string }{
		"unknown key":               {"kind: a\nbogus: 1\n", "bogus"},
		"repeated key":              {"kind: a\nkind: b\n", "kind"},
		"fraction in integer":       {"kind: a\nn: 2.5\n", "n"},
		"integer float cannot hold": {"kind: a\nf: 9007199254740993\n", "f"},
		"nan":                       {"kind: a\nf: .nan\n", "f"},
		"infinity":                  {"kind: a\nf: .inf\n", "f"},
		"recursive alias":           {"kind: &a [*a]\n", "alias"},
		"second document":           {"kind: a\n---\nkind: b\n", "second"},
		"too large":                 {strings.Repeat("# x\n", MaxDocumentBytes/4+1) + "kind: a\n", "limited"},
	} {
		var d doc
		err := DecodeStrict("doc.yaml", []byte(c.in), &d)
		if err == nil {
			t.Errorf("%s: accepted, decoded %+v", name, d)
			continue
		}
		if !strings.Contains(err.Error(), "doc.yaml") || !strings.Contains(strings.ToLower(err.Error()), c.want) {
			t.Errorf("%s: error %q does not name the file and %q", name, err, c.want)
		}
	}
	// A sequence-typed target is held to the same rules.
	var list []doc
	if err := DecodeStrict("l.yaml", []byte("- kind: a\n  n: 1.5\n"), &list); err == nil {
		t.Error("a fraction in an integer field of a list element was accepted")
	}
}

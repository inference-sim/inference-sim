package deployment

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// holder is a document carrying an engine block, as a pool does.
type holder struct {
	Engine EngineBlock `yaml:"engine,omitempty"`
	After  string      `yaml:"after,omitempty"`
}

// An engine block passes through unchanged: what was decoded is written back, every key and
// value as given, in the order given, including keys no schema type names.
func TestEngineBlockRoundTripsTheNodeAsGiven(t *testing.T) {
	const doc = "engine:\n" +
		"    zeta: 1\n" +
		"    block_size: 16\n" +
		"    nested:\n" +
		"        a: [1, 2]\n" +
		"        b: text\n" +
		"    unknown_to_any_schema: true\n" +
		"after: x\n"
	var h holder
	if err := yaml.Unmarshal([]byte(doc), &h); err != nil {
		t.Fatal(err)
	}
	if h.Engine.IsZero() {
		t.Fatal("a stated block decoded as absent")
	}
	out, err := yaml.Marshal(h)
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != doc {
		t.Errorf("round trip changed the block:\ngot\n%s\nwant\n%s", out, doc)
	}
}

// An absent block is zero, is omitted on encoding, and decodes nothing.
func TestEngineBlockAbsent(t *testing.T) {
	var h holder
	if err := yaml.Unmarshal([]byte("after: x\n"), &h); err != nil {
		t.Fatal(err)
	}
	if !h.Engine.IsZero() || h.Engine.Node() != nil {
		t.Fatal("an absent block is not zero")
	}
	out, err := yaml.Marshal(h)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(out), "engine") {
		t.Errorf("an absent block was written: %s", out)
	}
	// Without omitempty an absent block is an empty mapping, as a pool stating no engine
	// settings always encoded.
	bare, err := yaml.Marshal(struct {
		Engine EngineBlock `yaml:"engine"`
	}{})
	if err != nil || string(bare) != "engine: {}\n" {
		t.Errorf("an absent block without omitempty encodes as %q, %v; want %q", bare, err, "engine: {}\n")
	}
	v := struct{ BlockSize int }{BlockSize: 7}
	if err := h.Engine.Decode(&v); err != nil || v.BlockSize != 7 {
		t.Errorf("decoding an absent block changed the value or failed: %+v, %v", v, err)
	}
	if !NewEngineBlock(nil).IsZero() {
		t.Error("NewEngineBlock(nil) is not zero")
	}
}

// Decode is strict: a key the target has no field for is an error, not dropped.
func TestEngineBlockDecodeIsStrict(t *testing.T) {
	var h holder
	if err := yaml.Unmarshal([]byte("engine:\n  block_size: 16\n  blok_size: 32\n"), &h); err != nil {
		t.Fatal(err)
	}
	var v struct {
		BlockSize int `yaml:"block_size"`
	}
	err := h.Engine.Decode(&v)
	// The decoder's own words and the key's line in the document the block came from, as a
	// strict decode of the whole document gives them.
	if err == nil || !strings.Contains(err.Error(), "line 3: field blok_size not found in type struct") {
		t.Fatalf("an unknown key decoded: %v", err)
	}
	var ok struct {
		BlockSize int `yaml:"block_size"`
		BlokSize  int `yaml:"blok_size"`
	}
	if err := h.Engine.Decode(&ok); err != nil || ok.BlockSize != 16 || ok.BlokSize != 32 {
		t.Errorf("decode = %+v, %v", ok, err)
	}
}

// The block never shares nodes with its caller: editing the node a block was built from, or
// the copy Node returns, leaves the block as it was.
func TestEngineBlockCopies(t *testing.T) {
	var n yaml.Node
	if err := yaml.Unmarshal([]byte("block_size: 16\n"), &n); err != nil {
		t.Fatal(err)
	}
	b := NewEngineBlock(n.Content[0])
	n.Content[0].Content[1].Value = "99"
	got := b.Node()
	got.Content[1].Value = "77"
	var v struct {
		BlockSize int `yaml:"block_size"`
	}
	if err := b.Decode(&v); err != nil || v.BlockSize != 16 {
		t.Errorf("the block changed with its caller's node: %+v, %v", v, err)
	}
}

// Strictness reaches every level a decoder descends to: a nested struct, a pointer to one, a
// slice of them and a map's values; a type that decodes itself is its own to check.
func TestEngineBlockDecodeIsStrictAtEveryLevel(t *testing.T) {
	type inner struct {
		A int `yaml:"a"`
	}
	type target struct {
		Ptr  *inner           `yaml:"ptr"`
		List []inner          `yaml:"list"`
		Map  map[string]inner `yaml:"map"`
		Self selfDecoding     `yaml:"self"`
	}
	for doc, bad := range map[string]string{
		"ptr: {a: 1, b: 2}\n":                "field b not found",
		"list: [{a: 1}, {c: 3}]\n":           "field c not found",
		"map: {k: {a: 1, d: 4}}\n":           "field d not found",
		"self: {anything: 1}\nptr: {a: 1}\n": "",
	} {
		var n yaml.Node
		if err := yaml.Unmarshal([]byte(doc), &n); err != nil {
			t.Fatal(err)
		}
		var v target
		err := NewEngineBlock(n.Content[0]).Decode(&v)
		switch {
		case bad == "" && err != nil:
			t.Errorf("%q: %v", doc, err)
		case bad != "" && (err == nil || !strings.Contains(err.Error(), bad)):
			t.Errorf("%q: got %v, want %q", doc, err, bad)
		}
	}
}

// selfDecoding takes any mapping, as a yaml.Unmarshaler may.
type selfDecoding struct{}

func (*selfDecoding) UnmarshalYAML(*yaml.Node) error { return nil }

// Equal compares what a block holds, not where it was read from: a block decoded from a file
// equals one built from the same YAML, and a different value is not equal.
func TestEngineBlockEqual(t *testing.T) {
	var read holder
	if err := yaml.Unmarshal([]byte("after: x\nengine:\n    block_size: 16\n    nested: {a: 1}\n"), &read); err != nil {
		t.Fatal(err)
	}
	var n yaml.Node
	if err := yaml.Unmarshal([]byte("block_size: 16\nnested:\n  a: 1\n"), &n); err != nil {
		t.Fatal(err)
	}
	if !read.Engine.Equal(NewEngineBlock(n.Content[0])) {
		t.Error("the same YAML read at another position is not equal")
	}
	n.Content[0].Content[1].Value = "32"
	if read.Engine.Equal(NewEngineBlock(n.Content[0])) {
		t.Error("blocks with different values are equal")
	}
	if !(EngineBlock{}).Equal(EngineBlock{}) || read.Engine.Equal(EngineBlock{}) {
		t.Error("absent blocks: equal to each other only")
	}
}

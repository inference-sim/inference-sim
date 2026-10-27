// Original to blis-vllm.

package engineconfig

import (
	"go/build"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"gopkg.in/yaml.v3"
)

// block reads a file as a pool's engine block, as a deployment's decoding holds it.
func block(t *testing.T, path string) deployment.EngineBlock {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var n yaml.Node
	if err := yaml.Unmarshal(raw, &n); err != nil {
		t.Fatal(err)
	}
	return deployment.NewEngineBlock(n.Content[0])
}

// Decode then Encode writes an engine block exactly as blis-schemas encoded its Engine struct
// before it held the block unopened: every field, in the struct's order, with its omitempty
// tags. testdata/every-field.encoded.yaml is that encoding of testdata/every-field.yaml, taken
// from blis-schemas' deployment.Engine at 70246460a379f8b1898fbf6c36a3091364118950, so a
// consumer that re-encodes a deployment (a harness writing engine fields per point) writes the
// same bytes as before. indexer_kv_dtype was added after the move, at its place in the struct.
func TestDecodeThenEncodeIsTodaysEncoding(t *testing.T) {
	e, err := Decode(block(t, "testdata/every-field.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	b, err := Encode(e)
	if err != nil {
		t.Fatal(err)
	}
	got, err := yaml.Marshal(b)
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile("testdata/every-field.encoded.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Errorf("Decode then Encode wrote\n%s\nwant\n%s", got, want)
	}
	// And Encode is Decode's inverse: decoding what it wrote gives the same Engine.
	again, err := Decode(b)
	if err != nil || !reflect.DeepEqual(again, e) {
		t.Errorf("Decode(Encode(e)) = %+v, %v; want %+v", again, err, e)
	}
}

// Every field of Engine is stated in the every-field fixture, so the encoding test above
// covers each one: a field added to Engine without a value there fails here.
func TestEveryFieldFixtureStatesEveryField(t *testing.T) {
	e, err := Decode(block(t, "testdata/every-field.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	v := reflect.ValueOf(e)
	for i := 0; i < v.NumField(); i++ {
		if v.Field(i).IsZero() {
			t.Errorf("testdata/every-field.yaml states no %s", v.Type().Field(i).Name)
		}
	}
}

// Decode is strict: a misspelled key is an error naming it at its line, not a setting
// silently dropped.
func TestDecodeRefusesAnUnknownKey(t *testing.T) {
	var n yaml.Node
	if err := yaml.Unmarshal([]byte("block_size: 16\ncache_dytpe: fp8\n"), &n); err != nil {
		t.Fatal(err)
	}
	_, err := Decode(deployment.NewEngineBlock(n.Content[0]))
	if err == nil || !strings.Contains(err.Error(), "line 2: field cache_dytpe not found in type engineconfig.Engine") {
		t.Fatalf("Decode = %v, want the unknown key refused at its line", err)
	}
	if _, err := Decode(deployment.NewEngineBlock(nil)); err != nil {
		t.Errorf("an absent block: %v", err)
	}
}

// engineconfig imports only blis-schemas and yaml.v3 beside the standard library: a cost
// model reads it, and the rules packs build on it, so it may import neither a pack nor a cost
// model.
func TestEngineconfigImportsOnlyBlisSchemasAndYAML(t *testing.T) {
	pkg, err := build.ImportDir(".", 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, imp := range pkg.Imports {
		if !strings.Contains(imp, ".") {
			continue // the standard library
		}
		if !strings.HasPrefix(imp, "github.com/inference-sim/blis-schemas/") && imp != "gopkg.in/yaml.v3" {
			t.Errorf("engineconfig imports %s", imp)
		}
	}
}

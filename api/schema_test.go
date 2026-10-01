package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"slices"
	"testing"
)

// TestCommittedSchemaIsCurrent is the CI staleness gate the issue asks for: the committed
// schema must be byte-identical to what the Go types derive right now. A change to the
// envelope types without `go generate ./api/...` fails here, so the published contract
// cannot drift away from the code that implements it.
func TestCommittedSchemaIsCurrent(t *testing.T) {
	derived, err := GenerateSchema()
	if err != nil {
		t.Fatalf("GenerateSchema: %v", err)
	}
	committed, err := os.ReadFile(SchemaPath)
	if err != nil {
		t.Fatalf("read %s: %v", SchemaPath, err)
	}
	if !bytes.Equal(derived, committed) {
		t.Errorf("%s is STALE: it differs from the schema derived from the Go types.\n"+
			"Regenerate it with:\n    go generate ./api/...\n"+
			"derived (%d bytes):\n%s\ncommitted (%d bytes):\n%s",
			SchemaPath, len(derived), derived, len(committed), committed)
	}
	if !bytes.Equal(committed, SchemaJSON()) {
		t.Errorf("the embedded schema differs from %s on disk, which should be impossible "+
			"(the go:embed directive names that path)", SchemaPath)
	}
}

// TestGenerateSchemaIsDeterministic covers INV-6 for the generator: regeneration must be a
// zero-diff operation, so a map iteration order leaking into the output would show up as a
// spurious schema change on an unrelated PR.
func TestGenerateSchemaIsDeterministic(t *testing.T) {
	first, err := GenerateSchema()
	if err != nil {
		t.Fatalf("GenerateSchema: %v", err)
	}
	for i := 0; i < 20; i++ {
		again, err := GenerateSchema()
		if err != nil {
			t.Fatalf("GenerateSchema (run %d): %v", i, err)
		}
		if !bytes.Equal(first, again) {
			t.Fatalf("GenerateSchema is not deterministic; run %d differs:\n%s\nvs\n%s", i, first, again)
		}
	}
}

// TestSchemaJSONReturnsACopy guards the embedded bytes: a caller that mutated them would
// corrupt every later validation in the process.
func TestSchemaJSONReturnsACopy(t *testing.T) {
	stolen := SchemaJSON()
	if len(stolen) == 0 {
		t.Fatal("SchemaJSON is empty; the embed did not happen")
	}
	stolen[0] = 'X'
	if SchemaJSON()[0] == 'X' {
		t.Error("SchemaJSON handed out the embedded slice itself")
	}
}

// TestSchemaDerivesTheEnvelopeContract checks the derived schema says what the Go types say,
// field by field, instead of comparing it to a golden copy (R7/R12): a golden-only check
// would happily enshrine a wrong enum.
func TestSchemaDerivesTheEnvelopeContract(t *testing.T) {
	var schema map[string]any
	if err := json.Unmarshal(SchemaJSON(), &schema); err != nil {
		t.Fatalf("the committed schema is not valid JSON: %v", err)
	}

	if got := schema["$schema"]; got != SchemaDraft {
		t.Errorf("$schema = %v, want %q", got, SchemaDraft)
	}
	if got := schema["$id"]; got != SchemaID {
		t.Errorf("$id = %v, want %q", got, SchemaID)
	}
	if got := schema["type"]; got != "object" {
		t.Errorf("root type = %v, want object", got)
	}
	if got := schema["additionalProperties"]; got != false {
		t.Errorf("root additionalProperties = %v, want false (an unknown top-level key is a typo, R10)", got)
	}

	required, ok := asStrings(schema["required"])
	if !ok {
		t.Fatalf("root required is %T, want an array of strings", schema["required"])
	}
	if want := []string{"apiVersion", "kind"}; !reflect.DeepEqual(required, want) {
		t.Errorf("root required = %v, want %v — the body requirement belongs to the oneOf partition", required, want)
	}

	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("root properties is %T, want an object", schema["properties"])
	}
	for _, name := range []string{"apiVersion", "kind", "spec", "result"} {
		if _, present := properties[name]; !present {
			t.Errorf("the schema has no %q property; Document declares it", name)
		}
	}

	apiVersion, _ := properties["apiVersion"].(map[string]any)
	if got := apiVersion["const"]; got != string(Version) {
		t.Errorf("apiVersion const = %v, want %q (read off the Version constant)", got, Version)
	}

	kind, _ := properties["kind"].(map[string]any)
	gotEnum, ok := asStrings(kind["enum"])
	if !ok {
		t.Fatalf("kind enum is %T, want an array of strings", kind["enum"])
	}
	if want := kindStrings(AllKinds()); !reflect.DeepEqual(gotEnum, want) {
		t.Errorf("kind enum = %v, want %v (every kind the Go types define, sorted)", gotEnum, want)
	}

	// The bodies stay OPEN at API-1: constraining them here would freeze fields the later
	// PRs own.
	for _, name := range []string{"spec", "result"} {
		body, _ := properties[name].(map[string]any)
		if got := body["type"]; got != "object" {
			t.Errorf("%s type = %v, want object", name, got)
		}
		if got := body["additionalProperties"]; got != true {
			t.Errorf("%s additionalProperties = %v, want true (the body is opaque at API-1)", name, got)
		}
	}

	branches, ok := schema["oneOf"].([]any)
	if !ok || len(branches) != 2 {
		t.Fatalf("oneOf = %v, want exactly two branches (input, output)", schema["oneOf"])
	}
	for i, want := range []struct {
		kinds    []Kind
		body     string
		excluded string
	}{
		{InputKinds(), "spec", "result"},
		{OutputKinds(), "result", "spec"},
	} {
		branch, _ := branches[i].(map[string]any)
		branchRequired, _ := asStrings(branch["required"])
		if !reflect.DeepEqual(branchRequired, []string{want.body}) {
			t.Errorf("oneOf[%d] required = %v, want [%s]", i, branchRequired, want.body)
		}
		branchProperties, _ := branch["properties"].(map[string]any)
		branchKind, _ := branchProperties["kind"].(map[string]any)
		branchEnum, _ := asStrings(branchKind["enum"])
		if !reflect.DeepEqual(branchEnum, kindStrings(want.kinds)) {
			t.Errorf("oneOf[%d] kind enum = %v, want %v", i, branchEnum, kindStrings(want.kinds))
		}
		not, _ := branch["not"].(map[string]any)
		notRequired, _ := asStrings(not["required"])
		if !reflect.DeepEqual(notRequired, []string{want.excluded}) {
			t.Errorf("oneOf[%d] not.required = %v, want [%s] — a document must carry one body, not both",
				i, notRequired, want.excluded)
		}
	}
}

// TestSchemaUsesOnlyImplementedKeywords closes the loop between the generator and the
// checker: every keyword the generator emits must be one the checker enforces (or an
// explicit annotation). Without this, adding a keyword to the generator would make
// validation quietly incomplete rather than loud.
func TestSchemaUsesOnlyImplementedKeywords(t *testing.T) {
	var schema any
	if err := json.Unmarshal(SchemaJSON(), &schema); err != nil {
		t.Fatalf("the committed schema is not valid JSON: %v", err)
	}
	known := append(slices.Clone(annotationKeywords), assertionKeywords...)

	// The walk only descends where a schema object can appear, so a property NAMED after a
	// keyword (a document field called "type") is not mistaken for one.
	var walk func(node map[string]any, path string)
	walk = func(node map[string]any, path string) {
		for _, keyword := range sortedKeys(node) {
			if !slices.Contains(known, keyword) {
				t.Errorf("%s uses keyword %q, which the validator does not implement; "+
					"teach checker.check about it in the same change", path, keyword)
			}
		}
		if properties, ok := node["properties"].(map[string]any); ok {
			for _, name := range sortedKeys(properties) {
				if sub, ok := properties[name].(map[string]any); ok {
					walk(sub, path+".properties."+name)
				}
			}
		}
		for _, keyword := range []string{"items", "not", "additionalProperties"} {
			if sub, ok := node[keyword].(map[string]any); ok {
				walk(sub, path+"."+keyword)
			}
		}
		if branches, ok := node["oneOf"].([]any); ok {
			for i, branch := range branches {
				if sub, ok := branch.(map[string]any); ok {
					walk(sub, fmt.Sprintf("%s.oneOf[%d]", path, i))
				}
			}
		}
	}
	root, ok := schema.(map[string]any)
	if !ok {
		t.Fatalf("the committed schema is a %T, want an object", schema)
	}
	walk(root, "$")
}

// TestSchemaForTypeRefusesAnUntranslatableType is the R1 guard on the generator: an
// unhandled type must abort generation, not emit a schema that constrains nothing.
func TestSchemaForTypeRefusesAnUntranslatableType(t *testing.T) {
	type withChannel struct {
		Pipe chan int `json:"pipe"`
	}
	if _, err := schemaForType(reflect.TypeOf(withChannel{})); err == nil {
		t.Error("a struct with a channel field produced a schema; an untranslatable type must be refused")
	}

	type withNonStringKey struct {
		Table map[int]string `json:"table"`
	}
	if _, err := schemaForType(reflect.TypeOf(withNonStringKey{})); err == nil {
		t.Error("a map with a non-string key produced a schema; a JSON object key is always a string")
	}
}

// TestSchemaForTypeDerivesRequiredFromOmitempty pins the rule the envelope's required list
// depends on, on a type the envelope does not itself exercise in every combination.
func TestSchemaForTypeDerivesRequiredFromOmitempty(t *testing.T) {
	type sample struct {
		Always   string  `json:"always"`
		Optional string  `json:"optional,omitempty"`
		Renamed  int     `json:"renamed_field"`
		Hidden   string  `json:"-"`
		Pointer  *string `json:"pointer,omitempty"`
		Untagged bool
	}
	schema, err := schemaForType(reflect.TypeOf(sample{}))
	if err != nil {
		t.Fatalf("schemaForType: %v", err)
	}
	required, _ := asStrings(schema["required"])
	if want := []string{"Untagged", "always", "renamed_field"}; !reflect.DeepEqual(required, want) {
		t.Errorf("required = %v, want %v", required, want)
	}
	properties, _ := schema["properties"].(map[string]any)
	if _, present := properties["Hidden"]; present {
		t.Error(`a json:"-" field appears in the schema but is never serialized`)
	}
	pointer, _ := properties["pointer"].(map[string]any)
	if got := pointer["type"]; got != "string" {
		t.Errorf("*string property type = %v, want string (a pointer has its element's document shape)", got)
	}
}

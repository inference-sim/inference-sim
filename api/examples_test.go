package api

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// examplesDir holds one minimal envelope per kind. They are documentation first — the thing
// a reader of the schema copies — and a test corpus second, which is why they live in
// examples/ rather than testdata/.
const examplesDir = "examples"

// exampleFileFor is the naming rule: kind Run lives in run.yaml, kind RunResult in
// run-result.yaml. Deriving the path from the kind (rather than listing files and hoping) is
// what makes "every kind has an example" checkable.
func exampleFileFor(kind Kind) string {
	name := strings.ToLower(string(kind))
	if result, ok := strings.CutSuffix(name, strings.ToLower(string(resultSuffix))); ok {
		name = result + "-result"
	}
	return filepath.Join(examplesDir, name+".yaml")
}

// TestEveryKindHasExactlyOneExample is the issue's "at least one example document per planned
// kind" criterion, as a law: the directory and the kind list must correspond exactly, in both
// directions. A kind added without an example fails; a stale example left behind after a kind
// is renamed fails too.
func TestEveryKindHasExactlyOneExample(t *testing.T) {
	kinds := AllKinds()
	if len(kinds) != 10 {
		t.Errorf("non-vacuity: expected the ten planned kinds (five verbs × input/output), got %d: %v",
			len(kinds), kinds)
	}

	expected := map[string]Kind{}
	for _, kind := range kinds {
		path := exampleFileFor(kind)
		expected[filepath.Base(path)] = kind
		if _, err := os.Stat(path); err != nil {
			t.Errorf("kind %q has no example document at %s: %v", kind, path, err)
		}
	}

	entries, err := os.ReadDir(examplesDir)
	if err != nil {
		t.Fatalf("read %s: %v", examplesDir, err)
	}
	for _, entry := range entries {
		if entry.IsDir() {
			t.Errorf("%s contains a subdirectory %q; the corpus is flat, one file per kind",
				examplesDir, entry.Name())
			continue
		}
		if _, want := expected[entry.Name()]; !want {
			t.Errorf("%s/%s does not correspond to any kind; the envelope defines %v",
				examplesDir, entry.Name(), kinds)
		}
	}
}

// TestExamplesValidateAgainstTheCommittedSchema is the criterion the external contract rests
// on: the published schema accepts every documented envelope.
func TestExamplesValidateAgainstTheCommittedSchema(t *testing.T) {
	for _, kind := range AllKinds() {
		t.Run(string(kind), func(t *testing.T) {
			path := exampleFileFor(kind)
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read %s: %v", path, err)
			}
			if err := ValidateDocument(data); err != nil {
				t.Errorf("%s does not validate against %s: %v", path, SchemaPath, err)
			}
		})
	}
}

// TestExamplesParseStrictlyIntoTheEnvelopeTypes checks the other half of "the schema is
// derived from the Go types": every example must also decode into [Document] with
// KnownFields(true), carry the kind its filename claims, and satisfy Document.Validate. A
// struct tag that disagreed with the schema's property names would fail here.
func TestExamplesParseStrictlyIntoTheEnvelopeTypes(t *testing.T) {
	for _, kind := range AllKinds() {
		t.Run(string(kind), func(t *testing.T) {
			path := exampleFileFor(kind)
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read %s: %v", path, err)
			}

			var document Document
			decoder := yaml.NewDecoder(strings.NewReader(string(data)))
			decoder.KnownFields(true)
			if err := decoder.Decode(&document); err != nil {
				t.Fatalf("strict decode of %s into Document: %v", path, err)
			}
			if document.Kind != kind {
				t.Errorf("%s declares kind %q, but its name says %q", path, document.Kind, kind)
			}
			if err := document.Validate(); err != nil {
				t.Errorf("%s is rejected by Document.Validate: %v", path, err)
			}
			// The bodies are opaque but must be PRESENT and empty at API-1; a placeholder
			// field here would read as part of the contract.
			body := document.Spec
			if IsOutput(kind) {
				body = document.Result
			}
			if body == nil {
				t.Fatalf("%s has no %s", path, BodyField(kind))
			}
			if len(*body) != 0 {
				t.Errorf("%s carries %d %s field(s); API-1 examples are minimal envelopes, and a "+
					"placeholder body would read as part of the contract", path, len(*body), BodyField(kind))
			}
		})
	}
}

// TestGoValidationAgreesWithTheSchema is the anti-drift law between the two validators. The
// Go types and the published JSON Schema are two statements of one contract, and a divergence
// means external tooling and BLIS disagree about what a valid document is — the schema would
// be a lie. Every case is expressible as a Document, so both sides can judge it.
func TestGoValidationAgreesWithTheSchema(t *testing.T) {
	empty := Body{}
	cases := []struct {
		name     string
		document Document
	}{
		{"valid input envelope", Document{Version, KindRun, &empty, nil}},
		{"valid output envelope", Document{Version, KindConvertResult, nil, &empty}},
		{"wrong apiVersion", Document{"llm-d-perf-simulator/v2", KindRun, &empty, nil}},
		{"empty apiVersion", Document{"", KindRun, &empty, nil}},
		{"undefined kind", Document{Version, "Simulate", &empty, nil}},
		{"empty kind", Document{Version, "", &empty, nil}},
		{"input kind with no spec", Document{Version, KindObserve, nil, nil}},
		{"output kind with no result", Document{Version, KindObserveResult, nil, nil}},
		{"input kind carrying a result", Document{Version, KindRun, &empty, &empty}},
		{"output kind carrying a spec", Document{Version, KindRunResult, &empty, &empty}},
		{"input kind carrying only a result", Document{Version, KindRun, nil, &empty}},
		{"output kind carrying only a spec", Document{Version, KindRunResult, &empty, nil}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoded, err := yaml.Marshal(tc.document)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			goErr := tc.document.Validate()
			schemaErr := ValidateDocument(encoded)
			if (goErr == nil) != (schemaErr == nil) {
				t.Errorf("the Go types and %s disagree about this document:\n%s"+
					"Document.Validate: %v\nschema: %v",
					SchemaPath, encoded, goErr, schemaErr)
			}
		})
	}
}

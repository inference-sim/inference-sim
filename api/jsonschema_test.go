package api

import (
	"strings"
	"testing"
)

// TestValidateDocumentRejectsMalformedEnvelopes is the non-vacuity test for the whole
// harness: if these documents validated, the "every example validates" test would prove
// nothing, because a validator that accepts everything accepts the examples too.
func TestValidateDocumentRejectsMalformedEnvelopes(t *testing.T) {
	cases := []struct {
		name     string
		document string
		wantErr  string // a substring the diagnostic must name
	}{
		{
			name:     "wrong apiVersion",
			document: "apiVersion: llm-d-perf-simulator/v2\nkind: Run\nspec: {}\n",
			wantErr:  "apiVersion",
		},
		{
			name:     "missing apiVersion",
			document: "kind: Run\nspec: {}\n",
			wantErr:  "required property \"apiVersion\"",
		},
		{
			name:     "missing kind",
			document: "apiVersion: llm-d-perf-simulator/v1\nspec: {}\n",
			wantErr:  "required property \"kind\"",
		},
		{
			name:     "undefined kind",
			document: "apiVersion: llm-d-perf-simulator/v1\nkind: Simulate\nspec: {}\n",
			wantErr:  "not one of",
		},
		{
			name:     "kind differing only in case",
			document: "apiVersion: llm-d-perf-simulator/v1\nkind: run\nspec: {}\n",
			wantErr:  "not one of",
		},
		{
			name:     "input kind with no spec",
			document: "apiVersion: llm-d-perf-simulator/v1\nkind: Run\n",
			wantErr:  "alternatives",
		},
		{
			name:     "output kind with no result",
			document: "apiVersion: llm-d-perf-simulator/v1\nkind: RunResult\n",
			wantErr:  "alternatives",
		},
		{
			name:     "input kind carrying a result instead of a spec",
			document: "apiVersion: llm-d-perf-simulator/v1\nkind: Run\nresult: {}\n",
			wantErr:  "alternatives",
		},
		{
			name:     "both bodies at once",
			document: "apiVersion: llm-d-perf-simulator/v1\nkind: Run\nspec: {}\nresult: {}\n",
			wantErr:  "alternatives",
		},
		{
			name:     "unknown top-level key",
			document: "apiVersion: llm-d-perf-simulator/v1\nkind: Run\nspec: {}\nmetadata: {}\n",
			wantErr:  "unknown property \"metadata\"",
		},
		{
			name:     "misspelled top-level key",
			document: "apiVersion: llm-d-perf-simulator/v1\nkind: Run\nspecs: {}\n",
			wantErr:  "unknown property \"specs\"",
		},
		{
			name:     "spec is not an object",
			document: "apiVersion: llm-d-perf-simulator/v1\nkind: Run\nspec: 3\n",
			wantErr:  "expected object",
		},
		{
			name:     "spec is a list",
			document: "apiVersion: llm-d-perf-simulator/v1\nkind: Run\nspec: [a, b]\n",
			wantErr:  "expected object",
		},
		{
			name:     "kind is not a string",
			document: "apiVersion: llm-d-perf-simulator/v1\nkind: 7\nspec: {}\n",
			wantErr:  "expected string",
		},
		{
			name:     "document is not an object",
			document: "- apiVersion: llm-d-perf-simulator/v1\n",
			wantErr:  "expected object",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateDocument([]byte(tc.document))
			if err == nil {
				t.Fatalf("the document was accepted:\n%s", tc.document)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("diagnostic does not mention %q:\n%v", tc.wantErr, err)
			}
		})
	}
}

// TestValidateDocumentAcceptsEveryKindOfEnvelope covers the positive side inline, so the
// validator is exercised even if the examples directory is empty or moves.
func TestValidateDocumentAcceptsEveryKindOfEnvelope(t *testing.T) {
	for _, kind := range AllKinds() {
		t.Run(string(kind), func(t *testing.T) {
			document := "apiVersion: " + string(Version) + "\nkind: " + string(kind) + "\n" +
				BodyField(kind) + ": {}\n"
			if err := ValidateDocument([]byte(document)); err != nil {
				t.Errorf("a minimal %s envelope was rejected: %v\n%s", kind, err, document)
			}
		})
	}
}

// TestJSONAndYAMLValidateIdentically backs the "one parser" design principle: the same
// document in either encoding gets the same verdict.
func TestJSONAndYAMLValidateIdentically(t *testing.T) {
	cases := []struct {
		name  string
		yaml  string
		json  string
		valid bool
	}{
		{
			name:  "valid run document",
			yaml:  "apiVersion: llm-d-perf-simulator/v1\nkind: Run\nspec: {}\n",
			json:  `{"apiVersion":"llm-d-perf-simulator/v1","kind":"Run","spec":{}}`,
			valid: true,
		},
		{
			name:  "nested body contents stay opaque",
			yaml:  "apiVersion: llm-d-perf-simulator/v1\nkind: RunResult\nresult:\n  summary:\n    requests: 10\n",
			json:  `{"apiVersion":"llm-d-perf-simulator/v1","kind":"RunResult","result":{"summary":{"requests":10}}}`,
			valid: true,
		},
		{
			name:  "undefined kind",
			yaml:  "apiVersion: llm-d-perf-simulator/v1\nkind: Nope\nspec: {}\n",
			json:  `{"apiVersion":"llm-d-perf-simulator/v1","kind":"Nope","spec":{}}`,
			valid: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			yamlErr := ValidateDocument([]byte(tc.yaml))
			jsonErr := ValidateDocument([]byte(tc.json))
			if (yamlErr == nil) != tc.valid {
				t.Errorf("YAML verdict = %v, want valid=%v", yamlErr, tc.valid)
			}
			if (jsonErr == nil) != tc.valid {
				t.Errorf("JSON verdict = %v, want valid=%v", jsonErr, tc.valid)
			}
		})
	}
}

// TestCheckerRefusesAnUnimplementedKeyword is the reason this hand-rolled checker is safe to
// rely on: an unknown keyword is a reported schema defect, never a silent pass. Without this
// behavior a future `minimum` or `pattern` in the schema would be ignored and documents that
// violate it would be reported valid (R1).
func TestCheckerRefusesAnUnimplementedKeyword(t *testing.T) {
	schema := map[string]any{"type": "object", "minProperties": 1}
	err := validateAgainstSchema(schema, map[string]any{})
	if err == nil {
		t.Fatal("a schema using an unimplemented keyword validated successfully")
	}
	for _, want := range []string{"minProperties", "not implemented"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("diagnostic does not mention %q: %v", want, err)
		}
	}
}

// TestCheckerReportsMalformedSchemaShapes keeps the defect path honest for the other ways a
// schema can be unusable, rather than letting a wrong-typed keyword read as a valid document.
func TestCheckerReportsMalformedSchemaShapes(t *testing.T) {
	cases := []struct {
		name   string
		schema map[string]any
	}{
		{"type is not a string", map[string]any{"type": 3}},
		{"enum is not an array", map[string]any{"enum": "Run"}},
		{"required is not an array of strings", map[string]any{"required": []any{3}}},
		{"properties is not an object", map[string]any{"properties": "nope"}},
		{"a property schema is not an object", map[string]any{"properties": map[string]any{"kind": "string"}}},
		{"additionalProperties is a number", map[string]any{"additionalProperties": 1}},
		{"items is not an object", map[string]any{"items": "string"}},
		{"oneOf is not an array", map[string]any{"oneOf": map[string]any{}}},
		{"oneOf is empty", map[string]any{"oneOf": []any{}}},
		{"a oneOf branch is not an object", map[string]any{"oneOf": []any{"nope"}}},
		{"not is not an object", map[string]any{"not": "nope"}},
	}
	document := map[string]any{"kind": "Run"}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateAgainstSchema(tc.schema, document)
			if err == nil {
				t.Fatalf("a malformed schema validated the document successfully: %v", tc.schema)
			}
			if !strings.Contains(err.Error(), "cannot be applied") {
				t.Errorf("a schema defect was reported as a document problem: %v", err)
			}
		})
	}
}

// TestCheckerKeywordSemantics exercises the keywords the envelope does not currently use in
// every shape (items, additionalProperties-as-schema, number handling), so the checker's
// behavior is pinned before a later PR's spec sections rely on it.
func TestCheckerKeywordSemantics(t *testing.T) {
	cases := []struct {
		name   string
		schema map[string]any
		value  any
		valid  bool
	}{
		{
			name:   "array items are validated",
			schema: map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
			value:  []any{"a", "b"},
			valid:  true,
		},
		{
			name:   "a wrongly typed item is caught",
			schema: map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
			value:  []any{"a", 2},
			valid:  false,
		},
		{
			name:   "additionalProperties as a schema constrains extra keys",
			schema: map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "integer"}},
			value:  map[string]any{"anything": 3},
			valid:  true,
		},
		{
			name:   "additionalProperties as a schema rejects a bad extra key",
			schema: map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "integer"}},
			value:  map[string]any{"anything": "three"},
			valid:  false,
		},
		{
			name:   "an integral float satisfies integer",
			schema: map[string]any{"type": "integer"},
			value:  float64(7),
			valid:  true,
		},
		{
			name:   "a fractional float does not satisfy integer",
			schema: map[string]any{"type": "integer"},
			value:  7.5,
			valid:  false,
		},
		{
			name:   "an int satisfies number",
			schema: map[string]any{"type": "number"},
			value:  7,
			valid:  true,
		},
		{
			name:   "const compares numbers across encodings",
			schema: map[string]any{"const": 1},
			value:  float64(1),
			valid:  true,
		},
		{
			name:   "null is its own type",
			schema: map[string]any{"type": "null"},
			value:  nil,
			valid:  true,
		},
		{
			name:   "not inverts its subschema",
			schema: map[string]any{"not": map[string]any{"required": []string{"spec"}}},
			value:  map[string]any{"kind": "Run"},
			valid:  true,
		},
		{
			name:   "not rejects a value that satisfies its subschema",
			schema: map[string]any{"not": map[string]any{"required": []string{"spec"}}},
			value:  map[string]any{"spec": map[string]any{}},
			valid:  false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateAgainstSchema(tc.schema, tc.value)
			if (err == nil) != tc.valid {
				t.Errorf("verdict = %v, want valid=%v", err, tc.valid)
			}
			if err != nil && strings.Contains(err.Error(), "cannot be applied") {
				t.Errorf("the schema was reported as defective rather than applied: %v", err)
			}
		})
	}
}

// TestValidateDocumentReportsEveryProblem: a document with several independent mistakes must
// name all of them, deterministically (INV-6), so a caller is not walked through one fix per
// round trip.
func TestValidateDocumentReportsEveryProblem(t *testing.T) {
	document := []byte("apiVersion: nope\nkind: Simulate\nmetadata: {}\n")
	err := ValidateDocument(document)
	if err == nil {
		t.Fatal("a document with three independent problems was accepted")
	}
	for _, want := range []string{"apiVersion", "kind", "metadata"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("diagnostic does not mention %s: %v", want, err)
		}
	}
	for i := 0; i < 10; i++ {
		again := ValidateDocument(document)
		if again == nil || again.Error() != err.Error() {
			t.Fatalf("diagnostic is not deterministic:\n%v\nvs\n%v", err, again)
		}
	}
}

// TestValidateDocumentRejectsUnparseableInput: a parse failure must be reported as such, not
// swallowed into "valid".
func TestValidateDocumentRejectsUnparseableInput(t *testing.T) {
	for _, document := range []string{"apiVersion: [unclosed\n", "\tkind: Run\n", "a: 1\n b: 2\n"} {
		if err := ValidateDocument([]byte(document)); err == nil {
			t.Errorf("unparseable input was accepted: %q", document)
		}
	}
}

// TestNormalizeDecodedRefusesNonStringKeys: YAML allows a non-string object key, JSON does
// not, and the published contract is JSON Schema. Such a document must be refused rather
// than validated as if the key were absent.
func TestNormalizeDecodedRefusesNonStringKeys(t *testing.T) {
	if _, err := normalizeDecoded(map[any]any{3: "x"}, "$"); err == nil {
		t.Error("an integer object key was accepted")
	}
	if _, err := normalizeDecoded(map[any]any{"kind": map[any]any{true: "x"}}, "$"); err == nil {
		t.Error("a boolean object key nested in a body was accepted")
	}
	normalized, err := normalizeDecoded(map[any]any{"kind": "Run", "list": []any{map[any]any{"a": 1}}}, "$")
	if err != nil {
		t.Fatalf("a string-keyed document was refused: %v", err)
	}
	object, ok := normalized.(map[string]any)
	if !ok {
		t.Fatalf("normalized value is %T, want map[string]any", normalized)
	}
	if object["kind"] != "Run" {
		t.Errorf("normalization lost a value: %v", object)
	}
}

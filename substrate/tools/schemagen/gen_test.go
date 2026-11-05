package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/schema"
)

const root = "../.."

// TestPublishedSchemasMatchTheTypes is SUB-19's generator test: each committed schema
// equals what the generator writes from the Go types, and the directory holds no schema
// the generator no longer writes.
func TestPublishedSchemasMatchTheTypes(t *testing.T) {
	files, err := generate(root)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]bool{}
	for _, f := range files {
		want[filepath.Base(f.path)] = true
		got, err := os.ReadFile(filepath.Join(root, f.path))
		if err != nil {
			t.Errorf("%s is not committed: run `go run ./tools/schemagen`", f.path)
			continue
		}
		if !bytes.Equal(got, f.content) {
			t.Errorf("%s differs from the Go types: run `go run ./tools/schemagen`", f.path)
		}
	}
	committed, _ := filepath.Glob(filepath.Join(root, outDir, "*.schema.json"))
	for _, c := range committed {
		if !want[filepath.Base(c)] {
			t.Errorf("%s is committed but the generator does not write it", c)
		}
	}
	// Every published document has an embedded schema, and only those.
	for _, d := range documents {
		if schema.Documents[d.name] != d.file {
			t.Errorf("contracts/schema.Documents[%q] = %q, want %q", d.name, schema.Documents[d.name], d.file)
		}
		if _, err := schema.Published(d.name); err != nil {
			t.Error(err)
		}
	}
	if len(schema.Documents) != len(documents) {
		t.Errorf("contracts/schema lists %d documents, the generator %d", len(schema.Documents), len(documents))
	}
}

// TestExamplesConformToTheSchemas: each documented example document conforms to its
// published schema, and a copy with one unknown key does not.
func TestExamplesConformToTheSchemas(t *testing.T) {
	ex := func(name string) string { return filepath.Join(root, "docs", "contracts", "examples", name) }
	cases := []struct{ document, path string }{
		{"result", ex("result.json")},
		{"point-record", ex("point-record.json")},
		{"workload", ex("workload-spec.yaml")},
		{"trace", ex("trace-header.yaml")},
		{"scenario", ex("scenario.yaml")},
		{"scenario", ex("trace-binding.yaml")},
		{"scenario", ex("replay-binding.yaml")},
		{"deployment", ex("deployment.yaml")},
		{"deployment", ex("deployment-disaggregated.yaml")},
	}
	for _, c := range cases {
		t.Run(filepath.Base(c.path), func(t *testing.T) {
			s := loadSchema(t, c.document)
			doc := loadDoc(t, c.path)
			if errs := conform(s, s, doc, ""); len(errs) > 0 {
				t.Errorf("%s does not conform to the %s schema:\n%s", c.path, c.document, strings.Join(errs, "\n"))
			}
			doc.(map[string]any)["not_a_key"] = 1
			if errs := conform(s, s, doc, ""); len(errs) == 0 {
				t.Errorf("the %s schema accepted an unknown top-level key", c.document)
			}
		})
	}
}

// TestSchemaRefusesASeventhFate: a vocabulary the Go code closes is closed in the schema.
func TestSchemaRefusesASeventhFate(t *testing.T) {
	s := loadSchema(t, "result")
	doc := loadDoc(t, filepath.Join(root, "docs", "contracts", "examples", "result.json"))
	rows := doc.(map[string]any)["request_rows"].([]any)
	rows[0].(map[string]any)["fate"] = "length_capped"
	if errs := conform(s, s, doc, ""); len(errs) == 0 {
		t.Error("the result schema accepted a fate outside design §8.7")
	}
}

func loadSchema(t *testing.T, document string) map[string]any {
	t.Helper()
	b, err := schema.Published(document)
	if err != nil {
		t.Fatal(err)
	}
	var s map[string]any
	if err := json.Unmarshal(b, &s); err != nil {
		t.Fatal(err)
	}
	return s
}

// loadDoc reads YAML or JSON into the values JSON decoding gives.
func loadDoc(t *testing.T, path string) any {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var v any
	if err := yaml.Unmarshal(b, &v); err != nil {
		t.Fatal(err)
	}
	j, err := json.Marshal(normalize(v))
	if err != nil {
		t.Fatal(err)
	}
	var out any
	if err := json.Unmarshal(j, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func normalize(v any) any {
	switch x := v.(type) {
	case map[string]any:
		for k, e := range x {
			x[k] = normalize(e)
		}
	case []any:
		for i, e := range x {
			x[i] = normalize(e)
		}
	}
	return v
}

// conform checks v against s for the keywords the generator emits. It is a test helper,
// not a general validator: the published schema is for editors and other tools.
func conform(rootSchema, s map[string]any, v any, at string) []string {
	if ref, ok := s["$ref"].(string); ok {
		name := strings.TrimPrefix(ref, "#/$defs/")
		return conform(rootSchema, rootSchema["$defs"].(map[string]any)[name].(map[string]any), v, at)
	}
	var errs []string
	fail := func(format string, args ...any) { errs = append(errs, at+": "+fmt.Sprintf(format, args...)) }
	if c, ok := s["const"]; ok && fmt.Sprint(c) != fmt.Sprint(v) {
		fail("is %v, want %v", v, c)
	}
	if e, ok := s["enum"].([]any); ok && !contains(e, v) {
		fail("%v is not one of %v", v, e)
	}
	if one, ok := s["oneOf"].([]any); ok {
		n := 0
		for _, alt := range one {
			if len(conform(rootSchema, alt.(map[string]any), v, at)) == 0 {
				n++
			}
		}
		if n != 1 {
			fail("matches %d alternatives of oneOf", n)
		}
	}
	switch s["type"] {
	case "object":
		m, ok := v.(map[string]any)
		if !ok {
			fail("is not an object")
			return errs
		}
		props, _ := s["properties"].(map[string]any)
		req, _ := s["required"].([]any)
		for _, r := range req {
			if _, ok := m[r.(string)]; !ok {
				fail("required key %q is missing", r)
			}
		}
		keys := make([]string, 0, len(m))
		for k := range m {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			if pn, ok := s["propertyNames"].(map[string]any); ok {
				errs = append(errs, conform(rootSchema, pn, k, at+"."+k)...)
			}
			if ps, ok := props[k]; ok {
				errs = append(errs, conform(rootSchema, ps.(map[string]any), m[k], at+"."+k)...)
				continue
			}
			if pp, ok := s["patternProperties"].(map[string]any); ok {
				matched := false
				for re := range pp {
					matched = matched || regexp.MustCompile(re).MatchString(k)
				}
				if matched {
					continue
				}
			}
			switch ap := s["additionalProperties"].(type) {
			case bool:
				if !ap {
					fail("key %q is not allowed", k)
				}
			case map[string]any:
				errs = append(errs, conform(rootSchema, ap, m[k], at+"."+k)...)
			}
		}
	case "array":
		a, ok := v.([]any)
		if !ok {
			fail("is not an array")
			return errs
		}
		if it, ok := s["items"].(map[string]any); ok {
			for i, e := range a {
				errs = append(errs, conform(rootSchema, it, e, fmt.Sprintf("%s[%d]", at, i))...)
			}
		}
	case "string":
		if _, ok := v.(string); !ok {
			fail("is not a string")
		}
	case "integer":
		if f, ok := v.(float64); !ok || f != float64(int64(f)) {
			fail("is not an integer")
		}
	case "number":
		if _, ok := v.(float64); !ok {
			fail("is not a number")
		}
	case "boolean":
		if _, ok := v.(bool); !ok {
			fail("is not a boolean")
		}
	}
	return errs
}

func contains(e []any, v any) bool {
	for _, x := range e {
		if fmt.Sprint(x) == fmt.Sprint(v) {
			return true
		}
	}
	return false
}

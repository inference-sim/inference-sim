package api

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strings"

	"gopkg.in/yaml.v3"
)

// ValidateDocument parses data as YAML — a superset of JSON, so either encoding of the same
// document validates identically — and checks it against the committed JSON Schema
// ([SchemaJSON]).
//
// It is the external contract applied from inside Go: a document this accepts is a document
// a third-party JSON Schema validator accepts, and the example corpus under examples/ is
// checked both ways in the tests.
func ValidateDocument(data []byte) error {
	var raw any
	if err := yaml.Unmarshal(data, &raw); err != nil {
		return fmt.Errorf("parsing the document: %w", err)
	}
	value, err := normalizeDecoded(raw, "$")
	if err != nil {
		return err
	}
	return ValidateValue(value)
}

// ValidateValue checks an already-decoded document (the result of unmarshaling into `any`)
// against the committed JSON Schema.
func ValidateValue(value any) error {
	var schema map[string]any
	if err := json.Unmarshal(committedSchema, &schema); err != nil {
		return fmt.Errorf("parsing the committed schema %s: %w", SchemaPath, err)
	}
	return validateAgainstSchema(schema, value)
}

// validateAgainstSchema reports every way value violates schema, in a deterministic order.
//
// It implements exactly the JSON Schema vocabulary [GenerateSchema] emits, and REFUSES a
// schema that uses anything else (see checker.check). That refusal is the point: a
// validator that skipped keywords it did not understand would report a document as valid
// because it never checked the constraint that rejects it — a silent pass, which is the R1
// failure this repository treats as a bug rather than a limitation. A later PR that emits a
// new keyword must teach this validator the keyword in the same change.
func validateAgainstSchema(schema map[string]any, value any) error {
	c := &checker{}
	problems := c.check(schema, value, "$")
	if c.defect != nil {
		return fmt.Errorf("the committed schema %s cannot be applied: %w", SchemaPath, c.defect)
	}
	if len(problems) == 0 {
		return nil
	}
	return fmt.Errorf("document does not satisfy %s: %s", SchemaID, strings.Join(problems, "; "))
}

// checker walks a schema against a value. A schema defect (an unimplemented keyword, a
// malformed schema) is latched in defect and stops further reporting, so a defect is never
// mistaken for a document problem.
type checker struct {
	defect error
}

// annotationKeywords carry no constraint; they are read by humans and tools, not by the
// validator.
var annotationKeywords = []string{"$schema", "$id", "title", "description", "$comment", "examples", "default"}

// assertionKeywords are the keywords this validator implements. Kept beside
// annotationKeywords so the "do I understand this schema?" check is one lookup over two
// explicit lists rather than a default-allow switch.
var assertionKeywords = []string{"type", "const", "enum", "properties", "required", "additionalProperties", "items", "oneOf", "not"}

func (c *checker) check(schema map[string]any, value any, path string) []string {
	if c.defect != nil {
		return nil
	}
	for _, keyword := range sortedKeys(schema) {
		if !slices.Contains(annotationKeywords, keyword) && !slices.Contains(assertionKeywords, keyword) {
			c.defect = fmt.Errorf("at %s: keyword %q is not implemented by this validator", path, keyword)
			return nil
		}
	}

	var problems []string
	// Keyword order is fixed (not map order) so a document with several problems reports
	// them identically on every run (INV-6).
	problems = append(problems, c.checkType(schema, value, path)...)
	problems = append(problems, c.checkConst(schema, value, path)...)
	problems = append(problems, c.checkEnum(schema, value, path)...)
	problems = append(problems, c.checkObject(schema, value, path)...)
	problems = append(problems, c.checkItems(schema, value, path)...)
	problems = append(problems, c.checkOneOf(schema, value, path)...)
	problems = append(problems, c.checkNot(schema, value, path)...)
	if c.defect != nil {
		return nil
	}
	return problems
}

func (c *checker) checkType(schema map[string]any, value any, path string) []string {
	raw, ok := schema["type"]
	if !ok {
		return nil
	}
	want, ok := raw.(string)
	if !ok {
		c.defect = fmt.Errorf("at %s: \"type\" must be a string, got %T", path, raw)
		return nil
	}
	if matchesType(want, value) {
		return nil
	}
	return []string{fmt.Sprintf("%s: expected %s, got %s", path, want, describe(value))}
}

func (c *checker) checkConst(schema map[string]any, value any, path string) []string {
	want, ok := schema["const"]
	if !ok || equalValues(want, value) {
		return nil
	}
	return []string{fmt.Sprintf("%s: expected the constant %s, got %s", path, render(want), render(value))}
}

func (c *checker) checkEnum(schema map[string]any, value any, path string) []string {
	raw, ok := schema["enum"]
	if !ok {
		return nil
	}
	allowed, ok := asSlice(raw)
	if !ok {
		c.defect = fmt.Errorf("at %s: \"enum\" must be an array, got %T", path, raw)
		return nil
	}
	for _, candidate := range allowed {
		if equalValues(candidate, value) {
			return nil
		}
	}
	rendered := make([]string, len(allowed))
	for i, candidate := range allowed {
		rendered[i] = render(candidate)
	}
	return []string{fmt.Sprintf("%s: %s is not one of %s", path, render(value), strings.Join(rendered, ", "))}
}

// checkObject applies properties, required and additionalProperties. All three are inert
// for a non-object value: the type keyword is what reports "expected object", and piling a
// second complaint on the same mistake only makes the message harder to read.
func (c *checker) checkObject(schema map[string]any, value any, path string) []string {
	_, hasProperties := schema["properties"]
	_, hasRequired := schema["required"]
	_, hasAdditional := schema["additionalProperties"]
	if !hasProperties && !hasRequired && !hasAdditional {
		return nil
	}
	object, ok := value.(map[string]any)
	if !ok {
		return nil
	}

	properties := map[string]any{}
	if hasProperties {
		properties, ok = schema["properties"].(map[string]any)
		if !ok {
			c.defect = fmt.Errorf("at %s: \"properties\" must be an object, got %T", path, schema["properties"])
			return nil
		}
	}

	var problems []string
	if hasRequired {
		names, ok := asStrings(schema["required"])
		if !ok {
			c.defect = fmt.Errorf("at %s: \"required\" must be an array of strings, got %T", path, schema["required"])
			return nil
		}
		for _, name := range names {
			if _, present := object[name]; !present {
				problems = append(problems, fmt.Sprintf("%s: required property %q is missing", path, name))
			}
		}
	}

	// Sorted so the report order does not depend on Go's map iteration (INV-6, R2).
	for _, name := range sortedKeys(object) {
		sub, constrained := properties[name]
		if !constrained {
			problems = append(problems, c.checkAdditional(schema, hasAdditional, name, object[name], path)...)
			continue
		}
		subSchema, ok := sub.(map[string]any)
		if !ok {
			c.defect = fmt.Errorf("at %s: property %q must map to an object schema, got %T", path, name, sub)
			return nil
		}
		problems = append(problems, c.check(subSchema, object[name], path+"."+name)...)
	}
	return problems
}

// checkAdditional applies additionalProperties to one key the properties map does not
// mention: false refuses it, true (or absent) admits it, a schema validates it.
func (c *checker) checkAdditional(schema map[string]any, hasAdditional bool, name string, value any, path string) []string {
	if !hasAdditional {
		return nil
	}
	switch additional := schema["additionalProperties"].(type) {
	case bool:
		if additional {
			return nil
		}
		return []string{fmt.Sprintf("%s: unknown property %q", path, name)}
	case map[string]any:
		return c.check(additional, value, path+"."+name)
	default:
		c.defect = fmt.Errorf("at %s: \"additionalProperties\" must be a boolean or an object schema, got %T",
			path, schema["additionalProperties"])
		return nil
	}
}

func (c *checker) checkItems(schema map[string]any, value any, path string) []string {
	raw, ok := schema["items"]
	if !ok {
		return nil
	}
	itemSchema, ok := raw.(map[string]any)
	if !ok {
		c.defect = fmt.Errorf("at %s: \"items\" must be an object schema, got %T", path, raw)
		return nil
	}
	items, ok := value.([]any)
	if !ok {
		return nil // the type keyword reports a non-array
	}
	var problems []string
	for i, item := range items {
		problems = append(problems, c.check(itemSchema, item, fmt.Sprintf("%s[%d]", path, i))...)
	}
	return problems
}

// checkOneOf requires exactly one branch to match. When none does, every branch's reason is
// reported: with the envelope's input/output partition, "matched no branch" alone would not
// tell the reader that a `kind: Run` document is missing its spec.
func (c *checker) checkOneOf(schema map[string]any, value any, path string) []string {
	raw, ok := schema["oneOf"]
	if !ok {
		return nil
	}
	branches, ok := asSlice(raw)
	if !ok {
		c.defect = fmt.Errorf("at %s: \"oneOf\" must be an array, got %T", path, raw)
		return nil
	}
	if len(branches) == 0 {
		c.defect = fmt.Errorf("at %s: \"oneOf\" is empty, so nothing can satisfy it", path)
		return nil
	}

	matched := 0
	var reasons []string
	for i, branch := range branches {
		branchSchema, ok := branch.(map[string]any)
		if !ok {
			c.defect = fmt.Errorf("at %s: oneOf[%d] must be an object schema, got %T", path, i, branch)
			return nil
		}
		problems := c.check(branchSchema, value, path)
		if c.defect != nil {
			return nil
		}
		if len(problems) == 0 {
			matched++
			continue
		}
		reasons = append(reasons, fmt.Sprintf("alternative %d: %s", i+1, strings.Join(problems, ", ")))
	}
	switch {
	case matched == 1:
		return nil
	case matched == 0:
		return []string{fmt.Sprintf("%s: matches none of the %d alternatives (%s)",
			path, len(branches), strings.Join(reasons, "; "))}
	default:
		return []string{fmt.Sprintf("%s: matches %d of the %d alternatives, want exactly one",
			path, matched, len(branches))}
	}
}

// checkNot inverts a subschema: the value must NOT satisfy it. The envelope uses it for
// body exclusivity — an input document must not also carry a result — so the message names
// the forbidden shape rather than echoing the inner schema.
func (c *checker) checkNot(schema map[string]any, value any, path string) []string {
	raw, ok := schema["not"]
	if !ok {
		return nil
	}
	inner, ok := raw.(map[string]any)
	if !ok {
		c.defect = fmt.Errorf("at %s: \"not\" must be an object schema, got %T", path, raw)
		return nil
	}
	if len(c.check(inner, value, path)) > 0 || c.defect != nil {
		return nil // it violates the inner schema, which is exactly what "not" demands
	}
	if forbidden, ok := asStrings(inner["required"]); ok {
		return []string{fmt.Sprintf("%s: must not carry %s", path, strings.Join(forbidden, ", "))}
	}
	return []string{fmt.Sprintf("%s: must not satisfy the \"not\" subschema", path)}
}

// matchesType reports whether value has the JSON type named by want. "integer" accepts a
// float with an integral value, because a JSON document round-tripped through encoding/json
// decodes 3 as float64(3) while the same YAML decodes it as int.
func matchesType(want string, value any) bool {
	switch want {
	case "object":
		_, ok := value.(map[string]any)
		return ok
	case "array":
		_, ok := value.([]any)
		return ok
	case "string":
		_, ok := value.(string)
		return ok
	case "boolean":
		_, ok := value.(bool)
		return ok
	case "integer":
		f, ok := asNumber(value)
		return ok && f == math.Trunc(f) && !math.IsInf(f, 0)
	case "number":
		_, ok := asNumber(value)
		return ok
	case "null":
		return value == nil
	default:
		return false
	}
}

// equalValues compares a schema literal with a document value. Numbers are compared
// numerically so an int from YAML equals a float64 from JSON.
func equalValues(a, b any) bool {
	if af, ok := asNumber(a); ok {
		bf, ok := asNumber(b)
		return ok && af == bf
	}
	return reflect.DeepEqual(a, b)
}

func asNumber(value any) (float64, bool) {
	switch n := value.(type) {
	case int:
		return float64(n), true
	case int32:
		return float64(n), true
	case int64:
		return float64(n), true
	case uint64:
		return float64(n), true
	case float32:
		return float64(n), true
	case float64:
		return n, true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}

func asSlice(value any) ([]any, bool) {
	items, ok := value.([]any)
	return items, ok
}

// asStrings accepts both []any (a schema parsed from JSON) and []string (a schema built in
// Go by GenerateSchema), so the same checker validates a freshly derived schema.
func asStrings(value any) ([]string, bool) {
	switch items := value.(type) {
	case []string:
		return items, true
	case []any:
		out := make([]string, 0, len(items))
		for _, item := range items {
			s, ok := item.(string)
			if !ok {
				return nil, false
			}
			out = append(out, s)
		}
		return out, true
	default:
		return nil, false
	}
}

func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}

// describe names a value's JSON type for a diagnostic.
func describe(value any) string {
	switch value.(type) {
	case nil:
		return "null"
	case map[string]any:
		return "object"
	case []any:
		return "array"
	case string:
		return "string"
	case bool:
		return "boolean"
	default:
		if f, ok := asNumber(value); ok && f == math.Trunc(f) {
			return "integer"
		}
		if _, ok := asNumber(value); ok {
			return "number"
		}
		return fmt.Sprintf("%T", value)
	}
}

// render prints a scalar for a diagnostic, quoting strings so an empty or space-padded
// value is visible.
func render(value any) string {
	if s, ok := value.(string); ok {
		return fmt.Sprintf("%q", s)
	}
	return fmt.Sprintf("%v", value)
}

// normalizeDecoded converts a YAML-decoded value into the JSON data model: object keys
// must be strings, because that is the only key a JSON document (and therefore the
// published schema) can express. A non-string key is refused naming its path rather than
// dropped, so a YAML-only document cannot validate as if it were JSON.
func normalizeDecoded(value any, path string) (any, error) {
	switch typed := value.(type) {
	case map[string]any:
		out := make(map[string]any, len(typed))
		for key, item := range typed {
			normalized, err := normalizeDecoded(item, path+"."+key)
			if err != nil {
				return nil, err
			}
			out[key] = normalized
		}
		return out, nil
	case map[any]any:
		out := make(map[string]any, len(typed))
		for key, item := range typed {
			name, ok := key.(string)
			if !ok {
				return nil, fmt.Errorf("at %s: object key %v is a %T, but a document key must be a string", path, key, key)
			}
			normalized, err := normalizeDecoded(item, path+"."+name)
			if err != nil {
				return nil, err
			}
			out[name] = normalized
		}
		return out, nil
	case []any:
		out := make([]any, len(typed))
		for i, item := range typed {
			normalized, err := normalizeDecoded(item, fmt.Sprintf("%s[%d]", path, i))
			if err != nil {
				return nil, err
			}
			out[i] = normalized
		}
		return out, nil
	default:
		return value, nil
	}
}

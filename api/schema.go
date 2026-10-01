package api

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
)

// SchemaDraft is the JSON Schema dialect the generated schema declares.
const SchemaDraft = "https://json-schema.org/draft/2020-12/schema"

// SchemaID is the canonical identifier of the v1 envelope schema. External tools resolve
// the contract by this id; it changes only when the apiVersion does.
const SchemaID = "https://github.com/inference-sim/inference-sim/raw/main/api/schema/llm-d-perf-simulator-v1.json"

// SchemaPath is the committed schema's path relative to this package's directory. The
// go:generate directive in doc.go writes it and the embed below reads it, so the two
// cannot drift apart.
const SchemaPath = "schema/llm-d-perf-simulator-v1.json"

//go:embed schema/llm-d-perf-simulator-v1.json
var committedSchema []byte

// SchemaJSON returns the committed JSON Schema for the v1 envelope. The returned slice is
// a copy: the schema is embedded once, and a caller that mutated it would corrupt every
// later validation in the process.
func SchemaJSON() []byte { return slices.Clone(committedSchema) }

// schemaConstrainer is implemented by an envelope type that narrows its generated schema
// beyond what its Go kind implies — a const for APIVersion, an enum for Kind, an open
// object for Body. Deriving these from the types (rather than hand-writing them in the
// generator) is what keeps the schema honest: adding a kind updates the enum with no
// generator change, because the enum is read back off the type.
//
// Implement it with a VALUE receiver; the generator looks the method up on the zero value.
type schemaConstrainer interface {
	schemaConstraints() map[string]any
}

// GenerateSchema derives the JSON Schema of the envelope from [Document] and returns it as
// the exact bytes the committed file holds — indented with two spaces and newline
// terminated, with every object key and array element in a deterministic order (INV-6), so
// a regeneration that changed nothing produces a zero diff.
func GenerateSchema() ([]byte, error) {
	root, err := schemaForType(reflect.TypeOf(Document{}))
	if err != nil {
		return nil, fmt.Errorf("deriving the schema of %T: %w", Document{}, err)
	}
	root["$schema"] = SchemaDraft
	root["$id"] = SchemaID
	root["title"] = fmt.Sprintf("BLIS declarative document (%s)", Version)
	root["description"] = "The envelope shared by every BLIS input and output document. " +
		"API-1 constrains envelope structure only: the spec and result bodies are opaque " +
		"open objects until the per-verb sections land (epic #1855)."
	root["oneOf"] = kindPartition()

	out, err := json.MarshalIndent(root, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("marshaling the derived schema: %w", err)
	}
	return append(out, '\n'), nil
}

// kindPartition expresses "an input kind carries a spec, an output kind carries a result"
// as a two-branch oneOf. It is derived from the kind lists, so a kind added to inputKinds
// lands in both the enum and the partition.
//
// A document matches exactly one branch: the branches' kind enums are disjoint, so a
// document whose kind is an input kind but which carries only a result matches neither and
// is refused.
func kindPartition() []any {
	return []any{
		map[string]any{
			"description": "An input document carries a spec.",
			"properties": map[string]any{
				"kind": map[string]any{"enum": kindStrings(InputKinds())},
			},
			"required": []string{"spec"},
		},
		map[string]any{
			"description": "An output document carries a result.",
			"properties": map[string]any{
				"kind": map[string]any{"enum": kindStrings(OutputKinds())},
			},
			"required": []string{"result"},
		},
	}
}

// schemaForType derives the schema of a single Go type. It handles exactly the type shapes
// the envelope uses and returns an error for anything else: a silently-empty schema for an
// unhandled type would publish a contract that constrains nothing (R1), so a later PR that
// adds, say, a time.Time field is told to teach the generator about it instead.
func schemaForType(t reflect.Type) (map[string]any, error) {
	if c, ok := reflect.Zero(t).Interface().(schemaConstrainer); ok {
		return c.schemaConstraints(), nil
	}

	switch t.Kind() {
	case reflect.Pointer:
		// A pointer is the "zero is meaningful" encoding of its element (R9); the
		// document shape is the element's.
		return schemaForType(t.Elem())

	case reflect.Struct:
		return schemaForStruct(t)

	case reflect.String:
		return map[string]any{"type": "string"}, nil

	case reflect.Bool:
		return map[string]any{"type": "boolean"}, nil

	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}, nil

	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}, nil

	case reflect.Slice, reflect.Array:
		items, err := schemaForType(t.Elem())
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "array", "items": items}, nil

	case reflect.Map:
		if t.Key().Kind() != reflect.String {
			return nil, fmt.Errorf("map type %s has a non-string key: a JSON object key is always a string", t)
		}
		values, err := schemaForType(t.Elem())
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "object", "additionalProperties": values}, nil

	case reflect.Interface:
		if t.NumMethod() == 0 {
			return map[string]any{}, nil // `any`: unconstrained by construction
		}
		return nil, fmt.Errorf("interface type %s has methods, so its document shape is unknown", t)

	default:
		return nil, fmt.Errorf("type %s has no JSON Schema translation; teach schemaForType about it", t)
	}
}

// schemaForStruct derives an object schema from a struct's json tags. A field is REQUIRED
// unless its tag carries omitempty, which is the same rule encoding/json applies when
// writing the document — so the schema's required list is the set of fields a marshaled
// document always contains.
//
// additionalProperties is false: an unknown key in a document is a typo, and accepting it
// silently is the YAML-typo failure R10 exists to prevent.
func schemaForStruct(t reflect.Type) (map[string]any, error) {
	properties := map[string]any{}
	var required []string

	for i := range t.NumField() {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}
		name, optional, ok := jsonFieldName(field)
		if !ok {
			continue
		}
		fieldSchema, err := schemaForType(field.Type)
		if err != nil {
			return nil, fmt.Errorf("field %s.%s: %w", t.Name(), field.Name, err)
		}
		properties[name] = fieldSchema
		if !optional {
			required = append(required, name)
		}
	}

	slices.Sort(required)
	schema := map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"properties":           properties,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema, nil
}

// jsonFieldName reports the document key a struct field marshals to, whether it is
// optional (omitempty), and whether it is serialized at all (a `json:"-"` field is not).
func jsonFieldName(field reflect.StructField) (name string, optional, serialized bool) {
	tag, ok := field.Tag.Lookup("json")
	if !ok {
		// No tag: encoding/json uses the Go field name verbatim.
		return field.Name, false, true
	}
	parts := strings.Split(tag, ",")
	name = parts[0]
	if name == "-" && len(parts) == 1 {
		return "", false, false
	}
	if name == "" {
		name = field.Name
	}
	return name, slices.Contains(parts[1:], "omitempty"), true
}

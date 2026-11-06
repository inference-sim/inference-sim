package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/metrics"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/pointrecord"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	"github.com/inference-sim/blis-substrate/contracts/spec/trace"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

// outDir is where the schemas are committed, relative to the repository root.
const outDir = "contracts/schema"

const modulePath = "github.com/inference-sim/blis-substrate/"

// document is one published schema: its root type and the title and description it
// carries.
type document struct {
	name, file string
	root       reflect.Type
	// extra are further types the document defines at its top level, such as a trace's
	// record, which lives in a CSV rather than under the header.
	extra       []reflect.Type
	description string
}

var documents = []document{
	{name: "scenario", file: "scenario.schema.json", root: reflect.TypeOf(scenario.Scenario{}),
		description: "A BLIS scenario: the model, cluster, traffic, coefficient sets and engine a run is given."},
	{name: "deployment", file: "deployment.schema.json", root: reflect.TypeOf(deployment.Deployment{}),
		description: "A BLIS deployment: the pools, offload and transfer configuration applied to a scenario."},
	{name: "result", file: "result.schema.json", root: reflect.TypeOf(result.Result{}),
		description: "A BLIS Result: what one run predicts, with the engine, data, options, frame and fates it records."},
	{name: "point-record", file: "point-record.schema.json", root: reflect.TypeOf(pointrecord.Record{}),
		description: "The scoring harness's per-point record: identity, grade, substitutions, builds and paired figures."},
	{name: "workload", file: "workload.schema.json", root: reflect.TypeOf(workload.Spec{}),
		description: "The workload generator specification blis run --workload-spec reads and blis convert writes."},
	{name: "trace", file: "trace.schema.json", root: reflect.TypeOf(trace.Header{}),
		extra:       []reflect.Type{reflect.TypeOf(trace.Record{})},
		description: "A BLIS trace header. The data CSV's row is defined as $defs.Record; its columns are the record's keys."},
}

// special are types whose wire form is not their Go structure: each has its own
// marshaller, and the schema states the form the marshaller reads.
var special = map[reflect.Type]map[string]any{
	// A pool's engine block is the engine's, held unopened.
	reflect.TypeOf(deployment.EngineBlock{}): {"type": "object",
		"description": "The pool's engine settings, in the engine's own names; validated by the engine, not by this schema."},
	// An interval is a two-element sequence.
	reflect.TypeOf(coefficient.Interval{}): {"type": "array", "items": map[string]any{"type": "number"}, "minItems": 2, "maxItems": 2},
	// A context limit is a positive count or max_model_len.
	reflect.TypeOf(workload.ContextLimit(0)): {"oneOf": []any{
		map[string]any{"type": "integer", "minimum": 1}, map[string]any{"const": "max_model_len"}}},
}

// fieldOverrides replace or constrain one field's schema: the constants a document fixes,
// and the metric identifiers the registry closes (DEV-8).
var fieldOverrides = map[string]func(any) any{
	"result.Result.kind":                constant(result.Kind),
	"result.Result.schema_version":      constant(result.SchemaVersion),
	"pointrecord.Record.kind":           constant(pointrecord.Kind),
	"pointrecord.Record.schema_version": constant(pointrecord.SchemaVersion),
	"trace.Header.trace_version":        constant(trace.Version),
	"workload.Spec.version":             constant(workload.SpecVersion),
	"scenario.Scenario.kind":            constant("Scenario"),
	"deployment.Deployment.kind":        constant("Deployment"),
	"result.Result.metrics":             func(s any) any { return withKeys(s, metrics.IDs()) },
	"result.Frame.metrics":              func(any) any { return map[string]any{"type": "array", "items": map[string]any{"enum": metrics.IDs()}} },
	"pointrecord.Figure.metric":         func(any) any { return map[string]any{"type": "string", "enum": metrics.IDs()} },
	"result.Breakdowns.per_model":       nested(metrics.IDs()),
	"result.Breakdowns.per_tenant":      nested(metrics.IDs()),
	"result.Breakdowns.per_adapter":     nested(metrics.IDs()),
}

func constant(v any) func(any) any { return func(any) any { return map[string]any{"const": v} } }

// withKeys limits an object schema's keys to ids.
func withKeys(s any, ids []string) any {
	m := s.(map[string]any)
	m["propertyNames"] = map[string]any{"enum": ids}
	return m
}

// nested limits the inner keys of a map of maps to ids.
func nested(ids []string) func(any) any {
	return func(s any) any {
		m := s.(map[string]any)
		m["additionalProperties"] = withKeys(m["additionalProperties"], ids)
		return m
	}
}

// commentKeys are the types whose loaders accept and discard the catalog's _comment keys.
var commentKeys = map[reflect.Type]bool{
	reflect.TypeOf(workload.Shape{}): true, reflect.TypeOf(workload.Distribution{}): true,
}

// generator builds one document's schema, collecting each named struct as a definition.
type generator struct {
	root  string
	defs  map[string]any
	enums map[string]map[string][]string // package dir -> type name -> members
	// tables are the reference pages' field tables: for each `<!-- fields: pkg.Type -->`
	// marker, each key and whether its Required cell is exactly "yes". A key is required in
	// the schema exactly when the documentation says it is; docs_test keeps each table equal
	// to its type's keys.
	tables map[string]map[string]bool
}

// generate returns every schema file, in a stable order.
func generate(root string) ([]file, error) {
	var out []file
	for _, d := range documents {
		tables, err := fieldTables(root)
		if err != nil {
			return nil, err
		}
		g := &generator{root: root, defs: map[string]any{}, enums: map[string]map[string][]string{}, tables: tables}
		top, err := g.schemaOf(d.root)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", d.name, err)
		}
		for _, t := range d.extra {
			if _, err := g.schemaOf(t); err != nil {
				return nil, fmt.Errorf("%s: %w", d.name, err)
			}
		}
		doc := map[string]any{
			"$schema":     "https://json-schema.org/draft/2020-12/schema",
			"$id":         "https://inference-sim.github.io/blis-substrate/schema/" + d.file,
			"title":       d.name,
			"description": d.description + " Generated from the Go types by tools/schemagen; the Go validators are the definition.",
			"$defs":       g.defs,
		}
		for k, v := range top.(map[string]any) {
			doc[k] = v
		}
		b, err := marshal(doc)
		if err != nil {
			return nil, err
		}
		out = append(out, file{path: outDir + "/" + d.file, content: b})
	}
	return out, nil
}

// schemaOf returns the schema of t, adding named structs to the definitions.
func (g *generator) schemaOf(t reflect.Type) (any, error) {
	if s, ok := special[t]; ok {
		return s, nil
	}
	switch t.Kind() {
	case reflect.Pointer:
		return g.schemaOf(t.Elem())
	case reflect.Struct:
		name := defName(t)
		if _, done := g.defs[name]; !done {
			g.defs[name] = nil // reserve, for recursive types
			s, err := g.structSchema(t)
			if err != nil {
				return nil, err
			}
			g.defs[name] = s
		}
		return map[string]any{"$ref": "#/$defs/" + name}, nil
	case reflect.Slice, reflect.Array:
		items, err := g.schemaOf(t.Elem())
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "array", "items": items}, nil
	case reflect.Map:
		vals, err := g.schemaOf(t.Elem())
		if err != nil {
			return nil, err
		}
		s := map[string]any{"type": "object", "additionalProperties": vals}
		if t.Key().Kind() == reflect.String && t.Key().Name() != "string" {
			if members, err := g.members(t.Key()); err != nil {
				return nil, err
			} else if len(members) > 0 {
				s["propertyNames"] = map[string]any{"enum": members}
			}
		}
		return s, nil
	case reflect.Interface:
		return map[string]any{}, nil
	case reflect.String:
		if t.Name() != "string" && t.PkgPath() != "" {
			members, err := g.members(t)
			if err != nil {
				return nil, err
			}
			if len(members) > 0 {
				return map[string]any{"type": "string", "enum": withEmpty(t, members)}, nil
			}
		}
		return map[string]any{"type": "string"}, nil
	case reflect.Bool:
		return map[string]any{"type": "boolean"}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}, nil
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}, nil
	}
	return nil, fmt.Errorf("type %s (%s) has no JSON Schema form", t, t.Kind())
}

// structSchema is a struct's object schema: one property per yaml key, the keys without
// omitempty required, and no other key allowed (SUB-12).
func (g *generator) structSchema(t reflect.Type) (map[string]any, error) {
	props := map[string]any{}
	if err := g.fields(t, props); err != nil {
		return nil, err
	}
	table, ok := g.tables[defName(t)]
	if !ok {
		return nil, fmt.Errorf("%s has no field table in docs/contracts/reference, so its required keys are unknown", defName(t))
	}
	var required []string
	for k, req := range table {
		if req {
			required = append(required, k)
		}
	}
	sort.Strings(required)
	s := map[string]any{"type": "object", "properties": props, "additionalProperties": false}
	if len(required) > 0 {
		s["required"] = required
	}
	if commentKeys[t] {
		s["patternProperties"] = map[string]any{"^_comment(_.+)?$": map[string]any{}}
	}
	return s, nil
}

func (g *generator) fields(t reflect.Type, props map[string]any) error {
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		tag := f.Tag.Get("yaml")
		if tag == "-" || !f.IsExported() {
			continue
		}
		parts := strings.Split(tag, ",")
		opts := "," + strings.Join(parts[1:], ",") + ","
		if strings.Contains(opts, ",inline,") {
			if err := g.fields(f.Type, props); err != nil {
				return err
			}
			continue
		}
		name := parts[0]
		if name == "" {
			name = strings.ToLower(f.Name)
		}
		s, err := g.schemaOf(f.Type)
		if err != nil {
			return fmt.Errorf("%s.%s: %w", t.Name(), f.Name, err)
		}
		if o, ok := fieldOverrides[defName(t)+"."+name]; ok {
			s = o(s)
		}
		props[name] = s
	}
	return nil
}

// withEmpty adds the empty string to a vocabulary whose Valid method accepts it, as an
// optional SLO class does, so the schema accepts what the validator accepts.
func withEmpty(t reflect.Type, members []string) []string {
	for _, m := range members {
		if m == "" {
			return members
		}
	}
	valid := reflect.Zero(t).MethodByName("Valid")
	if !valid.IsValid() || valid.Type().NumIn() != 0 || valid.Type().NumOut() != 1 || valid.Type().Out(0).Kind() != reflect.Bool {
		return members
	}
	if valid.Call(nil)[0].Bool() {
		return append([]string{""}, members...)
	}
	return members
}

var (
	fieldsMarker = regexp.MustCompile(`^<!-- fields: (\S+) -->$`)
	firstCode    = regexp.MustCompile("`([^`]+)`")
)

// fieldTables reads every reference page's field tables.
func fieldTables(root string) (map[string]map[string]bool, error) {
	pages, err := filepath.Glob(filepath.Join(root, "docs", "contracts", "reference", "*.md"))
	if err != nil {
		return nil, err
	}
	sort.Strings(pages)
	out := map[string]map[string]bool{}
	for _, page := range pages {
		b, err := os.ReadFile(page)
		if err != nil {
			return nil, err
		}
		current := ""
		for _, line := range strings.Split(string(b), "\n") {
			line = strings.TrimSpace(line)
			if m := fieldsMarker.FindStringSubmatch(line); m != nil {
				current = m[1]
				if out[current] == nil {
					out[current] = map[string]bool{}
				}
				continue
			}
			if current == "" {
				continue
			}
			if !strings.HasPrefix(line, "|") {
				if line != "" {
					current = ""
				}
				continue
			}
			cells := strings.Split(strings.Trim(line, "|"), "|")
			m := firstCode.FindStringSubmatch(cells[0])
			if m == nil || len(cells) < 3 {
				continue
			}
			out[current][m[1]] = strings.TrimSpace(cells[2]) == "yes"
		}
	}
	return out, nil
}

// defName is a struct's definition name: its package and type, such as result.Frame.
func defName(t reflect.Type) string {
	return filepath.Base(t.PkgPath()) + "." + t.Name()
}

// members returns the string constants declared with type t in its package's source, in
// declaration order, so the schema's enum is the vocabulary the code closes. A type with no
// such constants is an open string.
func (g *generator) members(t reflect.Type) ([]string, error) {
	rel := strings.TrimPrefix(t.PkgPath(), modulePath)
	if rel == t.PkgPath() {
		return nil, nil
	}
	consts, ok := g.enums[rel]
	if !ok {
		var err error
		if consts, err = typedStringConsts(filepath.Join(g.root, rel)); err != nil {
			return nil, err
		}
		g.enums[rel] = consts
	}
	return consts[t.Name()], nil
}

// typedStringConsts reads, for each named type in a package's non-test files, the string
// values of the constants declared with that type explicitly.
func typedStringConsts(dir string) (map[string][]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	fset := token.NewFileSet()
	out := map[string][]string{}
	var names []string
	for _, e := range entries {
		if n := e.Name(); strings.HasSuffix(n, ".go") && !strings.HasSuffix(n, "_test.go") {
			names = append(names, n)
		}
	}
	sort.Strings(names)
	for _, n := range names {
		f, err := parser.ParseFile(fset, filepath.Join(dir, n), nil, 0)
		if err != nil {
			return nil, err
		}
		for _, d := range f.Decls {
			gd, ok := d.(*ast.GenDecl)
			if !ok || gd.Tok != token.CONST {
				continue
			}
			for _, sp := range gd.Specs {
				vs := sp.(*ast.ValueSpec)
				id, ok := vs.Type.(*ast.Ident)
				if !ok || len(vs.Values) != len(vs.Names) {
					continue
				}
				for _, v := range vs.Values {
					lit, ok := v.(*ast.BasicLit)
					if !ok || lit.Kind != token.STRING {
						continue
					}
					s, err := strconv.Unquote(lit.Value)
					if err != nil {
						return nil, err
					}
					out[id.Name] = append(out[id.Name], s)
				}
			}
		}
	}
	return out, nil
}

// marshal writes JSON with sorted keys, two-space indentation and a trailing newline, so
// the output is a deterministic function of the types.
func marshal(v any) ([]byte, error) {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

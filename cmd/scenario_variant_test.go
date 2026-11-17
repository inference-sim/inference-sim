package cmd

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// A scenario file holds several YAML documents (a Scenario, a Deployment, a Router). A test that
// needs a variant of one states the difference as a typed edit of a field, found by its document
// kind and path, instead of replacing text that happens to be unique in the file (#2107 I1). An
// edit whose field is missing, or whose path runs through a field of another shape, fails the
// test at once, so a fixture that changes shape cannot make a case pass vacuously.
//
// A path segment is a mapping key, or role=<name> to pick the pool of a pools list by its role.
type scenarioEdit func(t *testing.T, docs []*yaml.Node)

// setField sets the field at path in the document of the given kind to value, a YAML scalar or
// flow value ("4000000", "[a, b]", "{k: v}"). A field the mapping lacks is added.
func setField(kind, value string, path ...string) scenarioEdit {
	return func(t *testing.T, docs []*yaml.Node) {
		t.Helper()
		parent := walkScenario(t, docs, kind, path[:len(path)-1])
		var v yaml.Node
		if err := yaml.Unmarshal([]byte(value), &v); err != nil || len(v.Content) != 1 {
			t.Fatalf("setField %v: %q is not a YAML value: %v", path, value, err)
		}
		key := path[len(path)-1]
		for i := 0; i+1 < len(parent.Content); i += 2 {
			if parent.Content[i].Value == key {
				parent.Content[i+1] = v.Content[0]
				return
			}
		}
		parent.Content = append(parent.Content, &yaml.Node{Kind: yaml.ScalarNode, Value: key}, v.Content[0])
	}
}

// removeField removes the field at path, which must exist.
func removeField(kind string, path ...string) scenarioEdit {
	return func(t *testing.T, docs []*yaml.Node) {
		t.Helper()
		parent := walkScenario(t, docs, kind, path[:len(path)-1])
		key := path[len(path)-1]
		for i := 0; i+1 < len(parent.Content); i += 2 {
			if parent.Content[i].Value == key {
				parent.Content = append(parent.Content[:i], parent.Content[i+2:]...)
				return
			}
		}
		t.Fatalf("removeField: the %s document has no %v to remove", kind, path)
	}
}

// walkScenario returns the mapping at path inside the document of the given kind.
func walkScenario(t *testing.T, docs []*yaml.Node, kind string, path []string) *yaml.Node {
	t.Helper()
	var cur *yaml.Node
	for _, d := range docs {
		m := d.Content[0]
		if mappingValue(m, "kind") != nil && mappingValue(m, "kind").Value == kind {
			cur = m
		}
	}
	if cur == nil {
		t.Fatalf("the scenario has no %s document", kind)
	}
	for _, seg := range path {
		if role, ok := strings.CutPrefix(seg, "role="); ok {
			if cur.Kind != yaml.SequenceNode {
				t.Fatalf("path segment %q: not a list", seg)
			}
			var found *yaml.Node
			for _, item := range cur.Content {
				if r := mappingValue(item, "role"); r != nil && r.Value == role {
					found = item
				}
			}
			if found == nil {
				t.Fatalf("path segment %q: no such pool", seg)
			}
			cur = found
			continue
		}
		next := mappingValue(cur, seg)
		if next == nil {
			t.Fatalf("path segment %q: no such field", seg)
		}
		cur = next
	}
	return cur
}

func mappingValue(m *yaml.Node, key string) *yaml.Node {
	if m.Kind != yaml.MappingNode {
		return nil
	}
	for i := 0; i+1 < len(m.Content); i += 2 {
		if m.Content[i].Value == key {
			return m.Content[i+1]
		}
	}
	return nil
}

// editScenarioText applies edits to every document of a scenario file's text.
func editScenarioText(t *testing.T, body string, edits ...scenarioEdit) string {
	t.Helper()
	dec := yaml.NewDecoder(strings.NewReader(body))
	var docs []*yaml.Node
	for {
		var n yaml.Node
		err := dec.Decode(&n)
		if err != nil {
			break
		}
		docs = append(docs, &n)
	}
	for _, e := range edits {
		e(t, docs)
	}
	var out bytes.Buffer
	enc := yaml.NewEncoder(&out)
	enc.SetIndent(2)
	for _, d := range docs {
		if err := enc.Encode(d); err != nil {
			t.Fatal(err)
		}
	}
	_ = enc.Close()
	return out.String()
}

// writeVariant writes the scenario file src edited by edits into a fresh scenario directory as
// variant.yaml and returns the directory.
func writeVariant(t *testing.T, src string, edits ...scenarioEdit) string {
	t.Helper()
	dir := t.TempDir()
	body := editScenarioText(t, readFile(t, src), edits...)
	if err := os.WriteFile(filepath.Join(dir, "variant.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// A variant states its difference and nothing else changes: an edit that sets a field to the
// value it has leaves the file's meaning alone, and a stated difference is read back.
func TestScenarioVariant_EditsOnlyWhatIsStated(t *testing.T) {
	const doc = "kind: Scenario\nname: s\n---\nkind: Deployment\npools:\n  - role: prefill\n    engine:\n      block_size: 64\n  - role: decode\n    engine:\n      block_size: 64\n"
	out := editScenarioText(t, doc, setField("Deployment", "128", "pools", "role=decode", "engine", "block_size"),
		setField("Deployment", "4", "pools", "role=decode", "engine", "max_num_seqs"), removeField("Scenario", "name"))
	var docs []map[string]any
	dec := yaml.NewDecoder(strings.NewReader(out))
	for {
		var m map[string]any
		if dec.Decode(&m) != nil {
			break
		}
		docs = append(docs, m)
	}
	pools := docs[1]["pools"].([]any)
	pre, dec2 := pools[0].(map[string]any)["engine"].(map[string]any), pools[1].(map[string]any)["engine"].(map[string]any)
	if pre["block_size"] != 64 || len(pre) != 1 {
		t.Errorf("the prefill pool changed: %v", pre)
	}
	if dec2["block_size"] != 128 || dec2["max_num_seqs"] != 4 {
		t.Errorf("the decode pool = %v, want block_size 128 and max_num_seqs 4", dec2)
	}
	if _, ok := docs[0]["name"]; ok {
		t.Error("removeField left the field")
	}
}

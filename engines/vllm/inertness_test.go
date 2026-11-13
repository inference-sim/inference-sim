// Original to blis-vllm.

package blisvllm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/fields"
)

// A field is proven inert (design 7.4, tier T1) when the scenario set gives identical traces at its default
// and at each probe value, and every upstream function that reads it ran. oracle/inertness.py does that
// against vLLM's own code and writes the result into oracle/inertness/<field>.json. This test holds the
// list to those files, without vLLM:
//
//   - a proven_inert field names its proof file, the file exists, is for that field, and its verdict is inert;
//   - a proof file belongs to a proven_inert field;
//   - the probes cover every value class of the field (fields/value_classes.json), because a probe in one
//     class shows nothing about another;
//   - the engine's own Go code does not read the field: no non-test file outside the listed declaration and
//     validation files names the identifiers the proof lists, so the plan, the step shapes and the kernel's
//     settings do not receive it.
type proofFile struct {
	Field       string   `json:"field"`
	Kind        string   `json:"kind"`
	Claim       string   `json:"claim"`
	Identifiers []string `json:"go_identifiers"`
	Allowed     []string `json:"go_files_allowed"`
	// Refusals are the Go tests that pin the fact a reviewed static proof rests on (the Mooncake
	// refusal behind the MC_ variables): each must still exist in the file it names.
	Refusals []struct {
		File string `json:"file"`
		Func string `json:"func"`
	} `json:"refusal_tests"`
	Probes []struct {
		Value any `json:"value"`
	} `json:"probes"`
	Result struct {
		Verdict  string   `json:"verdict"`
		Problems []string `json:"problems"`
	} `json:"result"`
}

func readProof(t *testing.T, path string) proofFile {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var p proofFile
	if err := json.Unmarshal(raw, &p); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	return p
}

func proofProblems(l *fields.FieldList, doc valueClassesDoc, dir string) []string {
	var out []string
	inert := map[string]fields.Field{}
	for _, f := range l.Fields {
		if f.Status == fields.StatusProvenInert {
			inert[f.Name] = f
		}
	}
	files, _ := filepath.Glob(filepath.Join(dir, "oracle", "inertness", "*.json"))
	have := map[string]bool{}
	for _, path := range files {
		name := strings.TrimSuffix(filepath.Base(path), ".json")
		have[name] = true
		if _, ok := inert[name]; !ok {
			out = append(out, fmt.Sprintf("%s: a proof file for a field that is not proven_inert", path))
		}
	}
	for name, f := range inert {
		want := "oracle/inertness/" + name + ".json"
		if !strings.HasPrefix(f.Evidence.Proof, want) {
			out = append(out, fmt.Sprintf("%s: proof %q does not name %s", name, f.Evidence.Proof, want))
			continue
		}
		if !have[name] {
			out = append(out, fmt.Sprintf("%s: %s does not exist", name, want))
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dir, want))
		if err != nil {
			out = append(out, err.Error())
			continue
		}
		var p proofFile
		if err := json.Unmarshal(raw, &p); err != nil {
			out = append(out, fmt.Sprintf("%s: %v", want, err))
			continue
		}
		if p.Field != name {
			out = append(out, fmt.Sprintf("%s: the proof is for %q", name, p.Field))
		}
		if p.Result.Verdict != "inert" || len(p.Result.Problems) > 0 {
			out = append(out, fmt.Sprintf("%s: verdict %q, problems %v", name, p.Result.Verdict, p.Result.Problems))
		}
		out = append(out, classCoverage(name, doc, p)...)
		out = append(out, engineReads(dir, name, p)...)
		out = append(out, refusalTests(dir, name, p)...)
	}
	sort.Strings(out)
	return out
}

// classCoverage: the probes cover every class the value classes name.
func classCoverage(name string, doc valueClassesDoc, p proofFile) []string {
	rec, ok := doc.Fields[name]
	if !ok {
		return []string{name + ": no value classes"}
	}
	vc := rec.ValueClasses
	switch vc.Type {
	case "numeric":
		seen := map[int]bool{}
		for _, pr := range p.Probes {
			x, ok := pr.Value.(float64)
			if !ok {
				return []string{fmt.Sprintf("%s: probe %v is not a number", name, pr.Value)}
			}
			k := 0
			for _, t := range vc.Thresholds {
				if x > t.Value || x == t.Value && t.Closed == "above" {
					k++
				}
			}
			seen[k] = true
		}
		if len(seen) != len(vc.Thresholds)+1 {
			return []string{fmt.Sprintf("%s: the probes reach %d of %d classes", name, len(seen), len(vc.Thresholds)+1)}
		}
	default:
		want := vc.Values
		if vc.Type == "boolean" {
			want = []string{"true", "false"}
		}
		got := map[string]bool{}
		for _, pr := range p.Probes {
			got[fmt.Sprint(pr.Value)] = true
		}
		for _, w := range want {
			if !got[w] {
				return []string{fmt.Sprintf("%s: no probe sets %s", name, w)}
			}
		}
	}
	return nil
}

// refusalTests: each Go test the proof names as the refusal it rests on is a function of the file named.
func refusalTests(dir, name string, p proofFile) []string {
	var out []string
	for _, r := range p.Refusals {
		raw, err := os.ReadFile(filepath.Join(dir, r.File))
		if err != nil {
			out = append(out, fmt.Sprintf("%s: refusal test file %s: %v", name, r.File, err))
		} else if !strings.Contains(string(raw), "\nfunc "+r.Func+"(t *testing.T)") {
			out = append(out, fmt.Sprintf("%s: %s has no test %s", name, r.File, r.Func))
		}
	}
	return out
}

// engineReads: the proof's identifiers appear in no non-test Go file but the allowed ones.
func engineReads(dir, name string, p proofFile) []string {
	var out []string
	allowed := map[string]bool{}
	for _, a := range p.Allowed {
		allowed[a] = true
	}
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(dir, path)
		if info.IsDir() && (strings.HasPrefix(info.Name(), ".") || rel == "oracle" || rel == "docs") {
			return filepath.SkipDir
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") || allowed[rel] {
			return nil
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		for _, id := range p.Identifiers {
			if strings.Contains(string(raw), id) {
				out = append(out, fmt.Sprintf("%s: %s names %s, which the proof says nothing but %v reads", name, rel, id, p.Allowed))
			}
		}
		return nil
	})
	if err != nil {
		out = append(out, err.Error())
	}
	return out
}

func TestEveryProvenInertFieldHasAPassingProof(t *testing.T) {
	l, doc := loadClasses(t)
	for _, p := range proofProblems(l, doc, ".") {
		t.Error(p)
	}
}

func TestTheProofCheckFailsOnEachKnownBadInput(t *testing.T) {
	l, doc := loadClasses(t)
	var name string
	for _, f := range l.Fields {
		if f.Status == fields.StatusProvenInert {
			if p := readProof(t, "oracle/inertness/"+f.Name+".json"); len(p.Identifiers) > 0 {
				name = f.Name
				break
			}
		}
	}
	if name == "" {
		t.Skip("no proven_inert field with a runs proof yet")
	}
	copyTree := func() string {
		dir := t.TempDir()
		mustCopy(t, "oracle/inertness", filepath.Join(dir, "oracle", "inertness"), ".json")
		for _, d := range []string{"core", "compat", "vllmengine", "rules", "registry"} {
			mustCopy(t, d, filepath.Join(dir, d), ".go")
		}
		return dir
	}
	good := copyTree()
	if got := proofProblems(l, doc, good); len(got) != 0 {
		t.Fatalf("the copy of the tree is not clean: %v", got)
	}
	cases := map[string]func(dir string){
		"a verdict that is not inert": func(dir string) {
			p := filepath.Join(dir, "oracle", "inertness", name+".json")
			raw, _ := os.ReadFile(p)
			_ = os.WriteFile(p, []byte(strings.Replace(string(raw), `"verdict": "inert"`, `"verdict": "not_inert"`, 1)), 0o644)
		},
		"a missing proof file": func(dir string) { _ = os.Remove(filepath.Join(dir, "oracle", "inertness", name+".json")) },
		"an engine file that reads the field": func(dir string) {
			p := readProof(t, "oracle/inertness/"+name+".json")
			_ = os.WriteFile(filepath.Join(dir, "core", "reads.go"), []byte("package core\n\nvar _ = "+p.Identifiers[0]+"\n"), 0o644)
		},
		"a refusal test the proof names that no longer exists": func(dir string) {
			p := filepath.Join(dir, "oracle", "inertness", "MC_SLICE_SIZE.json")
			raw, _ := os.ReadFile(p)
			_ = os.WriteFile(p, []byte(strings.Replace(string(raw), "TestNoMooncakeConnectorIsServed", "TestRenamed", 1)), 0o644)
		},
		"a proof file for a field that is not proven_inert": func(dir string) {
			_ = os.WriteFile(filepath.Join(dir, "oracle", "inertness", "max_num_seqs.json"), []byte("{}"), 0o644)
		},
	}
	for what, mutate := range cases {
		dir := copyTree()
		mutate(dir)
		if got := proofProblems(l, doc, dir); len(got) == 0 {
			t.Errorf("%s: no problem found", what)
		}
	}
}

func mustCopy(t *testing.T, from, to, ext string) {
	t.Helper()
	err := filepath.Walk(from, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(from, path)
		dst := filepath.Join(to, rel)
		if info.IsDir() {
			return os.MkdirAll(dst, 0o755)
		}
		if !strings.HasSuffix(path, ext) {
			return nil
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(dst, raw, 0o644)
	})
	if err != nil {
		t.Fatal(err)
	}
}

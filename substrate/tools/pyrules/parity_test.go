// Package pyrules holds the X15 checks on Python code: the lint in pyrules.py, which flags
// schema-like rule code, and this parity test, which runs every Python check that restates
// a Go rule against the Go validator on one set of failing examples.
package pyrules

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

type restatement struct {
	Name     string   `json:"name"`
	Command  []string `json:"command"`
	Examples []string `json:"examples"`
}

func loadRestatements(t *testing.T) []restatement {
	raw, err := os.ReadFile("restatements.json")
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Comment      string        `json:"_comment"`
		Restatements []restatement `json:"restatements"`
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&doc); err != nil {
		t.Fatal(err)
	}
	return doc.Restatements
}

// goRefuses runs validate-registry on a registry holding one example set.
func goRefuses(t *testing.T, example string) (bool, string) {
	t.Helper()
	dir := t.TempDir()
	src, err := os.ReadFile(filepath.Join("parity", example))
	if err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(dir, "coefficients", filepath.Base(example))
	os.MkdirAll(filepath.Dir(dst), 0o755)
	os.WriteFile(dst, src, 0o644)
	cmd := exec.Command("go", "run", "../../validate/cmd/validate-registry", dir)
	out, err := cmd.CombinedOutput()
	if err == nil {
		return false, string(out)
	}
	if ee, ok := err.(*exec.ExitError); !ok || ee.ExitCode() != 1 {
		t.Fatalf("validate-registry could not run on %s: %v\n%s", example, err, out)
	}
	return true, string(out)
}

// Every failing example is refused by the Go validator. An example the Go rule accepts is
// not a failing example, and a parity test over it would prove nothing (X18).
func TestGoRefusesEveryParityExample(t *testing.T) {
	matches, _ := filepath.Glob("parity/registry/*.yaml")
	if len(matches) < 4 {
		t.Fatalf("%d parity examples; the set holds at least 4 (X22)", len(matches))
	}
	for _, m := range matches {
		rel, _ := filepath.Rel("parity", m)
		src, _ := os.ReadFile(m)
		first, _, _ := strings.Cut(string(src), "\n")
		want, ok := strings.CutPrefix(first, "# expect: ")
		if !ok {
			t.Errorf("%s: the first line names the refusal it expects, \"# expect: <message>\"", rel)
			continue
		}
		refused, out := goRefuses(t, rel)
		switch {
		case !refused:
			t.Errorf("validate-registry accepts %s, which is meant to fail:\n%s", rel, out)
		case !strings.Contains(out, want) || strings.Contains(out, "load error"):
			t.Errorf("%s is refused for another reason than %q:\n%s", rel, want, out)
		}
	}
}

// X15: each Python check that restates a Go rule refuses every failing example the Go rule
// refuses.
func TestEveryRestatementAgreesWithGo(t *testing.T) {
	for _, r := range loadRestatements(t) {
		if len(r.Examples) == 0 || len(r.Command) == 0 {
			t.Errorf("%s: names no command or no examples", r.Name)
			continue
		}
		for _, ex := range r.Examples {
			if refused, _ := goRefuses(t, ex); !refused {
				t.Errorf("%s: the Go rule accepts %s", r.Name, ex)
				continue
			}
			args := make([]string, len(r.Command))
			for i, a := range r.Command {
				args[i] = strings.ReplaceAll(a, "{file}", filepath.Join("tools/pyrules/parity", ex))
			}
			cmd := exec.Command(args[0], args[1:]...)
			cmd.Dir = "../.."
			if out, err := cmd.CombinedOutput(); err == nil {
				t.Errorf("%s accepts %s, which the Go rule refuses (X15):\n%s", r.Name, ex, out)
			}
		}
	}
}

// The lint over this tree is clean, with its allowed list exact.
func TestPythonRuleLint(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Fatal("python3 is required for the X15 lint")
	}
	vocab, err := exec.Command("go", "run", "../prcheck", "vocab").Output()
	if err != nil {
		t.Fatal(err)
	}
	vf := filepath.Join(t.TempDir(), "vocab.json")
	os.WriteFile(vf, vocab, 0o644)
	cmd := exec.Command(python, "-I", "tools/pyrules/pyrules.py", vf)
	cmd.Dir = "../.."
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
}

// Known-bad input for the lint (X18): a script that restates the method vocabulary.
func TestPythonRuleLintFindsARestatedVocabulary(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Fatal("python3 is required for the X15 lint")
	}
	script := `
import json, sys
sys.path.insert(0, "tools/pyrules")
import pyrules
vocab = {"methods": ["assumed", "measured"], "units": ["tokens"], "scope_keys": [], "provenances": [], "source_kinds": [], "source_roles": [], "licences": []}
src = '''
def validate_entry(e):
    if e.get("method") not in ("measured", "assumed"):
        raise ValueError("bad method")
'''
print(json.dumps(pyrules.findings("x.py", src, vocab)))
`
	cmd := exec.Command(python, "-I", "-c", script)
	cmd.Dir = "../.."
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	got := string(out)
	if !strings.Contains(got, "x.py: set methods assumed,measured") || !strings.Contains(got, "x.py: compare method measured in validate_entry") {
		t.Fatalf("the lint missed a restated rule: %s", got)
	}
}

// Original to blis-sglang.

package lower

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"
)

// A field the list calls inert is one the inertness proof states: no field is classed inert
// without a run that set it (blis-sglang#63, X1, SGL-3). Every inert ServerArgs field is probed at two or more values
// by TestInertFieldsChangeNoDecision, no field is probed that the list does not class inert, the
// values a proven field's proof file lists are the values the test states, and an inert
// environment variable names the engine test that reads it.
func TestEveryInertFieldIsProbedByTheProofItNames(t *testing.T) {
	l, err := fields.Load("../fields.json")
	if err != nil {
		t.Fatal(err)
	}
	inert := map[string]fields.Field{}
	for _, f := range l.Fields {
		if f.Status == fields.StatusProvenInert || f.Status == fields.StatusInertUnproven {
			inert[f.Name] = f
		}
	}
	if len(inert) < 10 {
		t.Fatalf("only %d inert fields in the list", len(inert))
	}
	engineTest := regexp.MustCompile(`sglangengine\.(Test\w+)`)
	engineSrc := ""
	for _, name := range sortedNames(inert) {
		f := inert[name]
		if strings.HasPrefix(name, "SGLANG_") {
			m := engineTest.FindStringSubmatch(f.Evidence.Note)
			if m == nil {
				t.Errorf("%s: an inert environment variable names the engine test that reads it", name)
				continue
			}
			if engineSrc == "" {
				engineSrc = readDir(t, "../sglangengine")
			}
			if !strings.Contains(engineSrc, "func "+m[1]+"(") {
				t.Errorf("%s: names sglangengine.%s, which does not exist", name, m[1])
			}
			continue
		}
		probes := inertProbes[name]
		if len(probes) < 2 {
			t.Errorf("%s: classed %s but TestInertFieldsChangeNoDecision states it at %d value(s)", name, f.Status, len(probes))
		}
		if f.Status == fields.StatusInertUnproven && !strings.Contains(f.Evidence.Note, "297e54fd") {
			t.Errorf("%s: the note does not cite SGLang's source at the pin", name)
		}
		if f.Status == fields.StatusInertUnproven && !strings.Contains(f.Evidence.Note, "TestInertFieldsChangeNoDecision") {
			t.Errorf("%s: the note does not name the test that sets it", name)
		}
		if f.Status != fields.StatusProvenInert {
			continue
		}
		b, err := os.ReadFile("../" + f.Evidence.Proof)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		var p struct {
			Probes []struct{ Value any } `json:"probes"`
		}
		if err := json.Unmarshal(b, &p); err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		stated := map[string]bool{}
		for _, v := range probes {
			stated[v] = true
		}
		for _, pr := range p.Probes {
			if !stated[fmt.Sprint(pr.Value)] {
				t.Errorf("%s: the proof file lists %v, which the test does not state", name, pr.Value)
			}
		}
		if len(p.Probes) != len(probes) {
			t.Errorf("%s: the proof file lists %d values, the test states %d", name, len(p.Probes), len(probes))
		}
	}
	for name := range inertProbes {
		if _, ok := inert[name]; !ok {
			t.Errorf("%s: probed as inert but the list does not class it inert", name)
		}
	}
}

func sortedNames(m map[string]fields.Field) []string {
	out := make([]string, 0, len(m))
	for n := range m {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

func readDir(t *testing.T, dir string) string {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var sb strings.Builder
	for _, e := range ents {
		if strings.HasSuffix(e.Name(), "_test.go") {
			b, err := os.ReadFile(dir + "/" + e.Name())
			if err != nil {
				t.Fatal(err)
			}
			sb.Write(b)
		}
	}
	return sb.String()
}

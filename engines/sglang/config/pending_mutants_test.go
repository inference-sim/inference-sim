// Original to blis-sglang.

package config

import (
	"encoding/json"
	"os"
	"regexp"
	"testing"
)

// pendingMutants is config/testdata/pending_mutants.json: the mutants of the ported files (the
// mutation sweep, oracle/mutate.py) that no case in the accepted input space can tell from the
// original. A pending entry names the input that reaches its branch and the stage that will accept
// it; an equivalent entry gives the reason no input tells it apart.
type pendingEntry struct {
	ID    string         `json:"id"`
	Needs map[string]any `json:"needs"`
	Stage string         `json:"stage"`
	Issue string         `json:"issue"`
}

type pendingMutants struct {
	Pending    []pendingEntry `json:"pending"`
	Equivalent []struct {
		ID  string `json:"id"`
		Why string `json:"why"`
	} `json:"equivalent"`
}

// TestPendingMutantsStayUnreachable: a pending entry whose input is accepted fails, so the list
// can only shrink. When a stage accepts the field (config/accepted.go), a model or a chip, the
// stage adds the cases that reach its branches and deletes the entries; until then the branch is
// reachable by no input Resolve takes.
func TestPendingMutantsStayUnreachable(t *testing.T) {
	b, err := os.ReadFile("testdata/pending_mutants.json")
	if err != nil {
		t.Fatal(err)
	}
	var p pendingMutants
	if err := json.Unmarshal(b, &p); err != nil {
		t.Fatal(err)
	}
	if len(p.Pending) == 0 {
		t.Fatal("no pending mutant listed (non-vacuity)")
	}
	issue := regexp.MustCompile(`#\d+`)
	seen := map[string]bool{}
	for _, e := range p.Pending {
		if seen[e.ID] {
			t.Errorf("%s is listed twice", e.ID)
		}
		seen[e.ID] = true
		if e.Stage == "" || !issue.MatchString(e.Issue) {
			t.Errorf("%s: a pending entry names its stage and the issue that closes it (got %q, %q)", e.ID, e.Stage, e.Issue)
		}
		if reached, what := e.reachable(t); reached {
			t.Errorf("%s: %s is accepted now; add the cases that reach this branch and delete the entry", e.ID, what)
		}
	}
	for _, e := range p.Equivalent {
		if seen[e.ID] {
			t.Errorf("%s is listed twice", e.ID)
		}
		seen[e.ID] = true
		if e.Why == "" {
			t.Errorf("%s: an equivalent mutant says why no input tells it apart", e.ID)
		}
	}
}

// reachable reports whether the input a pending entry needs is accepted by Resolve now, and what
// that input is.
func (e pendingEntry) reachable(t *testing.T) (bool, string) {
	t.Helper()
	str := func(k string) string { s, _ := e.Needs[k].(string); return s }
	switch {
	case str("field") != "":
		_, ok := acceptedFields[str("field")]
		return ok, "the field " + str("field")
	case str("chip") != "":
		cc := map[string][2]int{"sm100": {10, 0}, "sm120": {12, 0}}[str("chip")]
		chips := acceptedModels[str("architecture")]
		return (cc == [2]int{9, 0} || cc == [2]int{10, 0}) && containsInt(chips, cc[0]), str("architecture") + " on " + str("chip")
	case str("speculation_on") != "":
		return eagleDraftInTarget[str("speculation_on")], "speculative decoding on " + str("speculation_on")
	case str("architecture") != "":
		_, ok := acceptedModels[str("architecture")]
		return ok, "the architecture " + str("architecture")
	case e.Needs["value"] != nil:
		v := e.Needs["value"].([]any)
		allowed, ok := acceptedFields[v[0].(string)]
		return ok && (allowed == nil || contains(allowed, v[1])), "the value " + v[1].(string) + " of " + v[0].(string)
	case str("model_fact") == "dsa_index_kpool_above_1":
		for _, c := range readAnswers(t).Cases {
			if c.ModelFacts.DSAIndexKpool != nil && *c.ModelFacts.DSAIndexKpool > 1 {
				return true, "a case " + c.Name + " whose model's DSA index pool holds more than one page group"
			}
		}
		return false, ""
	}
	t.Fatalf("%s: needs %v is not an input this test knows", e.ID, e.Needs)
	return false, ""
}

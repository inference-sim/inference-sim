package pointrecord

import (
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/spec/result"
)

// A point the dashboard writes when its engine release is another one and two conditions of
// G1 could not be evaluated: G4, with an other_release and one unevaluated substitution per
// condition, each naming the condition as its subject. The record travels as a document, so it
// is read back strictly (the loader behind `inference-sim schema validate point-record`) and
// validated; the dashboard defines no schema of its own for it (SUB-19).
func TestAPointWithUnevaluatedConditionsReadsBackAndValidates(t *testing.T) {
	r := valid()
	r.Grade = G4
	r.Substitutions = []Substitution{
		{Kind: SubChangedFunctions, Subject: "ported_functions_unpublished", Detail: "the engine publishes no list of the functions it ports"},
		{Kind: SubOtherRelease, Subject: "0.33.0", Detail: "0.33.0, scored on 0.32.0"},
		{Kind: SubUnevaluated, Subject: "kernel_hash", Detail: "kernel_hash is unevaluated"},
		{Kind: SubUnevaluated, Subject: "value_class", Detail: "value_class is unevaluated"},
	}
	if p := r.Validate(); !p.OK() {
		t.Fatalf("the record was refused:\n%s", p.Error())
	}
	doc, err := yaml.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	var back Record
	if err := result.UnmarshalStrict(doc, &back); err != nil {
		t.Fatalf("the document does not read back: %v", err)
	}
	if p := back.Validate(); !p.OK() {
		t.Fatalf("the document read back is refused:\n%s", p.Error())
	}
	if n := len(back.Substitutions); n != 4 || back.Substitutions[2].Kind != SubUnevaluated || back.Substitutions[2].Subject != "kernel_hash" {
		t.Errorf("substitutions read back: %+v", back.Substitutions)
	}

	// The condition is the subject: a name that is not a condition of G1 is refused.
	back.Substitutions[2].Subject = "the kernel"
	if p := back.Validate(); p.OK() {
		t.Error("an unevaluated substitution naming no condition of G1 was accepted")
	}
}

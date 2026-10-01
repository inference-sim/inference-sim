package api

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// TestKindPartitionIsTotalAndDisjoint is a LAW over the kind lists rather than a restatement
// of them: every kind is exactly one of input and output, the two lists have the same
// length, and each list is sorted. A kind added to only one half of the envelope (an input
// verb with no result form) fails here.
func TestKindPartitionIsTotalAndDisjoint(t *testing.T) {
	inputs, outputs, all := InputKinds(), OutputKinds(), AllKinds()

	if len(inputs) != len(outputs) {
		t.Errorf("every input kind needs a result form: %d input kinds, %d output kinds", len(inputs), len(outputs))
	}
	if got, want := len(all), len(inputs)+len(outputs); got != want {
		t.Errorf("AllKinds returned %d kinds, want %d (inputs + outputs)", got, want)
	}
	for _, kinds := range [][]Kind{inputs, outputs, all} {
		if !slices.IsSorted(kinds) {
			t.Errorf("kind list %v is not sorted; a derived schema enum would depend on it (INV-6)", kinds)
		}
	}
	for _, k := range all {
		if IsInput(k) == IsOutput(k) {
			t.Errorf("kind %q is input=%v and output=%v; it must be exactly one", k, IsInput(k), IsOutput(k))
		}
		if !IsValidKind(k) {
			t.Errorf("kind %q is in AllKinds but IsValidKind rejects it", k)
		}
		if field := BodyField(k); field != "spec" && field != "result" {
			t.Errorf("kind %q has body field %q, want spec or result", k, field)
		}
	}
}

// TestResultKindPairsEveryInputKind pins the naming rule the format relies on: a reader can
// tell which verb produced a result document from the kind alone.
func TestResultKindPairsEveryInputKind(t *testing.T) {
	for _, k := range InputKinds() {
		result, ok := ResultKind(k)
		if !ok {
			t.Errorf("ResultKind(%q) reported no result form for an input kind", k)
			continue
		}
		if want := k + "Result"; result != want {
			t.Errorf("ResultKind(%q) = %q, want %q", k, result, want)
		}
		if !IsOutput(result) {
			t.Errorf("ResultKind(%q) = %q, which IsOutput rejects", k, result)
		}
		// No "RunResultResult": the suffix is applied to input kinds only.
		if _, ok := ResultKind(result); ok {
			t.Errorf("ResultKind(%q) returned a result form for a kind that is already a result", result)
		}
	}
	if _, ok := ResultKind("Nonsense"); ok {
		t.Error("ResultKind accepted an undefined kind")
	}
}

// TestAccessorsReturnCopies guards the R8 reason the kind lists are unexported: a caller
// must not be able to mutate the envelope's definition of its own kinds.
func TestAccessorsReturnCopies(t *testing.T) {
	stolen := InputKinds()
	stolen[0] = "Mutated"
	if got := InputKinds()[0]; got == "Mutated" {
		t.Error("InputKinds returned the package's own slice; a caller can redefine the input kinds")
	}
	if !IsInput(KindCalibrate) {
		t.Error("mutating the returned slice changed IsInput")
	}
}

func TestVersionConstantIsTheOnlyValidAPIVersion(t *testing.T) {
	if !Version.Valid() {
		t.Fatalf("Version %q reports itself invalid", Version)
	}
	for _, other := range []APIVersion{"", "llm-d-perf-simulator/v2", "v1", "LLM-D-PERF-SIMULATOR/V1"} {
		if other.Valid() {
			t.Errorf("apiVersion %q was accepted; exactly one value is defined", other)
		}
	}
}

func TestConstructorsStampTheEnvelope(t *testing.T) {
	spec, err := NewSpec(KindRun, Body{"deployment": "placeholder"})
	if err != nil {
		t.Fatalf("NewSpec(Run): %v", err)
	}
	if spec.APIVersion != Version || spec.Kind != KindRun || spec.Spec == nil || spec.Result != nil {
		t.Errorf("NewSpec(Run) = %+v, want apiVersion %q, kind Run, a spec and no result", spec, Version)
	}
	if err := spec.Validate(); err != nil {
		t.Errorf("NewSpec produced a document its own validator rejects: %v", err)
	}

	result, err := NewResult(KindRunResult, nil)
	if err != nil {
		t.Fatalf("NewResult(RunResult): %v", err)
	}
	if result.Result == nil || len(*result.Result) != 0 {
		t.Errorf("NewResult(nil body) = %+v, want an empty, non-nil result body", result)
	}
	if err := result.Validate(); err != nil {
		t.Errorf("NewResult produced a document its own validator rejects: %v", err)
	}

	// A constructor must refuse the wrong half of the partition rather than stamp a
	// document no reader accepts.
	if _, err := NewSpec(KindRunResult, nil); err == nil {
		t.Error("NewSpec accepted an output kind")
	}
	if _, err := NewResult(KindRun, nil); err == nil {
		t.Error("NewResult accepted an input kind")
	}
	if _, err := NewSpec("Nonsense", nil); err == nil {
		t.Error("NewSpec accepted an undefined kind")
	}
}

// TestDocumentValidateReportsEveryProblem is the R1 half of Validate: a caller fixing a
// document must see all of its problems, not the first one.
func TestDocumentValidateReportsEveryProblem(t *testing.T) {
	bad := Document{APIVersion: "llm-d-perf-simulator/v9", Kind: "Nonsense"}
	err := bad.Validate()
	if err == nil {
		t.Fatal("a document with a wrong apiVersion and an undefined kind was accepted")
	}
	for _, want := range []string{"apiVersion", "kind"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("Validate error %q does not mention %s", err, want)
		}
	}
}

func TestDocumentValidateEnforcesBodyExclusivity(t *testing.T) {
	empty := Body{}
	cases := []struct {
		name     string
		document Document
		wantErr  bool
	}{
		{"input kind with a spec", Document{Version, KindRun, &empty, nil}, false},
		{"output kind with a result", Document{Version, KindRunResult, nil, &empty}, false},
		{"input kind with no spec", Document{Version, KindRun, nil, nil}, true},
		{"output kind with no result", Document{Version, KindRunResult, nil, nil}, true},
		{"input kind carrying a result", Document{Version, KindRun, &empty, &empty}, true},
		{"output kind carrying a spec", Document{Version, KindRunResult, &empty, &empty}, true},
		{"input kind carrying only a result", Document{Version, KindRun, nil, &empty}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.document.Validate()
			if tc.wantErr && err == nil {
				t.Errorf("document %+v was accepted", tc.document)
			}
			if !tc.wantErr && err != nil {
				t.Errorf("document %+v was rejected: %v", tc.document, err)
			}
		})
	}
}

// TestEmptyBodySurvivesRoundTrip is why Spec and Result are pointers (R9): `spec: {}` is a
// valid minimal input document, and a plain map would be erased by omitempty, turning
// "present and empty" into "missing" — which Validate then rejects.
func TestEmptyBodySurvivesRoundTrip(t *testing.T) {
	original, err := NewSpec(KindRun, nil)
	if err != nil {
		t.Fatalf("NewSpec: %v", err)
	}

	t.Run("yaml", func(t *testing.T) {
		encoded, err := yaml.Marshal(original)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		if !strings.Contains(string(encoded), "spec:") {
			t.Fatalf("an empty spec was erased by omitempty:\n%s", encoded)
		}
		var decoded Document
		decoder := yaml.NewDecoder(strings.NewReader(string(encoded)))
		decoder.KnownFields(true) // strict: the struct tags must cover every emitted key (R10)
		if err := decoder.Decode(&decoded); err != nil {
			t.Fatalf("strict decode of our own output: %v", err)
		}
		if err := decoded.Validate(); err != nil {
			t.Errorf("round-tripped document is invalid: %v", err)
		}
	})

	t.Run("json", func(t *testing.T) {
		encoded, err := json.Marshal(original)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		if !strings.Contains(string(encoded), `"spec"`) {
			t.Fatalf("an empty spec was erased by omitempty: %s", encoded)
		}
		var decoded Document
		if err := json.Unmarshal(encoded, &decoded); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if err := decoded.Validate(); err != nil {
			t.Errorf("round-tripped document is invalid: %v", err)
		}
	})
}

// Original to blis-sglang.

package config

import (
	"encoding/json"
	"errors"
	"os"
	"sort"
	"strings"
	"testing"
)

// answers is config/testdata/serverargs.json: SGLang's own resolution of each case of
// oracle/serverargs/cases.json (oracle/run_serverargs.py), with each case's inputs.
type answers struct {
	Cases []struct {
		Name       string            `json:"name"`
		Args       map[string]any    `json:"args"`
		Chip       Chip              `json:"chip"`
		ModelFacts ModelFacts        `json:"model_facts"`
		Resolved   map[string]any    `json:"resolved"`
		DeclaredBy map[string]string `json:"declared_by"`
		Refused    *struct {
			Message string `json:"message"`
		} `json:"refused"`
		RefusedAt string `json:"refused_at"`
	} `json:"cases"`
}

func readAnswers(t *testing.T) answers {
	t.Helper()
	b, err := os.ReadFile("testdata/serverargs.json")
	if err != nil {
		t.Fatal(err)
	}
	var a answers
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if len(a.Cases) == 0 {
		t.Fatal("testdata/serverargs.json holds no case")
	}
	return a
}

func canon(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(b)
}

// TestResolveAgainstSGLang: every case resolves as SGLang v0.5.23's own pipeline resolves it,
// field by field over every field SGLang records, with the step that decided each changed field;
// a case SGLang refuses is refused with SGLang's message, raised in the same function.
func TestResolveAgainstSGLang(t *testing.T) {
	for _, c := range readAnswers(t).Cases {
		t.Run(c.Name, func(t *testing.T) {
			args := map[string]any{"model_path": "<model>"}
			for k, v := range c.Args {
				args[k] = v
			}
			got, err := Resolve(Input{Args: args, Chip: c.Chip, Model: c.ModelFacts})
			if c.Refused != nil {
				var up *UpstreamRefusal
				if !errors.As(err, &up) {
					t.Fatalf("SGLang refuses this case (%q at %s); Resolve returned %v", c.Refused.Message, c.RefusedAt, err)
				}
				if up.Message != c.Refused.Message || up.Site != c.RefusedAt {
					t.Errorf("refusal %q at %s, SGLang's is %q at %s", up.Message, up.Site, c.Refused.Message, c.RefusedAt)
				}
				return
			}
			if err != nil {
				t.Fatalf("SGLang accepts this case; Resolve refused it: %v", err)
			}
			if len(c.Resolved) != len(FieldNames())+3 {
				t.Errorf("SGLang records %d fields, ServerArgs in Go has %d and 3 internal ones", len(c.Resolved), len(FieldNames()))
			}
			names := make([]string, 0, len(c.Resolved))
			for n := range c.Resolved {
				names = append(names, n)
			}
			sort.Strings(names)
			for _, name := range names {
				want := c.Resolved[name]
				v, ok := got.Value(name)
				if !ok {
					t.Errorf("%s: SGLang records it; ServerArgs in Go has no such field", name)
					continue
				}
				if name == "random_seed" && want == "<random>" {
					if !got.RandomSeedDrawn || v != nil {
						t.Errorf("random_seed: SGLang draws it; Resolve gives %v, drawn=%v", v, got.RandomSeedDrawn)
					}
					continue
				}
				if canon(v) != canon(want) {
					t.Errorf("%s = %s, SGLang resolves %s", name, canon(v), canon(want))
				}
			}
			for name, src := range c.DeclaredBy {
				if strings.HasPrefix(name, "_") || name == "random_seed" {
					continue
				}
				if got.DeclaredBy[name] != src {
					t.Errorf("%s is decided by %q, SGLang's step is %q", name, got.DeclaredBy[name], src)
				}
			}
			for name, src := range got.DeclaredBy {
				if _, ok := c.DeclaredBy[name]; !ok {
					t.Errorf("%s is decided by %q, which SGLang's resolution leaves at its input", name, src)
				}
			}
		})
	}
}

// TestResolveRefusesAnUnknownFieldByItsName: a name that is not a ServerArgs field is refused,
// and the refusal names it (X1). The nearest accepted input resolves.
func TestResolveRefusesAnUnknownFieldByItsName(t *testing.T) {
	c := readAnswers(t).Cases[0]
	in := Input{Args: map[string]any{"model_path": "<model>"}, Chip: c.Chip, Model: c.ModelFacts}
	if _, err := Resolve(in); err != nil {
		t.Fatalf("the nearest accepted input: %v", err)
	}
	in.Args["max_num_seqs"] = 8 // vLLM's name for SGLang's max_running_requests
	_, err := Resolve(in)
	var u *UnknownField
	if !errors.As(err, &u) || u.Name != "max_num_seqs" || !strings.Contains(err.Error(), "max_num_seqs") {
		t.Fatalf("got %v, want an UnknownField naming max_num_seqs", err)
	}
}

// TestResolveRefusesWhatItDoesNotModelByName: a ServerArgs field outside the modelled input
// space, set away from its default, is refused naming the field and an open issue; the same
// field at its default resolves.
func TestResolveRefusesWhatItDoesNotModelByName(t *testing.T) {
	c := readAnswers(t).Cases[0]
	for _, tc := range []struct {
		field string
		value any
		issue string
	}{
		{"attn_cp_size", 2, "inference-sim/blis-sglang#26"},
		{"pp_size", 2, issueRest},
		{"disaggregation_transfer_backend", "mori", issueRest},
	} {
		in := Input{Args: map[string]any{"model_path": "<model>"}, Chip: c.Chip, Model: c.ModelFacts}
		def := Defaults()
		in.Args[tc.field] = def.value(tc.field)
		if _, err := Resolve(in); err != nil {
			t.Errorf("%s at its default: %v", tc.field, err)
		}
		in.Args[tc.field] = tc.value
		_, err := Resolve(in)
		var nm *NotModelled
		if !errors.As(err, &nm) || nm.Name != tc.field || nm.Issue != tc.issue {
			t.Errorf("%s=%v: got %v, want NotModelled naming it and %s", tc.field, tc.value, err, tc.issue)
		}
	}
	// P/D over the default transfer backend, mooncake, is refused citing its own issue.
	in := Input{Args: map[string]any{"model_path": "<model>", "disaggregation_mode": "prefill"}, Chip: c.Chip, Model: c.ModelFacts}
	_, err := Resolve(in)
	var nm *NotModelled
	if !errors.As(err, &nm) || nm.Name != "disaggregation_transfer_backend" || nm.Issue != "inference-sim/blis-sglang#18" {
		t.Errorf("P/D over mooncake: got %v, want NotModelled naming disaggregation_transfer_backend and #18", err)
	}
}

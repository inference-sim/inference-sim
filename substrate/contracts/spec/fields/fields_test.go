package fields

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// valid is a list that exercises every status, both sources and every estimation rule.
const valid = `{
  "kind": "FieldList",
  "engine": {
    "kind": "vllm",
    "version": "0.32.0"
  },
  "fields": [
    {
      "name": "VLLM_USE_V2_MODEL_RUNNER",
      "source": "env",
      "status": "inert_unproven",
      "evidence": {
        "note": "the dashboard's table marked it inert"
      },
      "estimation": {
        "rule": "none"
      }
    },
    {
      "name": "kv_cache_dtype",
      "source": "config",
      "status": "not_modelled",
      "evidence": {
        "issue": "inference-sim/blis-vllm#5"
      },
      "estimation": {
        "rule": "substitute",
        "setting": {
          "kv_cache_dtype": "auto",
          "max_num_seqs": 256
        },
        "reason": "an fp8 cache is priced as the model's own dtype"
      }
    },
    {
      "name": "max_num_seqs",
      "source": "config",
      "status": "modelled",
      "evidence": {
        "traces": [
          "oracle/traces/t2-basic.json"
        ]
      },
      "estimation": {
        "rule": "leave_unset"
      }
    },
    {
      "name": "max_num_seqs",
      "source": "env",
      "status": "proven_inert",
      "evidence": {
        "proof": "TestMaxNumSeqsEnvIsInert"
      },
      "estimation": {
        "rule": "leave_unset"
      }
    }
  ]
}
`

func TestAValidListParsesAndMarshalsToItsOwnBytes(t *testing.T) {
	l, err := Parse([]byte(valid))
	if err != nil {
		t.Fatal(err)
	}
	if len(l.Fields) != 4 || l.Engine != (Engine{Kind: "vllm", Version: "0.32.0"}) {
		t.Fatalf("parsed %+v", l)
	}
	if got := l.Fields[1].Estimation.Setting["max_num_seqs"]; got != json.Number("256") {
		t.Errorf("a setting's number reads as %#v, want json.Number(\"256\")", got)
	}
	out, err := l.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out, []byte(valid)) {
		t.Errorf("Marshal is not the canonical form:\n%s", out)
	}
}

func TestLoadNamesTheFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fields.json")
	if err := os.WriteFile(path, []byte(`{"kind": "FieldList", "x": 1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil || !strings.Contains(err.Error(), path) {
		t.Errorf("Load = %v, want an error naming %s", err, path)
	}
	path = filepath.Join(t.TempDir(), "ok.json")
	if err := os.WriteFile(path, []byte(valid), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err != nil {
		t.Errorf("Load(valid) = %v", err)
	}
}

// edit returns valid with old replaced by new, once; the test fails if old is absent.
func edit(t *testing.T, old, new string) []byte {
	t.Helper()
	if strings.Count(valid, old) != 1 {
		t.Fatalf("%q occurs %d times in the valid list", old, strings.Count(valid, old))
	}
	return []byte(strings.Replace(valid, old, new, 1))
}

// Each way a list can be wrong, one malformed document each (SUB-2, SUB-12). want is text the
// error must contain: the rule and, for a field, its index and name.
func TestEveryMalformedListIsRefusedNamingWhatIsWrong(t *testing.T) {
	for _, c := range []struct {
		name string
		doc  []byte
		want []string
	}{
		// The loader: SUB-12.
		{"unknown top-level key", edit(t, `"kind": "FieldList",`, `"kind": "FieldList", "extra": 1,`),
			[]string{"unknown key \"extra\"", "the top level"}},
		{"key differing only in case", edit(t, `"kind": "FieldList",`, `"Kind": "FieldList",`),
			[]string{"unknown key \"Kind\""}},
		{"unknown engine key", edit(t, `"version": "0.32.0"`, `"version": "0.32.0", "commit": "x"`),
			[]string{"engine", "unknown key \"commit\""}},
		{"unknown field key", edit(t, `"name": "max_num_seqs",
      "source": "config",`, `"name": "max_num_seqs",
      "source": "config", "default": 256,`),
			[]string{"fields[2](max_num_seqs)", "unknown key \"default\""}},
		{"unknown evidence key", edit(t, `"proof": "TestMaxNumSeqsEnvIsInert"`, `"proof": "TestMaxNumSeqsEnvIsInert", "url": "x"`),
			[]string{"fields[3](max_num_seqs).evidence", "unknown key \"url\""}},
		{"unknown estimation key", edit(t, `"rule": "none"`, `"rule": "none", "value": 1`),
			[]string{"fields[0](VLLM_USE_V2_MODEL_RUNNER).estimation", "unknown key \"value\""}},
		{"duplicate top-level key", edit(t, `"kind": "FieldList",`, `"kind": "FieldList", "kind": "FieldList",`),
			[]string{"key \"kind\" appears twice"}},
		{"duplicate key inside a setting", edit(t, `"kv_cache_dtype": "auto",`, `"kv_cache_dtype": "auto", "kv_cache_dtype": "fp8",`),
			[]string{"setting", "key \"kv_cache_dtype\" appears twice"}},
		{"trailing data", []byte(valid + "{}"), []string{"data after the document"}},
		{"integer that does not fit", edit(t, `"max_num_seqs": 256`, `"max_num_seqs": 99999999999999999999`),
			[]string{"setting.max_num_seqs", "does not fit in 64 bits"}},
		{"decimal that overflows", edit(t, `"max_num_seqs": 256`, `"max_num_seqs": 1e400`),
			[]string{"outside the range"}},
		{"decimal that underflows", edit(t, `"max_num_seqs": 256`, `"max_num_seqs": 1e-400`),
			[]string{"outside the range"}},
		{"a non-finite number", edit(t, `"max_num_seqs": 256`, `"max_num_seqs": NaN`),
			[]string{"invalid character"}},
		{"wrong type", edit(t, `"version": "0.32.0"`, `"version": 32`), []string{"version"}},

		// The validator.
		{"wrong kind", edit(t, `"kind": "FieldList"`, `"kind": "Fields"`), []string{"kind", "must be \"FieldList\""}},
		{"no engine kind", edit(t, `"kind": "vllm",`, `"kind": "",`), []string{"engine.kind", "required"}},
		{"no engine version", edit(t, `"version": "0.32.0"`, `"version": ""`), []string{"engine.version", "required"}},
		{"empty name", edit(t, `"name": "VLLM_USE_V2_MODEL_RUNNER"`, `"name": ""`), []string{"fields[0].name", "required"}},
		{"unknown source", edit(t, `"source": "env",
      "status": "inert_unproven"`, `"source": "cli",
      "status": "inert_unproven"`), []string{"fields[0](VLLM_USE_V2_MODEL_RUNNER).source", "\"cli\""}},
		{"unknown status", edit(t, `"status": "inert_unproven"`, `"status": "inert"`), []string{"status", "\"inert\""}},
		{"modelled without traces", edit(t, `"traces": [
          "oracle/traces/t2-basic.json"
        ]`, `"note": "x"`), []string{"fields[2](max_num_seqs).evidence.traces", "required for status \"modelled\"", "evidence.note"}},
		{"modelled with an empty trace list", edit(t, `"traces": [
          "oracle/traces/t2-basic.json"
        ]`, `"traces": []`), []string{"evidence.traces", "at least one trace"}},
		{"modelled with an empty trace path", edit(t, `"oracle/traces/t2-basic.json"`, `""`), []string{"traces[0]", "empty string"}},
		{"modelled with a trace named twice", edit(t, `"oracle/traces/t2-basic.json"`, `"a.json", "a.json"`), []string{"traces[1]", "twice"}},
		{"proven inert without a proof", edit(t, `"proof": "TestMaxNumSeqsEnvIsInert"`, `"proof": ""`), []string{"evidence.proof", "required for status \"proven_inert\""}},
		{"proven inert with a second key", edit(t, `"proof": "TestMaxNumSeqsEnvIsInert"`, `"proof": "TestMaxNumSeqsEnvIsInert", "issue": "a/b#1"`), []string{"evidence.issue", "not allowed for status \"proven_inert\""}},
		{"not modelled without an issue", edit(t, `"issue": "inference-sim/blis-vllm#5"`, `"proof": "x"`), []string{"evidence.issue", "required for status \"not_modelled\""}},
		{"not modelled with a malformed issue", edit(t, `"issue": "inference-sim/blis-vllm#5"`, `"issue": "blis-vllm#5"`), []string{"evidence.issue", "<owner>/<repo>#<n>"}},
		{"not modelled with issue zero", edit(t, `"issue": "inference-sim/blis-vllm#5"`, `"issue": "inference-sim/blis-vllm#0"`), []string{"evidence.issue"}},
		{"inert unproven without a note", edit(t, `"note": "the dashboard's table marked it inert"`, `"note": ""`), []string{"evidence.note", "required for status \"inert_unproven\""}},
		{"unknown rule", edit(t, `"rule": "none"`, `"rule": "guess"`), []string{"estimation.rule", "\"guess\""}},
		{"substitute without a setting", edit(t, `"setting": {
          "kv_cache_dtype": "auto",
          "max_num_seqs": 256
        },`, ``), []string{"estimation.setting", "required for rule \"substitute\""}},
		{"substitute with an empty setting", edit(t, `"setting": {
          "kv_cache_dtype": "auto",
          "max_num_seqs": 256
        },`, `"setting": {},`), []string{"estimation.setting", "required for rule \"substitute\""}},
		{"substitute with an empty setting name", edit(t, `"kv_cache_dtype": "auto",`, `"": "auto",`), []string{"estimation.setting", "empty key"}},
		{"substitute without a reason", edit(t, `,
        "reason": "an fp8 cache is priced as the model's own dtype"`, ``), []string{"estimation.reason", "required for rule \"substitute\""}},
		{"a setting without substitute", edit(t, `"rule": "none"`, `"rule": "none", "setting": {"a": 1}`), []string{"estimation.setting", "only with rule \"substitute\""}},
		{"a reason without substitute", edit(t, `"rule": "none"`, `"rule": "none", "reason": "x"`), []string{"estimation.reason", "only with rule \"substitute\""}},
		{"fields out of order", edit(t, `"name": "VLLM_USE_V2_MODEL_RUNNER"`, `"name": "zz"`), []string{"fields[1](kv_cache_dtype)", "out of order"}},
		{"sources out of order", bytes.Replace(edit(t, `"name": "max_num_seqs",
      "source": "config",`, `"name": "max_num_seqs",
      "source": "env_",`), []byte(`"source": "env",
      "status": "proven_inert"`), []byte(`"source": "config",
      "status": "proven_inert"`), 1), []string{"fields[3](max_num_seqs)", "out of order"}},
		{"a field listed twice", edit(t, `"name": "max_num_seqs",
      "source": "env",`, `"name": "max_num_seqs",
      "source": "config",`), []string{"fields[3](max_num_seqs)", "appear once"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := Parse(c.doc)
			if err == nil {
				t.Fatal("Parse accepted it")
			}
			for _, w := range c.want {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("error does not mention %q:\n%v", w, err)
				}
			}
		})
	}
}

// Validate reports every problem in one pass, not the first.
func TestValidateReportsEveryProblem(t *testing.T) {
	l := &FieldList{Kind: "x", Fields: []Field{{Name: "a", Source: "y", Status: "z",
		Estimation: Estimation{Rule: "w"}}}}
	err := l.Validate()
	if err == nil {
		t.Fatal("accepted")
	}
	for _, w := range []string{"kind", "engine.kind", "engine.version", "fields[0](a).source",
		"fields[0](a).status", "fields[0](a).estimation.rule"} {
		if !strings.Contains(err.Error(), w+":") {
			t.Errorf("error does not report %s:\n%v", w, err)
		}
	}
}

func ExampleParse() {
	l, err := Parse([]byte(`{
  "kind": "FieldList",
  "engine": {"kind": "vllm", "version": "0.32.0"},
  "fields": [
    {"name": "max_num_seqs", "source": "config", "status": "modelled",
     "evidence": {"traces": ["oracle/traces/t2-basic.json"]},
     "estimation": {"rule": "leave_unset"}}
  ]
}`))
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, f := range l.Fields {
		fmt.Println(f.Name, f.Source, f.Status, f.Estimation.Rule)
	}
	// Output: max_num_seqs config modelled leave_unset
}

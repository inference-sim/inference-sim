// Original to blis-vllm.

package engineconfig

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"gopkg.in/yaml.v3"
)

// notVLLMNames are the engine block keys that are not EngineArgs names, with why. env is a
// container of environment variables, whose keys are vLLM's own variables. allreduce_backend is
// the resolved all-reduce choice, which the engine derives and EngineArgs has no field for; the
// rules packs read it, and it is the one setting of the block with no vLLM name.
var notVLLMNames = map[string]string{
	"env":               "a container of environment variables, named as vLLM reads them",
	"allreduce_backend": "the resolved choice; EngineArgs has no such field",
}

// testdata/vllm_names.json is oracle/run_engineargs.py --key-names run at the pinned vLLM over
// the keys VLLMKeys lists (the oracle-tier2 job re-derives it): whether EngineArgs takes each
// top-level name, and whether the config class behind each config object takes each nested name.
type namesTrace struct {
	Top    map[string]bool            `json:"top"`
	Nested map[string]map[string]bool `json:"nested"`
}

func loadNames(t *testing.T) namesTrace {
	t.Helper()
	raw, err := os.ReadFile("testdata/vllm_names.json")
	if err != nil {
		t.Fatal(err)
	}
	var n namesTrace
	if err := json.Unmarshal(raw, &n); err != nil {
		t.Fatal(err)
	}
	return n
}

// Every key the engine block takes is a name vLLM's own EngineArgs takes (design X1), except the
// two listed above. The oracle's answer covers exactly the keys VLLMKeys lists, so a key added to
// Engine without a name in vLLM's, or a trace out of date, fails.
func TestEveryBlockKeyIsAVLLMName(t *testing.T) {
	tr := loadNames(t)
	top, nested := VLLMKeys()
	for _, k := range top {
		known, asked := tr.Top[k]
		if !asked {
			t.Errorf("%s: the oracle was not asked about this key; regenerate testdata/vllm_names.json", k)
			continue
		}
		if why, ok := notVLLMNames[k]; ok {
			if known {
				t.Errorf("%s: listed as not a vLLM name (%s) but EngineArgs takes it; remove it from the list", k, why)
			}
			continue
		}
		if !known {
			t.Errorf("%s: not a name vLLM's EngineArgs takes", k)
		}
	}
	for g, leaves := range nested {
		for _, l := range leaves {
			known, asked := tr.Nested[g][l]
			if !asked {
				t.Errorf("%s.%s: the oracle was not asked about this key; regenerate testdata/vllm_names.json", g, l)
			} else if !known {
				t.Errorf("%s.%s: not a field of vLLM's %s", g, l, g)
			}
		}
	}
	for k := range tr.Top {
		if !contains(top, k) {
			t.Errorf("the trace has %q, which the block no longer takes; regenerate it", k)
		}
	}
}

// The check can fail: a block key that is not a vLLM name is reported, and the earlier names are
// refused with the new one named. This is the known-bad input for the test above.
func TestAKeyThatIsNotAVLLMNameFails(t *testing.T) {
	tr := loadNames(t)
	// A top-level key vLLM does not take, as the earlier layout spelled the cache dtype.
	if known, asked := tr.Top["cache_dtype"]; asked && known {
		t.Error("vLLM's EngineArgs takes cache_dtype, so the earlier spelling would pass the check")
	}
	if _, asked := tr.Top["cache_dtype"]; asked {
		t.Error("the block still takes cache_dtype; vLLM's name is kv_cache_dtype")
	}
	// The earlier spellings are refused, each naming the key to use.
	for old, want := range map[string]string{
		"cache_dtype: fp8":                          "kv_cache_dtype",
		"cudagraph_mode: FULL":                      "compilation_config.cudagraph_mode",
		"speculative: {method: mtp}":                "speculative_config",
		"dbo: {enable_dbo: true}":                   "enable_dbo",
		"eplb: {enable_eplb: true}":                 "eplb_config",
		"kv_events: {enable_kv_cache_events: true}": "kv_events_config",
		"frontend: rust":                            "VLLM_USE_RUST_FRONTEND",
	} {
		var n yaml.Node
		if err := yaml.Unmarshal([]byte(old), &n); err != nil {
			t.Fatal(err)
		}
		if _, err := FromVLLM(n.Content[0]); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: err %v, want one naming %s", old, err, want)
		}
	}
	// And a name that is in neither layout is the strict decoder's, naming the line.
	n := yamlBlock(t, "no_such_setting: 1\n")
	if _, err := Decode(n); err == nil || !strings.Contains(err.Error(), "no_such_setting") {
		t.Errorf("an unknown key: %v", err)
	}
}

func yamlBlock(t *testing.T, src string) deployment.EngineBlock {
	t.Helper()
	var n yaml.Node
	if err := yaml.Unmarshal([]byte(src), &n); err != nil {
		t.Fatal(err)
	}
	return deployment.NewEngineBlock(n.Content[0])
}

func mustMarshal(t *testing.T, v any) []byte {
	t.Helper()
	out, err := yaml.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// Decode then Encode is the identity on a block written in vLLM's names, for every key the block
// takes: a translation that dropped or misplaced a key would change the block.
func TestTheTranslationRoundTripsEveryKey(t *testing.T) {
	e, err := Decode(block(t, "testdata/every-field.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	b, err := Encode(e)
	if err != nil {
		t.Fatal(err)
	}
	again, err := Decode(b)
	if err != nil || !reflect.DeepEqual(again, e) {
		t.Errorf("Decode(Encode(e)) = %+v, %v; want %+v", again, err, e)
	}
	top, nested := VLLMKeys()
	var written yaml.Node
	if err := yaml.Unmarshal(mustMarshal(t, b), &written); err != nil {
		t.Fatal(err)
	}
	have := map[string]bool{}
	for i := 0; i+1 < len(written.Content[0].Content); i += 2 {
		have[written.Content[0].Content[i].Value] = true
		if g := written.Content[0].Content[i].Value; nested[g] != nil {
			for j := 0; j+1 < len(written.Content[0].Content[i+1].Content); j += 2 {
				have[g+"."+written.Content[0].Content[i+1].Content[j].Value] = true
			}
		}
	}
	for _, k := range top {
		if !have[k] {
			t.Errorf("every-field.yaml states every key, but the encoded block lacks %s", k)
		}
	}
	for g, leaves := range nested {
		for _, l := range leaves {
			if !have[g+"."+l] {
				t.Errorf("the encoded block lacks %s.%s", g, l)
			}
		}
	}
}

package contracts

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
)

func withoutEngine(pools []deployment.Pool) []deployment.Pool {
	out := append([]deployment.Pool(nil), pools...)
	for i := range out {
		out[i].Engine = deployment.EngineBlock{}
	}
	return out
}

func exampleResult(t *testing.T) *result.Result {
	t.Helper()
	r, err := LoadResult(filepath.Join("..", "docs", "contracts", "examples", "result.json"))
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// TestResultRoundTripsInBothFormats: a Result written by result.Marshal, in either format,
// reads back equal, the engine block a pool holds unopened included, and writing it again
// gives the same bytes.
func TestResultRoundTripsInBothFormats(t *testing.T) {
	r := exampleResult(t)
	// A component-owned block is an opaque map, so a value keeps its type only if the
	// encoding does.
	r.Engine = map[string]any{"colocated": map[string]any{"final": map[string]any{
		"vllm:kv_cache_usage_perc": 0.31, "vllm:num_preemptions_total": 3, "note": "x\ty"}}}
	if r.Input.Deployment.Pools[0].Engine.IsZero() {
		t.Fatal("the example's pool must carry an engine block for this test to mean anything")
	}
	for _, f := range []result.Format{result.FormatJSON, result.FormatYAML} {
		t.Run(string(f), func(t *testing.T) {
			b, err := result.Marshal(r, f)
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(t.TempDir(), "r."+string(f))
			if err := os.WriteFile(path, b, 0o644); err != nil {
				t.Fatal(err)
			}
			back, err := LoadResult(path)
			if err != nil {
				t.Fatalf("LoadResult refused what Marshal wrote: %v\n%s", err, b)
			}
			if !back.Input.Deployment.Pools[0].Engine.Equal(r.Input.Deployment.Pools[0].Engine) {
				t.Error("the pool's engine block did not survive the round trip")
			}
			// An engine block holds a node with read positions; compare the rest with it
			// cleared, on copies of the pools.
			a, z := *r, *back
			a.Input.Deployment.Pools = withoutEngine(r.Input.Deployment.Pools)
			z.Input.Deployment.Pools = withoutEngine(back.Input.Deployment.Pools)
			if !reflect.DeepEqual(a, z) {
				t.Errorf("the document read back differs from the one written:\n%s", b)
			}
			again, err := result.Marshal(back, f)
			if err != nil || string(again) != string(b) {
				t.Errorf("writing the read-back document gave other bytes (err %v)", err)
			}
			if f == result.FormatJSON && !strings.Contains(string(b), `"input": {`) {
				t.Errorf("JSON keys are not the schema's keys:\n%.300s", b)
			}
		})
	}
}

// TestLoadResultReadsJSONEscapes: a JSON document is read by JSON's rules, so an escape
// JSON allows and YAML does not reads as JSON defines it.
func TestLoadResultReadsJSONEscapes(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "docs", "contracts", "examples", "result.json"))
	if err != nil {
		t.Fatal(err)
	}
	src := `"ref": "llama-3.1-8b-h100-tp2-dp4.c64-s1.requests.csv"`
	if !strings.Contains(string(b), src) {
		t.Fatal("the example's per_request ref moved; update this test")
	}
	b = []byte(strings.Replace(string(b), src, `"ref": "out\/x.csv"`, 1))
	path := filepath.Join(t.TempDir(), "r.json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	r, err := LoadResult(path)
	if err != nil {
		t.Fatalf("a legal JSON escape was refused: %v", err)
	}
	if got := r.Artifacts["per_request"].Ref; got != "out/x.csv" {
		t.Errorf("ref = %q, want out/x.csv", got)
	}
	// A string in JSON stays a string, whatever YAML would make of its text.
	b = []byte(strings.Replace(string(b), `"name": "llama-3.1-8b-h100-tp2-dp4.c64-s1"`, `"name": "null"`, 1))
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	if r, err := LoadResult(path); err != nil || r.Name != "null" {
		t.Errorf("a JSON string read as another type: %v %q", err, r.Name)
	}
}

// TestLoadResultReadsTheExampleAsYAML: the YAML form of the example loads to the same
// document and validates.
func TestLoadResultReadsTheExampleAsYAML(t *testing.T) {
	r := exampleResult(t)
	b, err := result.Marshal(r, result.FormatYAML)
	if err != nil {
		t.Fatal(err)
	}
	if strings.HasPrefix(strings.TrimSpace(string(b)), "{") {
		t.Fatal("the YAML form must not be JSON, or this tests the JSON path again")
	}
	path := filepath.Join(t.TempDir(), "r.yaml")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	y, err := LoadResult(path)
	if err != nil {
		t.Fatal(err)
	}
	requireSilent(t, "result.yaml", Validate(Bundle{Result: y}))
}

// TestBundleReportsAnInvalidResult: Validate merges the Result's findings under result.
func TestBundleReportsAnInvalidResult(t *testing.T) {
	r := exampleResult(t)
	r.Requests.Completed--
	rep := Validate(Bundle{Result: r})
	if rep.Field.OK() {
		t.Fatal("a result whose ledger does not close passed")
	}
	for _, p := range rep.Field.Errors() {
		if p.Path == "result.requests" && strings.Contains(p.Message, "INV-1") {
			return
		}
	}
	t.Errorf("no finding at result.requests:\n%s", rep.Field.Error())
}

// TestBundleSaysAResultsInputWasNotRuleChecked: with rules given and only a Result, the
// rule layer says the embedded input was not rule-checked rather than stay silent; given
// the embedded documents too, the rules run on them.
func TestBundleSaysAResultsInputWasNotRuleChecked(t *testing.T) {
	r := exampleResult(t)
	rep := Validate(Bundle{Result: r, Rules: openAnyEngine})
	found := false
	for _, p := range rep.Rule.All() {
		found = found || p.Rule == "result-input-rules"
	}
	if !found {
		t.Errorf("rules given with only a result produced no result-input-rules finding:\n%s", rep.Rule.Error())
	}
	rep = Validate(Bundle{Result: r, Rules: openAnyEngine,
		Scenario: &r.Input.Scenario, Deployment: &r.Input.Deployment})
	for _, p := range rep.Rule.All() {
		if p.Rule == "result-input-rules" {
			t.Errorf("the embedded input was passed and still reported unchecked: %s", p)
		}
	}
	if rep.RulesApplied == "" {
		t.Error("no rules ran on the embedded input")
	}
}

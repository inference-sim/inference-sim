// Original to blis-vllm.

package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	sim "github.com/inference-sim/blis-vllm/compat/engine"
)

// The InferenceX DeepSeek-V4.1-Flash H200 rows (vLLM v0.33.0, agentic, DSpark) state three settings the
// engine's field list once refused: the Rust front end (env VLLM_USE_RUST_FRONTEND), synthetic rejection
// sampling with a supplied acceptance length, and a container variable (PYTHONUNBUFFERED). The scenario
// below is that recipe's shape; each test changes one of the three and reads what the engine prices.

const h200Recipe = "deepseek-v41-flash-h200-tp8-dspark5-synthetic-rust.yaml"

const (
	rustLine    = `env: {VLLM_USE_RUST_FRONTEND: "1", PYTHONUNBUFFERED: "1"}`
	specSynth   = `speculative_config: {method: dspark, num_speculative_tokens: 5, rejection_sample_method: synthetic, synthetic_acceptance_length: 3.51}`
	specBlock   = `speculative_config: {method: dspark, num_speculative_tokens: 5, rejection_sample_method: block}`
	pinThreeFiv = "acceptance:\n  value:\n    num_spec_tokens: 5\n    acceptance_length: 3.51\n    source: \"the run's server log\"\n"
)

// h200Variant opens the recipe with each engine-block line in replace swapped for its replacement
// (the key is the line's text, without indentation), and pin, if not empty, added to the scenario.
func h200Variant(t *testing.T, replace map[string]string, pin string) (*Model, Shape, error) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "vllmengine", "testdata", h200Recipe))
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	for old, repl := range replace {
		if !strings.Contains(body, "      "+old+"\n") {
			t.Fatalf("the recipe has no engine line %q", old)
		}
		if repl == "" {
			body = strings.Replace(body, "      "+old+"\n", "", 1)
		} else {
			body = strings.Replace(body, "      "+old+"\n", "      "+repl+"\n", 1)
		}
	}
	if pin != "" {
		body = strings.Replace(body, "model: deepseek-v4.1-flash\n", "model: deepseek-v4.1-flash\n"+pin, 1)
	}
	r := repos()
	r.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(r.Scenarios, "s.yaml"), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	sh, err := ShapeOf("s.yaml", r)
	if err != nil {
		return nil, sh, err
	}
	m, err := Open("s.yaml", r)
	return m, sh, err
}

func mustOpenH200(t *testing.T, replace map[string]string, pin string) (*Model, Shape) {
	t.Helper()
	m, sh, err := h200Variant(t, replace, pin)
	if err != nil {
		t.Fatalf("%v: %v", replace, err)
	}
	return m, sh
}

// The Rust front end tokenizes through fastokens, so a pool on it takes the registry's Rust admission rate
// where a pool on the Python server takes the Python one. A 290,000-token prompt (the mean input of the
// rows at one client) is admitted in a fraction of the time. Stating the variable as "0" is the Python path,
// and the point's evidence names the Rust rate as an assumption: it is a vendor figure applied to the
// Python-path value, not a measurement (the registry's own method for it).
func TestTheRustFrontendIsPricedAtItsAdmissionRate(t *testing.T) {
	const prompt = 290000
	rust, _ := mustOpenH200(t, nil, "")
	unset, _ := mustOpenH200(t, map[string]string{rustLine: `env: {PYTHONUNBUFFERED: "1"}`}, "")
	off, _ := mustOpenH200(t, map[string]string{rustLine: `env: {VLLM_USE_RUST_FRONTEND: "0"}`}, "")
	r, u, o := rust.Kernel().AdmissionOverhead(prompt), unset.Kernel().AdmissionOverhead(prompt), off.Kernel().AdmissionOverhead(prompt)
	if u != o {
		t.Errorf("an unstated front end admits in %v and VLLM_USE_RUST_FRONTEND=0 in %v: both are the Python server", u, o)
	}
	if r <= 0 || r*2 >= u {
		t.Errorf("the Rust front end admits a %d-token prompt in %v and the Python server in %v: the stated variable does not reach the price", prompt, r, u)
	}
	assumed := false
	for _, a := range rust.Provenance().Assumptions {
		assumed = assumed || a.Name == "host_admission_per_token_rust_frontend"
	}
	var method string
	for _, o := range rust.Kernel().Provenance() {
		if o.Name == "host_admission_per_token_rust_frontend" {
			method = o.Method
		}
	}
	if !assumed && method != "assumed" {
		t.Errorf("the Rust rate is neither a recorded assumption nor of method assumed (method %q): the point would present it as measured", method)
	}
}

// rejection_sample_method synthetic fixes the run's mean acceptance length, which vLLM takes from
// synthetic_acceptance_length. With the length stated the engine accepts the field and the supplied
// acceptance is the length: the shape carries 3.51 at five drafts, and the pool prices a step as the
// same recipe with the length pinned by the scenario and the sampler left at its recorded "block". (Without
// the length vLLM refuses the setting, and so does the engine factory, by name: see vllmengine's recipe tests.)
func TestSyntheticRejectionSamplingTakesTheSuppliedAcceptance(t *testing.T) {
	m, sh := mustOpenH200(t, nil, "")
	a := sh.Acceptance
	if a == nil || a.Value == nil || a.Value.AcceptanceLength == nil || *a.Value.AcceptanceLength != 3.51 || a.Value.NumSpecTokens != 5 {
		t.Fatalf("the shape's acceptance is %+v, want a length of 3.51 at 5 drafts", a)
	}
	pinned, psh := mustOpenH200(t, map[string]string{specSynth: specBlock}, pinThreeFiv)
	if psh.Acceptance == nil || *psh.Acceptance.Value.AcceptanceLength != 3.51 {
		t.Fatalf("the pinned variant's acceptance is %+v", psh.Acceptance)
	}
	batch := func() []*sim.Request { return []*sim.Request{decodeReq(290000, 290100), decodeReq(120000, 120050)} }
	if got, want := m.StepTime(batch()), pinned.StepTime(batch()); got != want {
		t.Errorf("a step prices at %d with the length stated by the engine block and %d with it pinned by the scenario", got, want)
	}
}

// PYTHONUNBUFFERED is proven inert: a recipe that sets it prices a step and an admission exactly as one
// that does not.
func TestAContainerVariableThePythonInterpreterReadsPricesNothing(t *testing.T) {
	with, _ := mustOpenH200(t, nil, "")
	without, _ := mustOpenH200(t, map[string]string{rustLine: `env: {VLLM_USE_RUST_FRONTEND: "1"}`}, "")
	batch := func() []*sim.Request { return []*sim.Request{decodeReq(290000, 290100), decodeReq(1024, 1030)} }
	if a, b := with.StepTime(batch()), without.StepTime(batch()); a != b {
		t.Errorf("a step prices at %d with PYTHONUNBUFFERED and %d without", a, b)
	}
	if a, b := with.Kernel().AdmissionOverhead(290000), without.Kernel().AdmissionOverhead(290000); a != b {
		t.Errorf("admission is %v with PYTHONUNBUFFERED and %v without", a, b)
	}
}

package cmd

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/acceptance"
)

// runAcceptanceCmd executes `blis <args...>` against the acceptance fixture registry in a
// re-exec leg (the command's flags are package state), returning stdout, stderr and the exit
// error.
func runAcceptanceCmd(t *testing.T, registry, catalog string, args ...string) (string, string, error) {
	t.Helper()
	full := append(append([]string{}, args...), "--catalog", catalog)
	if args[0] == "acceptance" {
		full = append(full, "--registry", registry)
	}
	c := exec.Command(os.Args[0], "-test.run=^TestKernelCLILeg$")
	c.Env = append(os.Environ(), kernelCLIArgsEnv+"="+strings.Join(full, "\x1f"))
	var out, errBuf bytes.Buffer
	c.Stdout, c.Stderr = &out, &errBuf
	err := c.Run()
	return out.String(), errBuf.String(), err
}

var dsparkLookup = []string{"acceptance", "--model", "deepseek-v4.1-flash", "--method", "dspark",
	"--category", "coding", "--thinking", "off", "--json"}

func lookupArgs(extra ...string) []string {
	return append(append([]string{}, dsparkLookup...), extra...)
}

func decodeLookup(t *testing.T, stdout string) acceptanceLookup {
	t.Helper()
	var l acceptanceLookup
	dec := json.NewDecoder(strings.NewReader(stdout))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&l); err != nil {
		t.Fatalf("the output is not an acceptance lookup: %v\n%s", err, stdout)
	}
	return l
}

// A request the registry has a profile for resolves the exact key; the value is the profile's own, and the output names the layer, set and profile
// the way a run's record does.
func TestAcceptanceCommandResolvesTheExactProfile(t *testing.T) {
	registry, catalog := acceptanceRepos(t, "")
	stdout, stderr, err := runAcceptanceCmd(t, registry, catalog, lookupArgs("--num-spec-tokens", "5")...)
	if err != nil {
		t.Fatalf("the lookup was refused: %v\n%s", err, stderr)
	}
	l := decodeLookup(t, stdout)
	if l.Step != string(acceptance.StepExact) || len(l.Differed) != 0 {
		t.Errorf("step %q differing in %v, want exact with nothing differing", l.Step, l.Differed)
	}
	if want := fixtureAL(t, 5, acceptance.ThinkingOff); l.AcceptanceLength != want {
		t.Errorf("acceptance length %v, want the fixture's %v", l.AcceptanceLength, want)
	}
	if l.Layer != "registry" || l.Set != "inferencex-golden-al" ||
		l.Profile != "deepseek-v4.1-flash/dspark/k5/thinking-off/coding" {
		t.Errorf("layer %q, set %q, profile %q: not the fixture's profile", l.Layer, l.Set, l.Profile)
	}
}

// A K the registry has no profile for borrows the closest, and the output states the step, what
// differed and how K was brought to the request's: the lookup is never a bare number.
func TestAcceptanceCommandStatesTheBorrowingStep(t *testing.T) {
	registry, catalog := acceptanceRepos(t, "")
	stdout, stderr, err := runAcceptanceCmd(t, registry, catalog, lookupArgs("--num-spec-tokens", "7")...)
	if err != nil {
		t.Fatalf("the lookup was refused: %v\n%s", err, stderr)
	}
	l := decodeLookup(t, stdout)
	if l.Step != string(acceptance.StepOtherK) || !reflect.DeepEqual(l.Differed, []string{"num_spec_tokens"}) || l.KRule == "" {
		t.Errorf("step %q differing in %v with K rule %q, want same_drafter_other_k differing in num_spec_tokens with a K rule",
			l.Step, l.Differed, l.KRule)
	}
	if l.NumSpecTokens != 7 || !strings.Contains(l.Profile, "/k5/") {
		t.Errorf("request K %d resolved through profile %s, want K 7 through the K=5 profile", l.NumSpecTokens, l.Profile)
	}
}

// The command runs the resolution a run does: for the same selectors and K the lookup and a
// run's record agree on the acceptance length and the profile.
func TestAcceptanceCommandAgreesWithARun(t *testing.T) {
	registry, catalog := acceptanceRepos(t, "")
	for _, k := range []int{3, 5} {
		rec, _, stderr, err := runAccept(t, dsparkVariant(t, k, ""), registry, catalog,
			append(append([]string{}, acceptRunArgs...), "--speculative-thinking", "off", "--speculative-category", "coding")...)
		if err != nil {
			t.Fatalf("K=%d: the run was refused: %v\n%s", k, err, lastLines(stderr, 4))
		}
		stdout, cstderr, err := runAcceptanceCmd(t, registry, catalog,
			lookupArgs("--num-spec-tokens", strconv.Itoa(k), "--engine", "vllm")...)
		if err != nil {
			t.Fatalf("K=%d: the lookup was refused: %v\n%s", k, err, cstderr)
		}
		l := decodeLookup(t, stdout)
		if l.AcceptanceLength != rec.Length || rec.Entry.Key == "" || l.Profile != rec.Entry.Key {
			t.Errorf("K=%d: lookup %v via %s, run %v via %+v", k, l.AcceptanceLength, l.Profile, rec.Length, rec.Entry)
		}
	}
}

// A request nothing matches is refused with the resolver's reason and a non-zero exit, as are
// a missing required flag and a name outside the vocabulary: no number is invented.
func TestAcceptanceCommandRefusals(t *testing.T) {
	registry, catalog := acceptanceRepos(t, "")
	for _, tt := range []struct {
		name string
		args []string
		want string
	}{
		{"exact-only with no exact profile", lookupArgs("--num-spec-tokens", "7", "--exact-only"), "exact_only forbids borrowing"},
		{"a category no profile has", lookupArgs("--num-spec-tokens", "5", "--category", "chat"), "no fallback_category was requested"},
		{"no category", []string{"acceptance", "--model", "deepseek-v4.1-flash", "--method", "dspark",
			"--thinking", "off", "--num-spec-tokens", "5"}, "--category is required"},
		{"no num-spec-tokens", lookupArgs(), "--num-spec-tokens is required"},
		{"thinking outside the vocabulary", []string{"acceptance", "--model", "deepseek-v4.1-flash", "--method", "dspark",
			"--category", "coding", "--thinking", "maybe", "--num-spec-tokens", "5"}, "--thinking \"maybe\""},
		{"a model not in the catalog", []string{"acceptance", "--model", "no-such-model", "--method", "dspark",
			"--category", "coding", "--thinking", "off", "--num-spec-tokens", "5"}, "no-such-model"},
		{"sampling without the engine", lookupArgs("--num-spec-tokens", "5", "--draft-sample-method", "greedy"), "--engine"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			stdout, stderr, err := runAcceptanceCmd(t, registry, catalog, tt.args...)
			if err == nil {
				t.Fatalf("the lookup succeeded:\n%s", stdout)
			}
			if !strings.Contains(stderr, tt.want) {
				t.Errorf("stderr does not say %q:\n%s", tt.want, lastLines(stderr, 4))
			}
			if strings.TrimSpace(stdout) != "" {
				t.Errorf("a refused lookup wrote to stdout:\n%s", stdout)
			}
		})
	}
}

// A separate drafter is named as catalog resolve lists it, and the lookup carries it in the request.
func TestAcceptanceCommandTakesAListedDrafter(t *testing.T) {
	registry, catalog := acceptanceRepos(t, "")
	stdout, stderr, err := runAcceptanceCmd(t, registry, catalog, "acceptance", "--model", "minimax-m3",
		"--drafter", "minimax-m3-eagle3", "--method", "eagle3", "--num-spec-tokens", "3",
		"--category", "coding", "--thinking", "off", "--engine", "vllm", "--json")
	if err != nil {
		t.Fatalf("the lookup was refused: %v\n%s", err, stderr)
	}
	l := decodeLookup(t, stdout)
	if l.Drafter != "minimax-m3-eagle3" || !strings.HasPrefix(l.Profile, "minimax-m3/eagle3/k3/") {
		t.Errorf("drafter %q resolved profile %s", l.Drafter, l.Profile)
	}
}

// catalog resolve lists a model's drafters with each one's method and draft length, in the
// entry's order: the in-graph speculator, then each separate drafter's own graph's.
func strPtr(s string) *string { return &s }

func TestCatalogResolveListsDrafters(t *testing.T) {
	data := testData()
	for _, tt := range []struct {
		model string
		want  []drafterInfo
	}{
		{"deepseek-v4.1-flash", []drafterInfo{{Name: "speculator", Method: strPtr("dspark"), NumSpecTokens: intPtr(5)}}},
		{"minimax-m3", []drafterInfo{
			{Name: "speculator", Method: strPtr("mtp"), NumSpecTokens: intPtr(7)},
			{Name: "minimax-m3-eagle3", Method: strPtr("eagle3"), NumSpecTokens: intPtr(1)},
			{Name: "minimax-m3-eagle3-gqa", Method: strPtr("eagle3"), NumSpecTokens: intPtr(1)},
		}},
		{"kimi-k3", []drafterInfo{{Name: "kimi-k3-dspark", Method: strPtr("dspark"), NumSpecTokens: intPtr(7)}}},
	} {
		got, err := draftersOf(data, tt.model)
		if err != nil {
			t.Fatalf("%s: %v", tt.model, err)
		}
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("%s drafters %+v, want %+v", tt.model, got, tt.want)
		}
	}
	// every listed name is one the acceptance command can be given, and every one whose graph the
	// catalog ships states a method and a positive draft length.
	for _, name := range data.Models() {
		drafters, err := draftersOf(data, name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for _, d := range drafters {
			if d.Method == nil || *d.Method == "" || d.NumSpecTokens == nil || *d.NumSpecTokens < 1 {
				t.Errorf("%s lists drafter %q with method %v and K %v", name, d.Name, d.Method, d.NumSpecTokens)
			}
		}
	}
}

// The field is absent from a chip's and an unresolved model's output, so the three keys those
// carry are unchanged.
func TestCatalogResolveJSONHasDraftersOnlyForAModelThatListsThem(t *testing.T) {
	registry, catalog := acceptanceRepos(t, "")
	for _, tt := range []struct {
		args    []string
		drafter bool
	}{
		{[]string{"catalog", "resolve", "--model", "deepseek-v4.1-flash", "--json"}, true},
		{[]string{"catalog", "resolve", "--model", "no-such-model", "--json"}, false},
		{[]string{"catalog", "resolve", "--chip", "h200", "--json"}, false},
	} {
		stdout, stderr, err := runAcceptanceCmd(t, registry, catalog, tt.args...)
		if err != nil {
			t.Fatalf("%v: %v\n%s", tt.args, err, stderr)
		}
		var m map[string]json.RawMessage
		if err := json.Unmarshal([]byte(stdout), &m); err != nil {
			t.Fatal(err)
		}
		if _, has := m["drafters"]; has != tt.drafter {
			t.Errorf("%v: drafters present = %t, want %t\n%s", tt.args, has, tt.drafter, stdout)
		}
	}
}

// The engine block's sampling settings reach the request: the same lookup under an engine's
// stated draft sampler and verifier differs in them from a profile that states neither, and
// an unknown sampler is refused rather than looked up.
func TestAcceptanceCommandCarriesTheEngineSampling(t *testing.T) {
	registry, catalog := acceptanceRepos(t, "")
	stdout, stderr, err := runAcceptanceCmd(t, registry, catalog, lookupArgs("--num-spec-tokens", "5", "--engine", "vllm",
		"--draft-sample-method", "probabilistic", "--rejection-sample-method", "block")...)
	if err != nil {
		t.Fatalf("the lookup was refused: %v\n%s", err, stderr)
	}
	l := decodeLookup(t, stdout)
	if l.Engine != "vllm" || l.Step == string(acceptance.StepExact) || !slices.Contains(l.Differed, "draft_sampler") {
		t.Errorf("engine %q step %q differing in %v, want a borrow differing in draft_sampler", l.Engine, l.Step, l.Differed)
	}
	if _, _, err := runAcceptanceCmd(t, registry, catalog, lookupArgs("--num-spec-tokens", "5", "--engine", "vllm",
		"--draft-sample-method", "wild")...); err == nil {
		t.Error("an unknown draft sampler was looked up")
	}
}

// Moved from blis-latency-kernel/prefill_forms_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/coefficient"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// contractEntries are blis-registry PR #47's DeepSeek-V4.1-Flash h200 entries as
// docs/COEFF_CONTRACT.md v2.1 states them, model-scoped as the registry scopes them.
func contractEntries(families ...string) []coefficient.Entry {
	all := map[string][]coefficient.Entry{
		"sparse_mla": {
			{Name: "attention_prefill_floor_sparse_mla", Value: 14.8, Units: "us_per_transfer"},
			{Name: "attention_prefill_per_request_sparse_mla", Value: 1.323, Units: "us_per_request"},
			{Name: "attention_prefill_per_token_sparse_mla", Value: 0.049, Units: "us_per_token"},
			{Name: "attention_prefill_rate_sparse_mla", Value: 3159685, Units: "bytes_per_us"},
		},
		"block_index_scores": {
			{Name: "attention_prefill_floor_block_index_scores", Value: 4.1, Units: "us_per_transfer"},
			{Name: "attention_prefill_per_token_block_index_scores", Value: 0.00466, Units: "us_per_token"},
			{Name: "attention_prefill_rate_block_index_scores", Value: 7298637, Units: "bytes_per_us"},
		},
		"mhc": {
			{Name: "mhc_floor_pre", Value: 8.9, Units: "us_per_transfer"},
			{Name: "mhc_rate_pre", Value: 1799384, Units: "bytes_per_us"},
			{Name: "mhc_floor_post", Value: 3.0, Units: "us_per_transfer"},
			{Name: "mhc_rate_post", Value: 1850611, Units: "bytes_per_us"},
		},
		"host": {
			{Name: "host_admission_per_token", Value: 2.67, Units: "us_per_token", Method: "measured"},
			{Name: "host_admission_per_request", Value: 0, Units: "us_per_request", Method: "not_charged"},
		},
	}
	var out []coefficient.Entry
	for _, f := range families {
		for _, e := range all[f] {
			if e.Method == "" {
				e.Method = "assumed"
			}
			if e.Method != "measured" {
				e.Rationale = "test"
			} else {
				e.Sources = []coefficient.Source{{Kind: "discussion", Cite: "test", Role: "primary"}}
			}
			e.Scope = coefficient.Scope{Hardware: []string{"h200"},
				Model: []string{"deepseek-v4.1-flash"}}
			out = append(out, e)
		}
	}
	return out
}

// contractNames is every entry name contractEntries can return, and the Rust-frontend
// admission rate TestTheFrontendChoosesTheAdmissionRate supplies itself.
func contractNames() map[string]bool {
	names := map[string]bool{"host_admission_per_token_rust_frontend": true}
	for _, e := range contractEntries("sparse_mla", "block_index_scores", "mhc", "host") {
		names[e.Name] = true
	}
	return names
}

// withoutContractEntries is in with every contractEntries name removed from its sets, so a
// test controls those families itself. blis-registry v0.2.0 commits them for V4.1 on h200;
// without this the "without" side of every comparison below would carry them.
func withoutContractEntries(in Inputs) Inputs { return withoutEntries(in, contractNames()) }

// withoutEntries is in with every entry whose name is in names removed from its sets.
func withoutEntries(in Inputs, names map[string]bool) Inputs {
	var sets []*coefficient.Set
	for _, set := range in.Coefficients {
		c := *set
		c.Coefficients = nil
		for _, e := range set.Coefficients {
			if !names[e.Name] {
				c.Coefficients = append(c.Coefficients, e)
			}
		}
		sets = append(sets, &c)
	}
	in.Coefficients = sets
	return in
}

// withEntries is the V4.1 fixture, less the registry's contract entries, with entries
// appended as their own set, less any entry named in drop.
func withEntries(t *testing.T, entries []coefficient.Entry, drop ...string) *Kernel {
	t.Helper()
	in := withoutContractEntries(fixtureInputs(t, v41))
	var kept []coefficient.Entry
	for _, e := range entries {
		dropped := false
		for _, d := range drop {
			dropped = dropped || e.Name == d
		}
		if !dropped {
			kept = append(kept, e)
		}
	}
	if len(kept) > 0 {
		in.Coefficients = append(append([]*coefficient.Set{}, in.Coefficients...),
			&coefficient.Set{Kind: "CoefficientSet", Name: "test-v41", Coefficients: kept})
	}
	return build(t, in)
}

// chunkOn is one prefill chunk of sched tokens on prefix computed tokens of a longer prompt.
func chunkOn(sched, prefix int) kernel.Batch {
	return kernel.Batch{DecodeThreshold: 8, Reqs: []kernel.ReqShape{
		{Scheduled: sched, Computed: prefix, PromptLen: prefix + sched + 10000}}}
}

// A FAMILY IS ALL OR NONE, AND ITS ABSENCE PRICES AS BEFORE. Each family with one member
// dropped prices every step exactly as with no entries at all, and Provenance records the
// fallback; with every entry absent, the rate's absence is recorded as optional records it.
func TestAPartialFamilyFallsBackAndSaysSo(t *testing.T) {
	none := withEntries(t, nil)
	steps := []kernel.Batch{decodeBatch(4, 6, 90000), chunkOn(8192, 81808),
		chunkOn(1808, 88192)}
	for _, c := range []struct{ family, drop, record string }{
		{"sparse_mla", "attention_prefill_per_request_sparse_mla", "attention_prefill_sparse_mla"},
		{"block_index_scores", "attention_prefill_floor_block_index_scores",
			"attention_prefill_block_index_scores"},
		{"mhc", "mhc_floor_pre", "mhc_pre"},
		{"mhc", "mhc_rate_post", "mhc"},
	} {
		k := withEntries(t, contractEntries(c.family), c.drop)
		for _, b := range steps {
			if a, z := k.StepTime(b), none.StepTime(b); a.NoOverlap != z.NoOverlap ||
				a.Overlap != z.Overlap {
				t.Errorf("%s without %s: a %d-token step prices %v, want the absent "+
					"family's %v", c.family, c.drop, b.Tokens(), a.NoOverlap, z.NoOverlap)
			}
		}
		if !hasFallback(k, c.record) {
			t.Errorf("%s without %s: no fallback recorded under %s", c.family, c.drop,
				c.record)
		}
	}
	for _, name := range []string{"attention_prefill_rate_sparse_mla",
		"attention_prefill_rate_block_index_scores", "mhc_rate_pre", "mhc_rate_post"} {
		if !hasAssumption(none, name) {
			t.Errorf("%s is absent and its absence is not recorded", name)
		}
	}
}

func hasFallback(k *Kernel, name string) bool {
	for _, o := range k.Provenance() {
		if o.Name == name && strings.HasPrefix(o.Scope, "fallback") {
			return true
		}
	}
	return false
}

// THE CONTRACT'S HOST ENTRIES RESOLVE. A per-request admission of zero is accepted where it
// is not_charged, and the model's per-token cost replaces the part-wide one: admission is
// 2.67 us a prompt token and nothing a request.
func TestTheContractsAdmissionEntriesResolve(t *testing.T) {
	k := withEntries(t, contractEntries("host"))
	if got, want := k.AdmissionOverhead(90000),
		time.Duration(2.67*90000*float64(time.Microsecond)); got != want {
		t.Errorf("admission of a 90,000-token prompt %v, want %v", got, want)
	}
}

// THE FRONTEND SETS THE ADMISSION RATE. A pool on vLLM's Rust frontend (engine.frontend:
// rust) tokenizes through fastokens, so it takes the registry's Rust rate; the default
// Python frontend keeps the Python one; and a Rust pool with no Rust rate keeps the Python
// one and says so in Provenance.
func TestTheFrontendChoosesTheAdmissionRate(t *testing.T) {
	rust := coefficient.Entry{Name: "host_admission_per_token_rust_frontend", Value: 0.267,
		Units: "us_per_token", Method: "assumed", Rationale: "test",
		Scope: coefficient.Scope{Hardware: []string{"h200"}}}
	admit := func(frontend engineconfig.Frontend, extra ...coefficient.Entry) (*Kernel, time.Duration) {
		t.Helper()
		in := withoutContractEntries(fixtureInputs(t, v41))
		editEngine(t, &in, func(e *engineconfig.Engine) { e.Frontend = frontend })
		entries := append(contractEntries("host"), extra...)
		in.Coefficients = append(append([]*coefficient.Set{}, in.Coefficients...),
			&coefficient.Set{Kind: "CoefficientSet", Name: "test-v41", Coefficients: entries})
		k := build(t, in)
		return k, k.AdmissionOverhead(90000)
	}
	us := func(v float64) time.Duration { return time.Duration(v * 90000 * float64(time.Microsecond)) }
	if _, got := admit("", rust); got != us(2.67) {
		t.Errorf("python frontend: admission %v, want the Python rate's %v", got, us(2.67))
	}
	if _, got := admit(engineconfig.FrontendRust, rust); got != us(0.267) {
		t.Errorf("rust frontend: admission %v, want the Rust rate's %v", got, us(0.267))
	}
	k, got := admit(engineconfig.FrontendRust)
	if got != us(2.67) {
		t.Errorf("rust frontend, no Rust rate: admission %v, want the Python rate's %v", got,
			us(2.67))
	}
	recorded := false
	for _, o := range k.Provenance() {
		recorded = recorded || (o.Name == rust.Name && o.Set == KernelAssumptionSet)
	}
	if !recorded {
		t.Error("a Rust pool priced at the Python rate is not recorded as an assumption")
	}
}

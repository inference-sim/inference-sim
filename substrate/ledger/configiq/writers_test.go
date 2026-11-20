package configiq_test

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/ledger/configiq"
)

func ident(set map[string]any) map[string]any {
	id := map[string]any{"source": configiq.SourceID, "request_type": "", "prefix_caching": "", "tp": int64(2),
		"max_seq_len": nil, "gpu_memory_utilization": nil, "accelerator": "H200"}
	for k, v := range set {
		id[k] = v
	}
	return id
}

func TestWriterFamilyByRequestType(t *testing.T) {
	cases := []struct {
		name string
		set  map[string]any
		want configiq.Family
		fail bool
	}{
		{"forge chat", map[string]any{"request_type": "/v1/chat/completions", "prefix_caching": "yes"}, configiq.FamilyA, false},
		{"forge completions", map[string]any{"request_type": "/v1/completions", "prefix_caching": "yes"}, configiq.FamilyA, false},
		{"manual chat", map[string]any{"request_type": "chat_completions", "prefix_caching": "no"}, configiq.FamilyC, false},
		{"manual text, prefix caching blank", map[string]any{"request_type": "text_completions"}, configiq.FamilyC, false},
		{"blank request type", nil, configiq.FamilyD, false},
		{"blank request type, null", map[string]any{"request_type": nil}, configiq.FamilyD, false},
		{"a /v1/ type without prefix caching on is no family", map[string]any{"request_type": "/v1/completions", "prefix_caching": "no"}, "", true},
		{"a /v1/ type with prefix caching blank is no family", map[string]any{"request_type": "/v1/completions"}, "", true},
		{"an unknown request type is no family", map[string]any{"request_type": "embeddings"}, "", true},
		{"a non-string request type is refused", map[string]any{"request_type": int64(1)}, "", true},
	}
	for _, c := range cases {
		got, err := configiq.WriterFamily(ident(c.set))
		if (err != nil) != c.fail || got != c.want {
			t.Errorf("%s: got %q, %v; want %q, failure %v", c.name, got, err, c.want, c.fail)
		}
	}
}

// Every family has a rule, and only those three: a blank means "not passed" for forge's rows
// alone, a measurement the writer's records do not give is flagged as an estimate for family
// D alone, and every fill names its source.
func TestEveryFamilyHasARule(t *testing.T) {
	rs := configiq.Rules()
	if len(rs) != 3 {
		t.Fatalf("%d rules, want the three families A, C and D", len(rs))
	}
	for _, r := range rs {
		if r.Writer == "" || r.Signature == "" || r.ScoreOrder == 0 {
			t.Errorf("family %s: incomplete rule %+v", r.Family, r)
		}
		if (r.Blank == configiq.BlankNotPassed) != (r.Family == configiq.FamilyA) {
			t.Errorf("family %s reads a blank as %q", r.Family, r.Blank)
		}
		for _, f := range r.Fills {
			if f.Source == "" || f.Value == "" {
				t.Errorf("family %s: fill %+v has no value or source", r.Family, f)
			}
			if f.Estimated != (r.Family == configiq.FamilyD) {
				t.Errorf("family %s: fill %q estimated %v", r.Family, f.Field, f.Estimated)
			}
		}
		if r.UnscoredAtTP1 == (r.Family == configiq.FamilyA) {
			t.Errorf("family %s: UnscoredAtTP1 %v", r.Family, r.UnscoredAtTP1)
		}
	}
	a, c, d := configiq.RuleFor(configiq.FamilyA), configiq.RuleFor(configiq.FamilyC), configiq.RuleFor(configiq.FamilyD)
	if a == nil || c == nil || d == nil || a.ScoreOrder >= c.ScoreOrder || a.ScoreOrder >= d.ScoreOrder {
		t.Error("family A is scored first")
	}
	if configiq.RuleFor("B") != nil {
		t.Error("no family B was found")
	}
}

func TestUnscoredAtTP1(t *testing.T) {
	for _, c := range []struct {
		name string
		set  map[string]any
		want string
	}{
		{"forge at TP 1 is scored", map[string]any{"request_type": "/v1/completions", "prefix_caching": "yes", "tp": int64(1)}, ""},
		{"manual at TP 1", map[string]any{"request_type": "chat_completions", "tp": int64(1)}, "tp_1_may_use_data_parallelism"},
		{"earlier launcher at TP 1", map[string]any{"tp": int64(1)}, "tp_1_may_use_data_parallelism"},
		{"earlier launcher at TP 4", map[string]any{"tp": int64(4)}, ""},
	} {
		got, err := configiq.UnscoredReason(ident(c.set))
		if err != nil || got != c.want {
			t.Errorf("%s: %q, %v; want %q", c.name, got, err, c.want)
		}
	}
}

func TestSubstitutions(t *testing.T) {
	fields := func(set map[string]any) []string {
		fs, err := configiq.Substitutions(ident(set))
		if err != nil {
			t.Fatal(err)
		}
		var out []string
		for _, f := range fs {
			out = append(out, f.Field)
		}
		return out
	}
	if got := fields(map[string]any{"request_type": "/v1/completions", "prefix_caching": "yes"}); len(got) != 3 {
		t.Errorf("family A fills %v, want its three named substitutions", got)
	}
	if got := fields(map[string]any{"request_type": "chat_completions"}); len(got) != 0 {
		t.Errorf("family C fills %v, want none", got)
	}
	usual := map[string]any{"max_seq_len": "8192", "gpu_memory_utilization": "0.92"}
	if got := fields(usual); len(got) != 2 {
		t.Errorf("family D at the launcher's defaults fills %v, want prefix caching and duration", got)
	}
	recorded := map[string]any{"max_seq_len": "10000", "gpu_memory_utilization": "0.92", "prefix_caching": "yes"}
	if got := fields(recorded); len(got) != 1 || got[0] != "measurement duration" {
		t.Errorf("family D with prefix caching recorded fills %v, want the duration only", got)
	}
	for _, set := range []map[string]any{
		{"max_seq_len": "34816", "gpu_memory_utilization": "0.92"},
		{"max_seq_len": "8192", "gpu_memory_utilization": "0.9"},
		{"max_seq_len": nil, "gpu_memory_utilization": "0.92"},
	} {
		if got := fields(set); len(got) != 0 {
			t.Errorf("family D off the launcher's defaults (%v) fills %v, want none", set, got)
		}
	}
}

// On the pinned revision (21cc6eb, 1,702 units) every unit has a family. The counts are those
// of the committed unit list, which `ledger ingest-configiq -check` reproduces from the
// downloaded revision; they differ from the 779, 128 and 803 of §3C.1, which count the current
// revision.
func TestFamiliesOfThePinnedRevision(t *testing.T) {
	l := committed(t)
	units := l.Units[configiq.SourceID]
	byFamily := map[configiq.Family]int{}
	forgeByChip := map[string]int{}
	unscored := map[configiq.Family]int{}
	d := 0
	for _, u := range units {
		f, err := configiq.WriterFamily(u.Identity)
		if err != nil {
			t.Fatalf("unit %s: %v", u.UnitID, err)
		}
		byFamily[f]++
		if f == configiq.FamilyA {
			forgeByChip[u.Identity["accelerator"].(string)]++
		}
		if why, _ := configiq.UnscoredReason(u.Identity); why != "" {
			unscored[f]++
		}
		if fs, _ := configiq.Substitutions(u.Identity); f == configiq.FamilyD && len(fs) > 0 {
			d++
		}
	}
	want := map[configiq.Family]int{configiq.FamilyA: 776, configiq.FamilyC: 128, configiq.FamilyD: 798}
	for f, n := range want {
		if byFamily[f] != n {
			t.Errorf("family %s: %d units, want %d", f, byFamily[f], n)
		}
	}
	if len(byFamily) != len(want) || len(units) != 1702 {
		t.Errorf("families %v over %d units", byFamily, len(units))
	}
	if forgeByChip["H200"] != 205 || forgeByChip["B200"] != 168 || forgeByChip["MI355X"] != 403 {
		t.Errorf("family A by chip %v, want H200 205, B200 168, MI355X 403", forgeByChip)
	}
	if unscored[configiq.FamilyA] != 0 || unscored[configiq.FamilyC] != 48 || unscored[configiq.FamilyD] != 279 {
		t.Errorf("units unscored at TP 1: %v, want none of A, 48 of C, 279 of D", unscored)
	}
	if d != 284 {
		t.Errorf("%d family D units at the earlier launcher's defaults, want 284", d)
	}
}

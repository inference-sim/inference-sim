package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func writeSchedule(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "schedule.txt")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}

func TestParseLoRAPlacementScheduleReadsEntriesAndSkipsComments(t *testing.T) {
	path := writeSchedule(t, `# arm 3, L = 10 s, delta = 0

0 0=adapter_0,adapter_1;1=adapter_2
10000000 0=adapter_3;1=adapter_4,adapter_5
`)
	got, err := parseLoRAPlacementSchedule(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("want 2 entries, got %d: %#v", len(got), got)
	}
	if got[0].AtUs != 0 || got[1].AtUs != 10_000_000 {
		t.Errorf("timestamps: got %d and %d", got[0].AtUs, got[1].AtUs)
	}
	if ids := got[0].Placement[0]; len(ids) != 2 || ids[0] != "adapter_0" || ids[1] != "adapter_1" {
		t.Errorf("entry 0 instance 0: got %#v", ids)
	}
	if ids := got[1].Placement[1]; len(ids) != 2 || ids[0] != "adapter_4" || ids[1] != "adapter_5" {
		t.Errorf("entry 1 instance 1: got %#v", ids)
	}
}

// TestParseLoRAPlacementScheduleParsesAHarnessRenderedFixture pins the cross-language
// contract: this fixture body is the literal stdout of
//
//	python3 -c "
//	from harness.schedule import write_schedule
//	write_schedule('/tmp/sched.txt', [(0, {0: ['adapter_0','adapter_1'], 1: ['adapter_2']}),
//	                                  (10_000_000, {0: ['adapter_3']})], header='round-trip probe')
//	print(open('/tmp/sched.txt').read())"
//
// run against harness/schedule.py (lora-control, Spec 4 Slice B, Task 7). A Python renderer
// and a Go parser can drift on a shared grammar with every test on both sides still green;
// this is the only check that catches it.
func TestParseLoRAPlacementScheduleParsesAHarnessRenderedFixture(t *testing.T) {
	path := writeSchedule(t, `# round-trip probe
# <t_us> <idx=id[,id...];idx=id...>   -- BLIS --lora-placement-schedule
0 0=adapter_0,adapter_1;1=adapter_2
10000000 0=adapter_3
`)
	got, err := parseLoRAPlacementSchedule(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("want 2 entries, got %d: %#v", len(got), got)
	}
	if got[0].AtUs != 0 || got[1].AtUs != 10_000_000 {
		t.Errorf("timestamps: got %d and %d", got[0].AtUs, got[1].AtUs)
	}
	if ids := got[0].Placement[0]; len(ids) != 2 || ids[0] != "adapter_0" || ids[1] != "adapter_1" {
		t.Errorf("entry 0 instance 0: got %#v", ids)
	}
	if ids := got[0].Placement[1]; len(ids) != 1 || ids[0] != "adapter_2" {
		t.Errorf("entry 0 instance 1: got %#v", ids)
	}
	if ids := got[1].Placement[0]; len(ids) != 1 || ids[0] != "adapter_3" {
		t.Errorf("entry 1 instance 0: got %#v", ids)
	}
}

// Each rejected form gets its own case: the point of this parser is that NONE of them
// degrade into a silently different experiment.
func TestParseLoRAPlacementScheduleRejects(t *testing.T) {
	for _, tc := range []struct{ name, body, want string }{
		{"empty file", "", "no entries"},
		{"comments only", "# nothing here\n\n", "no entries"},
		{"no space separator", "0\n", "want \"<t_us> <placement>\""},
		{"non-integer timestamp", "0.5 0=adapter_0\n", "not an integer number of microseconds"},
		{"negative timestamp", "-1 0=adapter_0\n", "is negative"},
		{"equal timestamps", "0 0=a\n0 1=b\n", "does not increase"},
		{"decreasing timestamps", "10 0=a\n5 1=b\n", "does not increase"},
		{"unparseable placement", "0 not-a-placement\n", "placement schedule"},
		// The trailing space is stripped by the outer TrimSpace before Cut ever
		// runs, so this collapses to the same !found path as "no space separator"
		// above — a timestamp with no placement text at all, not an empty spec.
		{"timestamp followed by only whitespace", "0 \n", "want \"<t_us> <placement>\""},
		// parseLoRAAdapterPlacement returns an empty, non-nil map with no error for a
		// spec that is non-empty but parses to no chunks (all ';'-separated pieces are
		// blank) — e.g. ";". That reaches the len(placement) == 0 guard below the
		// delegated parse, distinct from the case above which is caught earlier by
		// the "<t_us> <placement>" format check.
		{"placement parses to nothing", "0 ;\n", "empty placement"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseLoRAPlacementSchedule(writeSchedule(t, tc.body))
			if err == nil {
				t.Fatalf("want an error for %q, got nil", tc.body)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not mention %q", err.Error(), tc.want)
			}
		})
	}
}

func TestParseLoRAPlacementScheduleRejectsAMissingFile(t *testing.T) {
	_, err := parseLoRAPlacementSchedule(filepath.Join(t.TempDir(), "absent.txt"))
	if err == nil {
		t.Fatal("a missing schedule must be an error, not an empty schedule")
	}
}

func TestValidateLoRAScheduleFlags(t *testing.T) {
	entry := func(at int64, p map[int][]string) sim.PlacementScheduleEntry {
		return sim.PlacementScheduleEntry{AtUs: at, Placement: p}
	}
	seeded := map[int][]string{0: {"a0"}, 1: {"a1"}}
	matching := []sim.PlacementScheduleEntry{
		entry(0, map[int][]string{0: {"a0"}, 1: {"a1"}}),
		entry(10_000_000, map[int][]string{0: {"a2"}}),
	}

	for _, tc := range []struct {
		name        string
		policy      string
		path        string
		schedule    []sim.PlacementScheduleEntry
		placement   map[int][]string
		wantErrPart string
	}{
		{name: "scheduled with a matching t=0 entry", policy: "scheduled", path: "s.txt",
			schedule: matching, placement: seeded},
		{name: "scheduled with a schedule starting after t=0", policy: "scheduled", path: "s.txt",
			schedule:  []sim.PlacementScheduleEntry{entry(10_000_000, map[int][]string{0: {"a2"}})},
			placement: seeded},
		{name: "gate-only policy with no schedule", policy: "pre-placement", placement: seeded},
		{name: "on-demand with nothing", policy: ""},

		{name: "scheduled without a schedule", policy: "scheduled",
			placement: seeded, wantErrPart: "requires --lora-placement-schedule"},
		{name: "schedule without scheduled", policy: "pre-placement", path: "s.txt",
			schedule: matching, placement: seeded,
			wantErrPart: "only the \"scheduled\" creation policy reads it"},
		{name: "t=0 entry disagrees on ids", policy: "scheduled", path: "s.txt",
			schedule:  []sim.PlacementScheduleEntry{entry(0, map[int][]string{0: {"a9"}, 1: {"a1"}})},
			placement: seeded, wantErrPart: "disagrees with --lora-adapter-placement"},
		{name: "t=0 entry disagrees on instances", policy: "scheduled", path: "s.txt",
			schedule:  []sim.PlacementScheduleEntry{entry(0, map[int][]string{0: {"a0"}})},
			placement: seeded, wantErrPart: "disagrees with --lora-adapter-placement"},
		// Fix round 1, Important 3: samePlacement's order-insensitivity was unguarded --
		// deleting both sort.Strings calls in samePlacement made no existing test fail.
		// This row pins the property the function exists to have: reordering ids WITHIN
		// one instance's list must still compare equal.
		{name: "t=0 entry agrees when ids are reordered within an instance", policy: "scheduled",
			path: "s.txt",
			schedule: []sim.PlacementScheduleEntry{
				entry(0, map[int][]string{0: {"a1", "a0"}, 1: {"a2"}})},
			placement: map[int][]string{0: {"a0", "a1"}, 1: {"a2"}}},
		// The pre-existing "disagrees on instances" row above exits through the
		// len(a) != len(b) shortcut (1 vs 2 keys) and never reaches the !ok branch (an
		// instance present in one map, absent from the other, at EQUAL key counts). This
		// row forces that branch: both maps have 2 keys, but index 2 replaces index 1.
		{name: "t=0 entry disagrees on instance keys at equal counts", policy: "scheduled",
			path: "s.txt",
			schedule: []sim.PlacementScheduleEntry{
				entry(0, map[int][]string{0: {"a0"}, 2: {"a1"}})},
			placement: seeded, wantErrPart: "disagrees with --lora-adapter-placement"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := validateLoRAScheduleFlags(tc.policy, tc.path, tc.schedule, tc.placement)
			if tc.wantErrPart == "" {
				if err != nil {
					t.Fatalf("want no error, got %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("want an error mentioning %q, got nil", tc.wantErrPart)
			}
			if !strings.Contains(err.Error(), tc.wantErrPart) {
				t.Errorf("error %q does not mention %q", err.Error(), tc.wantErrPart)
			}
		})
	}
}

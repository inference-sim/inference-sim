package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
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

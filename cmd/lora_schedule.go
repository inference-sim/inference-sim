package cmd

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
)

// parseLoRAPlacementSchedule reads a placement-schedule file into the timed entries the
// `scheduled` creation policy consumes (Spec 4 Slice B, design §13.3).
//
// Grammar, one entry per line: "<t_us> <idx=id[,id...];idx=id...>". Blank lines and lines
// whose first non-space character is '#' are ignored, so a generated schedule can carry a
// provenance header. The placement half is delegated to parseLoRAAdapterPlacement, which
// --lora-adapter-placement already uses, so the two flags cannot drift apart on the grammar
// they share.
//
// Every failure is an error rather than a skipped line. This file IS the independent
// variable of the re-solve arms: a dropped entry would leave a run that is not the arm its
// manifest claims, which is the exact failure mode issues #46 and #48 describe.
//
// Timestamps must strictly increase because the policy selects the latest entry with
// AtUs <= now; an out-of-order file would silently apply the wrong placement instead of
// failing.
func parseLoRAPlacementSchedule(path string) ([]sim.PlacementScheduleEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("placement schedule: %w", err)
	}
	defer func() { _ = f.Close() }()

	var out []sim.PlacementScheduleEntry
	line := 1
	scanner := bufio.NewScanner(f)
	for ; scanner.Scan(); line++ {
		text := strings.TrimSpace(scanner.Text())
		if text == "" || strings.HasPrefix(text, "#") {
			continue
		}
		stamp, spec, found := strings.Cut(text, " ")
		spec = strings.TrimSpace(spec)
		if !found {
			return nil, fmt.Errorf("placement schedule %s line %d: want \"<t_us> <placement>\", got %q",
				path, line, text)
		}
		atUs, err := strconv.ParseInt(stamp, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("placement schedule %s line %d: timestamp %q is not an integer "+
				"number of microseconds: %w", path, line, stamp, err)
		}
		if atUs < 0 {
			return nil, fmt.Errorf("placement schedule %s line %d: timestamp %d is negative",
				path, line, atUs)
		}
		if n := len(out); n > 0 && atUs <= out[n-1].AtUs {
			return nil, fmt.Errorf("placement schedule %s line %d: timestamp %d does not increase "+
				"past the previous entry's %d; the policy selects the latest entry at or before "+
				"now, so an out-of-order file silently applies the wrong placement",
				path, line, atUs, out[n-1].AtUs)
		}
		placement, err := parseLoRAAdapterPlacement(spec)
		if err != nil {
			return nil, fmt.Errorf("placement schedule %s line %d: %w", path, line, err)
		}
		if len(placement) == 0 {
			// parseLoRAAdapterPlacement returns a nil map only for spec == "", which
			// can't happen here: found == true guarantees a nonempty remainder after
			// Cut, because text was already fully trimmed and so never ends in
			// whitespace. This branch IS reachable, though — a spec that is non-empty
			// but parses to no chunks (e.g. ";", where every ';'-separated piece is
			// blank) yields an empty, non-nil map with no error. Kept as defence in
			// depth so that case is still an error rather than a silent no-op entry.
			return nil, fmt.Errorf("placement schedule %s line %d: empty placement; an entry "+
				"naming no adapter is a no-op that reads as a policy decision", path, line)
		}
		out = append(out, sim.PlacementScheduleEntry{AtUs: atUs, Placement: placement})
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("placement schedule %s line %d: %w", path, line, err)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("placement schedule %s has no entries; an empty schedule makes "+
			"`scheduled` a silent no-op indistinguishable from pre-placement", path)
	}
	return out, nil
}

// resolveLoRAPlacementSchedule parses --lora-placement-schedule, or returns nil when the flag
// is unset. Fatal on a bad file: an unreadable or malformed schedule cannot be degraded into
// "no schedule" without silently converting the scheduled arm into pre-placement.
func resolveLoRAPlacementSchedule() []sim.PlacementScheduleEntry {
	if loraPlacementSchedule == "" {
		return nil
	}
	schedule, err := parseLoRAPlacementSchedule(loraPlacementSchedule)
	if err != nil {
		logrus.Fatalf("Invalid --lora-placement-schedule: %v", err)
	}
	return schedule
}

// validateLoRAScheduleFlags enforces the three cross-flag rules the schedule introduces. Pure
// so it is unit-testable; the caller fatals (library returns an error, cmd decides fatality).
//
//  1. creation_policy="scheduled" requires a schedule. Without one the policy proposes nothing
//     and is indistinguishable from pre-placement -- the gate-only-policy failure mode of
//     issues #46 and #48, applied to a new knob.
//  2. A schedule requires creation_policy="scheduled". Under any other policy it reaches no
//     code path, so accepting it would let a run claim an arm it never exercised.
//  3. A t=0 entry must agree with --lora-adapter-placement. The two flags jointly define t=0
//     residency -- the flag seeds it uncharged, the entry is what the tick drives toward -- so
//     disagreement is a mis-specified arm, not a preference.
//
// A schedule whose first entry is after t=0 is legal and is exactly how a lookahead offset is
// expressed: nothing is in force until that entry, and t=0 residency comes from the flag alone.
func validateLoRAScheduleFlags(creationPolicy, schedulePath string,
	schedule []sim.PlacementScheduleEntry, placement map[int][]string) error {
	scheduled := creationPolicy == "scheduled"
	if scheduled && len(schedule) == 0 {
		return fmt.Errorf("--creation-policy=scheduled requires --lora-placement-schedule: with no "+
			"schedule the policy proposes nothing and is indistinguishable from pre-placement "+
			"(got --lora-placement-schedule=%q)", schedulePath)
	}
	if !scheduled && len(schedule) > 0 {
		return fmt.Errorf("--lora-placement-schedule is set (%q, %d entries) but "+
			"--creation-policy=%q; only the \"scheduled\" creation policy reads it, so this run "+
			"would silently not be the arm it claims", schedulePath, len(schedule), creationPolicy)
	}
	if len(schedule) == 0 || schedule[0].AtUs != 0 {
		return nil
	}
	if !samePlacement(schedule[0].Placement, placement) {
		return fmt.Errorf("the schedule's t=0 entry disagrees with --lora-adapter-placement: "+
			"schedule has %s, flag has %s. Both define t=0 residency -- the flag seeds it "+
			"uncharged and the entry is what the first tick drives toward -- so a disagreement "+
			"is a mis-specified arm", formatPlacementMap(schedule[0].Placement),
			formatPlacementMap(placement))
	}
	return nil
}

// samePlacement compares two placements for equality, order-insensitively within an instance.
// Order does not matter: the policy sorts each entry's ids before emitting decisions, and
// ValidateLoRAPlacement forbids duplicates within an instance.
func samePlacement(a, b map[int][]string) bool {
	if len(a) != len(b) {
		return false
	}
	for idx, want := range a {
		got, ok := b[idx]
		if !ok || len(got) != len(want) {
			return false
		}
		x := append([]string(nil), want...)
		y := append([]string(nil), got...)
		sort.Strings(x)
		sort.Strings(y)
		for i := range x {
			if x[i] != y[i] {
				return false
			}
		}
	}
	return true
}

// formatPlacementMap renders a placement deterministically for error text (INV-6): a ranged map
// would make the message depend on Go's map order.
func formatPlacementMap(p map[int][]string) string {
	if len(p) == 0 {
		return "(empty)"
	}
	indices := make([]int, 0, len(p))
	for idx := range p {
		indices = append(indices, idx)
	}
	sort.Ints(indices)
	parts := make([]string, 0, len(indices))
	for _, idx := range indices {
		ids := append([]string(nil), p[idx]...)
		sort.Strings(ids)
		parts = append(parts, fmt.Sprintf("%d=%s", idx, strings.Join(ids, ",")))
	}
	return strings.Join(parts, ";")
}

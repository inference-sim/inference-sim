package cmd

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

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

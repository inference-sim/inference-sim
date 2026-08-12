// sim/saturation/reduce.go
package saturation

// The final-label reducer (#1517). It collapses a detector's per-event verdict
// TRACE (the []TraceRecord produced by #1516/#1519) into ONE headline Level, by
// the same last-window plurality rule for every detector. It is a pure function
// — deliberately NOT a Detector method — so every detector is collapsed
// identically (fair cross-detector comparison) and new detectors get final
// labeling for free. cmd emits the result on stdout via the goodput pattern.

// ReduceOne collapses ONE detector's records into a single final Level:
//
//  1. Trailing window: lastT = max(Timestamp); keep records with
//     Timestamp >= lastT - windowUs (boundary inclusive). All timestamps are µs.
//  2. Plurality vote over Result.Level among the kept records; the most frequent
//     level wins. The shape stays 3-level (not collapsed to binary).
//  3. Deterministic tie-break: on a count tie, the MORE SEVERE level wins
//     (OVERLOADED > BACKLOGGED > STABLE) — conservative and order-independent, so
//     the result depends only on the multiset of kept levels, never their order
//     (INV-6).
//
// An empty group (no records) → STABLE, the degenerate default: there is no
// Result to read, so the label alone is STABLE (R20).
//
// Because the vote is over the trailing window, a run that recovered by the end
// reads STABLE even if it spiked mid-run, and a single transient blip in the
// final instant cannot flip the verdict alone.
func ReduceOne(records []TraceRecord, windowUs int64) Level {
	if len(records) == 0 {
		return Stable
	}

	var lastT int64
	for i, r := range records {
		if i == 0 || r.Timestamp > lastT {
			lastT = r.Timestamp
		}
	}
	cutoff := lastT - windowUs

	// counts is indexed by Level (Stable=0, Backlogged=1, Overloaded=2), so a
	// descending scan over the index space naturally visits the most severe level
	// first — that IS the severity tie-break, with no separate comparison.
	var counts [3]int
	for _, r := range records {
		if r.Timestamp < cutoff {
			continue
		}
		if r.Result.Level >= Stable && r.Result.Level <= Overloaded {
			counts[r.Result.Level]++
		}
	}

	// Most severe level with the maximum count wins. Scanning severity-descending
	// and using strict `>` means the first (most severe) level holding the max is
	// selected, so a tie resolves toward the more severe level.
	best := Stable
	bestCount := -1
	for lvl := Overloaded; lvl >= Stable; lvl-- {
		if counts[lvl] > bestCount {
			bestCount = counts[lvl]
			best = lvl
		}
	}
	return best
}

// ReduceAll groups records by TraceRecord.Detector and applies ReduceOne to each
// group independently, returning a name→Level map. This is the single entry point
// cmd calls for every selection: a one-detector selection yields a one-entry map,
// `all` yields the full map. Each group's trailing window is anchored to that
// group's own max timestamp (ReduceOne is called per group), so one detector's
// timeline never truncates another's. An empty input yields an empty map, so
// cmd's len()>0 guard drops the stdout saturation field (BC-8).
func ReduceAll(records []TraceRecord, windowUs int64) map[string]Level {
	if len(records) == 0 {
		return map[string]Level{}
	}
	byDetector := make(map[string][]TraceRecord)
	for _, r := range records {
		byDetector[r.Detector] = append(byDetector[r.Detector], r)
	}
	out := make(map[string]Level, len(byDetector))
	for name, group := range byDetector {
		out[name] = ReduceOne(group, windowUs)
	}
	return out
}

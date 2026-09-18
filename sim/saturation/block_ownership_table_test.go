package saturation

import "testing"

// The block-ownership decision for every (block, selected detector) pair, pinned as
// a table rather than verified by reading the loops in config.go.
//
// The table is exhaustive over blockOwners() x AllDetectorNames() and is checked on
// BOTH the single-detector and bank paths, so the two cannot drift apart -- the
// failure that let a composite: block be silently dropped on the bank path before
// the ownership table existed (R1).
func TestBlockOwnership_ExhaustiveTruthTable(t *testing.T) {
	one := 1.0
	ms := 5000.0
	k := 3.0
	set := map[string]func(*SaturationConfig){
		"composite":     func(c *SaturationConfig) { c.Composite = &CompositeBlock{Sensitivity: &one} },
		"threshold":     func(c *SaturationConfig) { c.Threshold = &ThresholdBlock{ThresholdMs: &ms} },
		"backlog_drift": func(c *SaturationConfig) { c.BacklogDrift = &BacklogDriftBlock{SlopeK: &k} },
		"peak_rate":     func(c *SaturationConfig) { c.PeakRate = &PeakRateBlock{Threshold: &one} },
	}

	// block -> selected detector -> accepted? Each block is valid for exactly its
	// own owner.
	want := map[string]map[string]bool{
		"composite":     {"composite": true, "threshold": false, "backlog-drift": false, "peak-rate": false},
		"threshold":     {"composite": false, "threshold": true, "backlog-drift": false, "peak-rate": false},
		"backlog_drift": {"composite": false, "threshold": false, "backlog-drift": true, "peak-rate": false},
		"peak_rate":     {"composite": false, "threshold": false, "backlog-drift": false, "peak-rate": true},
	}

	// The table must cover the whole roster, so a new detector cannot be added
	// without extending it.
	if len(set) != len(AllDetectorNames()) {
		t.Fatalf("the truth table covers %d blocks but the roster has %d detectors; extend the table", len(set), len(AllDetectorNames()))
	}

	for block, row := range want {
		for detector, accept := range row {
			cfg := SaturationConfig{}
			set[block](&cfg)

			_, err := BuildDetector(detector, cfg)
			if accept && err != nil {
				t.Errorf("single: block %q + --detectors %s should be ACCEPTED, got %v", block, detector, err)
			}
			if !accept && err == nil {
				t.Errorf("single: block %q + --detectors %s should be REJECTED, got no error", block, detector)
			}

			_, bankErr := NewBank([]string{detector}, cfg, NewInMemoryCollector())
			if accept && bankErr != nil {
				t.Errorf("bank: block %q + --detectors %s should be ACCEPTED, got %v", block, detector, bankErr)
			}
			if !accept && bankErr == nil {
				t.Errorf("bank: block %q + --detectors %s should be REJECTED, got no error", block, detector)
			}
		}
	}

	// A config carrying EVERY block is valid only under a selection containing every
	// owner -- i.e. `all`.
	full := SaturationConfig{}
	for _, f := range set {
		f(&full)
	}
	if _, err := NewBank(AllDetectorNames(), full, NewInMemoryCollector()); err != nil {
		t.Errorf("a full config under --detectors all should be accepted, got %v", err)
	}
	for _, subset := range [][]string{{"composite"}, {"threshold"}, {"peak-rate"}, {"composite", "threshold"}} {
		if _, err := NewBank(subset, full, NewInMemoryCollector()); err == nil {
			t.Errorf("a full config under the subset %v should be rejected, got no error", subset)
		}
	}
}

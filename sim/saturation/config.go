// sim/saturation/config.go
package saturation

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// SaturationConfig is the strict-YAML replacement for the 11 saturation tuning
// flags (#1516). It carries one optional block per parameterized detector:
//
//   - composite:      the CompositeDetector's noise-floor knob (sensitivity, #1614)
//   - threshold:      the ThresholdDetector's single knob (threshold_ms)
//   - backlog_drift:  the BacklogDriftDetector's tuning knobs (mirrors
//     BacklogDriftConfig)
//
// Every detector is now tunable. That is a correctness property, not a
// convenience: detector scores are comparable only when each detector has first
// been calibrated to the same false-alarm rate, and a detector with no knob
// cannot be moved onto that rate — it can only be disqualified (#1614).
//
// Fields are pointers so an absent key keeps the detector's default while a
// present key overrides only the field it names (R9: distinguish "unset" from
// "zero"). An empty file parses to a SaturationConfig with all-nil blocks, which
// means "all defaults" — not an error.
type SaturationConfig struct {
	Composite    *CompositeBlock    `yaml:"composite,omitempty"`
	Threshold    *ThresholdBlock    `yaml:"threshold,omitempty"`
	BacklogDrift *BacklogDriftBlock `yaml:"backlog_drift,omitempty"`
}

// CompositeBlock overrides the CompositeDetector's noise-floor multiplier.
// A larger sensitivity raises the floor, so the detector fires less; 1.0 is the
// historical behaviour.
type CompositeBlock struct {
	Sensitivity *float64 `yaml:"sensitivity"`
}

// ThresholdBlock overrides the ThresholdDetector's mean-E2E threshold.
type ThresholdBlock struct {
	ThresholdMs *float64 `yaml:"threshold_ms"`
}

// BacklogDriftBlock overrides fields of BacklogDriftConfig. Each field
// is optional; absent fields keep DefaultBacklogDriftConfig's value.
// window_size_sec is expressed in whole seconds (matching the retired
// --saturation-window flag, which was also seconds).
type BacklogDriftBlock struct {
	WindowSizeSec       *int     `yaml:"window_size_sec"`
	MinWindows          *int     `yaml:"min_windows"`
	PeakRatio           *float64 `yaml:"peak_ratio"`
	PeakRatioBand       *float64 `yaml:"peak_ratio_band"`
	ConfidenceCI        *float64 `yaml:"confidence_ci"`
	WarmupWindows       *int     `yaml:"warmup_windows"`
	TailWindows         *int     `yaml:"tail_windows"`
	SaturatedDrainRatio *float64 `yaml:"saturated_drain_ratio"`
	TransientDrainRatio *float64 `yaml:"transient_drain_ratio"`
	// SlopeK is backlog-drift's false-alarm calibration knob (#1614): the
	// multiplier separating BACKLOGGED from OVERLOADED. Absent keeps the
	// documented default (3.0).
	SlopeK *float64 `yaml:"slope_k"`
}

// LoadSaturationConfig reads and strictly parses a saturation config file. An
// empty path returns the zero config (all defaults) without touching disk.
// Unknown keys error via KnownFields(true) — including a misspelled field inside
// an otherwise valid block.
func LoadSaturationConfig(path string) (SaturationConfig, error) {
	var cfg SaturationConfig
	if path == "" {
		return cfg, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, fmt.Errorf("read saturation config %s: %w", path, err)
	}
	// An empty file is valid — decode leaves cfg at its zero value (all defaults).
	if len(bytes.TrimSpace(data)) == 0 {
		return cfg, nil
	}
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&cfg); err != nil {
		return cfg, fmt.Errorf("parse saturation config %s: %w", path, err)
	}
	return cfg, nil
}

// defaultThresholdMs is the ThresholdDetector's default mean-E2E threshold when
// no threshold.threshold_ms override is supplied (matches the retired
// --saturation-threshold-ms default and NewThresholdDetector's own fallback).
const defaultThresholdMs = 5000.0

// defaultCompositeSensitivityYAML is the sensitivity used when no
// composite.sensitivity override is supplied. It mirrors
// defaultCompositeSensitivity so the resolved value is explicit at this layer.
const defaultCompositeSensitivityYAML = defaultCompositeSensitivity

// BuildDetector constructs the named detector, applying any relevant overrides
// from cfg. Returns an error (never panics — R6) when a name is unknown, a
// supplied parameter is out of range, or a config block is present that does not
// belong to the selected detector; the error names the offending field.
//
// This is the SINGLE-detector entry point (#1516): it enforces block↔detector
// ownership (checkBlockOwnership) because exactly one detector runs, so any
// foreign block is a user mistake. The bank (#1519) drives several detectors and
// enforces ownership over the selected SET once (checkBlockOwnershipSet in
// NewBank), then calls buildDetector per detector — so a block whose owner is not
// in the bank's selection is likewise a hard error, not a silent drop (R1).
func BuildDetector(name string, cfg SaturationConfig) (Detector, error) {
	// Reject config blocks that do not belong to the selected detector rather
	// than silently dropping the user's tuning (R1). SaturationConfig always
	// knows both keys (strict parsing can't tell which detector is active), so
	// the block↔detector match is enforced here.
	if err := checkBlockOwnership(name, cfg); err != nil {
		return nil, err
	}
	return buildDetector(name, cfg)
}

// buildDetector constructs the named detector, applying only the block that
// belongs to name and ignoring the rest of cfg. It does NOT enforce block
// ownership — that is the caller's job (checkBlockOwnership for the single-detector
// path, checkBlockOwnershipSet for the bank). It still validates the values of the
// block it reads (range/finiteness), so a selected detector with an out-of-range
// parameter errors (never panics — R6).
func buildDetector(name string, cfg SaturationConfig) (Detector, error) {
	switch name {
	case "composite":
		sensitivity := defaultCompositeSensitivityYAML
		if cfg.Composite != nil && cfg.Composite.Sensitivity != nil {
			sensitivity = *cfg.Composite.Sensitivity
			if sensitivity <= 0 || math.IsNaN(sensitivity) || math.IsInf(sensitivity, 0) {
				return nil, fmt.Errorf("saturation config: composite.sensitivity must be a finite value > 0, got %v", sensitivity)
			}
		}
		return NewCompositeDetectorWithSensitivity(sensitivity), nil
	case "threshold":
		thresholdMs := defaultThresholdMs
		if cfg.Threshold != nil && cfg.Threshold.ThresholdMs != nil {
			thresholdMs = *cfg.Threshold.ThresholdMs
			if thresholdMs <= 0 || math.IsNaN(thresholdMs) || math.IsInf(thresholdMs, 0) {
				return nil, fmt.Errorf("saturation config: threshold.threshold_ms must be a finite value > 0, got %v", thresholdMs)
			}
		}
		return NewThresholdDetector(thresholdMs), nil
	case "backlog-drift":
		bdc, err := resolveBacklogDriftConfig(cfg.BacklogDrift)
		if err != nil {
			return nil, err
		}
		return NewBacklogDriftDetectorWithConfig(bdc), nil
	default:
		// Derived from the roster so a future detector cannot desync this list.
		return nil, fmt.Errorf("unknown saturation detector %q; valid: %s", name, strings.Join(AllDetectorNames(), ", "))
	}
}

// blockOwner pairs one tuning block with the detector that owns it.
type blockOwner struct {
	block   string                      // YAML key, as it appears in error messages
	owner   string                      // the only detector that may carry it
	present func(SaturationConfig) bool // whether the block is set
}

// blockOwners returns the block↔detector ownership table.
//
// This is the SINGLE source of truth both ownership checks derive from
// (#1614). They were previously two hand-written functions enumerating blocks in
// separate if-statements, which could — and did — disagree: before #1614
// checkBlockOwnershipSet had no composite case at all, so once composite gained a
// block, a composite: block supplied to a bank selection that omitted composite
// would have been silently dropped (R1). Adding a tunable detector now means
// adding one row here.
//
// The slice is constructed per call so no mutable package-level slice escapes
// (R8). The predicates are pure readers of the config they are handed.
func blockOwners() []blockOwner {
	return []blockOwner{
		{"composite", "composite", func(c SaturationConfig) bool { return c.Composite != nil }},
		{"threshold", "threshold", func(c SaturationConfig) bool { return c.Threshold != nil }},
		{"backlog_drift", "backlog-drift", func(c SaturationConfig) bool { return c.BacklogDrift != nil }},
	}
}

// checkBlockOwnership rejects a config that carries a tuning block for a detector
// other than the selected one (single-detector path). Exactly one detector runs,
// so any foreign block is a user mistake and is reported rather than dropped (R1).
func checkBlockOwnership(name string, cfg SaturationConfig) error {
	for _, bo := range blockOwners() {
		if bo.present(cfg) && bo.owner != name {
			return fmt.Errorf("saturation config: %s block is not valid for --detectors %s (it belongs to %s)",
				bo.block, name, bo.owner)
		}
	}
	return nil
}

// checkBlockOwnershipSet is the multi-detector generalization for the bank
// (#1519). A tuning block is valid only if its owning detector is among the
// selected names; a block whose owner is NOT selected is a hard error rather than
// a silent drop (R1), matching the single-detector contract. `--detectors all`
// selects every owner, so a full shared config trivially passes; the check bites
// on subset selections that omit a detector whose block was nonetheless supplied.
func checkBlockOwnershipSet(names []string, cfg SaturationConfig) error {
	selected := make(map[string]bool, len(names))
	for _, n := range names {
		selected[n] = true
	}
	for _, bo := range blockOwners() {
		if bo.present(cfg) && !selected[bo.owner] {
			return fmt.Errorf("saturation config: %s block is not valid for --detectors %q (%s is not among the selected detectors)",
				bo.block, strings.Join(names, ","), bo.owner)
		}
	}
	return nil
}

// resolveBacklogDriftConfig merges a BacklogDriftBlock over the defaults and
// validates the result, returning errors (naming the YAML field) rather than
// panicking so the library boundary stays panic-free (R6). Bounds mirror
// NewBacklogDriftConfig so the subsequent construction cannot panic.
func resolveBacklogDriftConfig(block *BacklogDriftBlock) (BacklogDriftConfig, error) {
	def := DefaultBacklogDriftConfig()

	windowSize := def.WindowSize
	minWindows := def.MinWindows
	peakRatio := def.PeakRatio
	peakRatioBand := def.PeakRatioBand
	confidenceCI := def.ConfidenceCI
	warmupWindows := def.WarmupWindows
	tailWindows := def.TailWindows
	saturatedDrainRatio := def.SaturatedDrainRatio
	transientDrainRatio := def.TransientDrainRatio
	slopeK := def.effectiveSlopeK()

	if block != nil {
		if block.WindowSizeSec != nil {
			if *block.WindowSizeSec <= 0 {
				return BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.window_size_sec must be > 0, got %d", *block.WindowSizeSec)
			}
			windowSize = time.Duration(*block.WindowSizeSec) * time.Second
		}
		if block.MinWindows != nil {
			if *block.MinWindows <= 0 {
				return BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.min_windows must be > 0, got %d", *block.MinWindows)
			}
			minWindows = *block.MinWindows
		}
		if block.PeakRatio != nil {
			if *block.PeakRatio <= 0 || math.IsNaN(*block.PeakRatio) || math.IsInf(*block.PeakRatio, 0) {
				return BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.peak_ratio must be a finite value > 0, got %v", *block.PeakRatio)
			}
			peakRatio = *block.PeakRatio
		}
		if block.PeakRatioBand != nil {
			if *block.PeakRatioBand < 0 || math.IsNaN(*block.PeakRatioBand) || math.IsInf(*block.PeakRatioBand, 0) {
				return BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.peak_ratio_band must be >= 0, got %v", *block.PeakRatioBand)
			}
			peakRatioBand = *block.PeakRatioBand
		}
		if block.ConfidenceCI != nil {
			if *block.ConfidenceCI <= 0 || *block.ConfidenceCI >= 1 || math.IsNaN(*block.ConfidenceCI) || math.IsInf(*block.ConfidenceCI, 0) {
				return BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.confidence_ci must be in (0, 1), got %v", *block.ConfidenceCI)
			}
			confidenceCI = *block.ConfidenceCI
		}
		if block.WarmupWindows != nil {
			if *block.WarmupWindows < 0 {
				return BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.warmup_windows must be >= 0, got %d", *block.WarmupWindows)
			}
			warmupWindows = *block.WarmupWindows
		}
		if block.TailWindows != nil {
			if *block.TailWindows < 0 {
				return BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.tail_windows must be >= 0, got %d", *block.TailWindows)
			}
			tailWindows = *block.TailWindows
		}
		if block.SaturatedDrainRatio != nil {
			if *block.SaturatedDrainRatio <= 0 || *block.SaturatedDrainRatio > 1 || math.IsNaN(*block.SaturatedDrainRatio) || math.IsInf(*block.SaturatedDrainRatio, 0) {
				return BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.saturated_drain_ratio must be in (0, 1], got %v", *block.SaturatedDrainRatio)
			}
			saturatedDrainRatio = *block.SaturatedDrainRatio
		}
		if block.TransientDrainRatio != nil {
			if *block.TransientDrainRatio <= 0 || *block.TransientDrainRatio > 1 || math.IsNaN(*block.TransientDrainRatio) || math.IsInf(*block.TransientDrainRatio, 0) {
				return BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.transient_drain_ratio must be in (0, 1], got %v", *block.TransientDrainRatio)
			}
			transientDrainRatio = *block.TransientDrainRatio
		}
		// Validated HERE rather than relying on effectiveSlopeK's fallback: the
		// fallback exists for in-process struct literals, so letting a user's
		// slope_k: 0 reach it would silently coerce the value to 3.0 instead of
		// reporting the mistake (R1).
		if block.SlopeK != nil {
			if *block.SlopeK <= 0 || math.IsNaN(*block.SlopeK) || math.IsInf(*block.SlopeK, 0) {
				return BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.slope_k must be a finite value > 0, got %v", *block.SlopeK)
			}
			slopeK = *block.SlopeK
		}
	}

	// Cross-field invariant (mirrors NewBacklogDriftConfig): the two drain-ratio
	// thresholds must not overlap. Checked here so we return an error instead of
	// letting NewBacklogDriftConfig panic.
	if saturatedDrainRatio > transientDrainRatio {
		return BacklogDriftConfig{}, fmt.Errorf(
			"saturation config: backlog_drift.saturated_drain_ratio (%v) must be <= transient_drain_ratio (%v); regions would overlap",
			saturatedDrainRatio, transientDrainRatio)
	}

	// NewBacklogDriftConfig validates and returns a fresh struct literal, so
	// SlopeK is zero-filled there and must be set afterwards. Its value was
	// already validated above; effectiveSlopeK() would otherwise mask it.
	resolved := NewBacklogDriftConfig(
		windowSize, minWindows, peakRatio, peakRatioBand, confidenceCI,
		warmupWindows, tailWindows, saturatedDrainRatio, transientDrainRatio,
	)
	resolved.SlopeK = slopeK
	return resolved, nil
}

// sim/saturation/config.go
package saturation

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// SaturationConfig is the strict-YAML replacement for the 11 saturation tuning
// flags (#1516). It carries one optional block per parameterized detector:
//
//   - threshold:      the ThresholdDetector's single knob (threshold_ms)
//   - backlog_drift:  the BacklogDriftDetector's tuning knobs (mirrors
//     workload.BacklogDriftConfig)
//
// composite has no tunable parameters, so it has no block — a "composite:" key
// therefore fails strict parsing (KnownFields), which is the intended contract.
//
// Fields are pointers so an absent key keeps the detector's default while a
// present key overrides only the field it names (R9: distinguish "unset" from
// "zero"). An empty file parses to a SaturationConfig with all-nil blocks, which
// means "all defaults" — not an error.
type SaturationConfig struct {
	Threshold    *ThresholdBlock    `yaml:"threshold,omitempty"`
	BacklogDrift *BacklogDriftBlock `yaml:"backlog_drift,omitempty"`
}

// ThresholdBlock overrides the ThresholdDetector's mean-E2E threshold.
type ThresholdBlock struct {
	ThresholdMs *float64 `yaml:"threshold_ms"`
}

// BacklogDriftBlock overrides fields of workload.BacklogDriftConfig. Each field
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
}

// LoadSaturationConfig reads and strictly parses a saturation config file. An
// empty path returns the zero config (all defaults) without touching disk.
// Unknown keys (including a "composite:" block) error via KnownFields(true).
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

// BuildDetector constructs the named detector, applying any relevant overrides
// from cfg. Returns an error (never panics — R6) when a name is unknown, a
// supplied parameter is out of range, or a config block is present that does not
// belong to the selected detector; the error names the offending field.
//
// The bank / multi-detector selection (`all`, comma-lists) is intentionally not
// handled here — that belongs to #1519. Callers must pass exactly one name.
func BuildDetector(name string, cfg SaturationConfig) (Detector, error) {
	// Reject config blocks that do not belong to the selected detector rather
	// than silently dropping the user's tuning (R1). SaturationConfig always
	// knows both keys (strict parsing can't tell which detector is active), so
	// the block↔detector match is enforced here.
	if err := checkBlockOwnership(name, cfg); err != nil {
		return nil, err
	}

	switch name {
	case "composite":
		return NewCompositeDetector(), nil
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
		return nil, fmt.Errorf("unknown saturation detector %q; valid: composite, threshold, backlog-drift", name)
	}
}

// checkBlockOwnership rejects a config that carries a tuning block for a detector
// other than the selected one. composite has no tunable params, so ANY block is
// a mistake when composite is selected; threshold accepts only threshold:;
// backlog-drift accepts only backlog_drift:.
func checkBlockOwnership(name string, cfg SaturationConfig) error {
	switch name {
	case "composite":
		if cfg.Threshold != nil {
			return fmt.Errorf("saturation config: threshold block is not valid for --detectors composite (composite has no tunable parameters)")
		}
		if cfg.BacklogDrift != nil {
			return fmt.Errorf("saturation config: backlog_drift block is not valid for --detectors composite (composite has no tunable parameters)")
		}
	case "threshold":
		if cfg.BacklogDrift != nil {
			return fmt.Errorf("saturation config: backlog_drift block is not valid for --detectors threshold")
		}
	case "backlog-drift":
		if cfg.Threshold != nil {
			return fmt.Errorf("saturation config: threshold block is not valid for --detectors backlog-drift")
		}
	}
	return nil
}

// resolveBacklogDriftConfig merges a BacklogDriftBlock over the defaults and
// validates the result, returning errors (naming the YAML field) rather than
// panicking so the library boundary stays panic-free (R6). Bounds mirror
// workload.NewBacklogDriftConfig so the subsequent construction cannot panic.
func resolveBacklogDriftConfig(block *BacklogDriftBlock) (workload.BacklogDriftConfig, error) {
	def := workload.DefaultBacklogDriftConfig()

	windowSize := def.WindowSize
	minWindows := def.MinWindows
	peakRatio := def.PeakRatio
	peakRatioBand := def.PeakRatioBand
	confidenceCI := def.ConfidenceCI
	warmupWindows := def.WarmupWindows
	tailWindows := def.TailWindows
	saturatedDrainRatio := def.SaturatedDrainRatio
	transientDrainRatio := def.TransientDrainRatio

	if block != nil {
		if block.WindowSizeSec != nil {
			if *block.WindowSizeSec <= 0 {
				return workload.BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.window_size_sec must be > 0, got %d", *block.WindowSizeSec)
			}
			windowSize = time.Duration(*block.WindowSizeSec) * time.Second
		}
		if block.MinWindows != nil {
			if *block.MinWindows <= 0 {
				return workload.BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.min_windows must be > 0, got %d", *block.MinWindows)
			}
			minWindows = *block.MinWindows
		}
		if block.PeakRatio != nil {
			if *block.PeakRatio <= 0 || math.IsNaN(*block.PeakRatio) || math.IsInf(*block.PeakRatio, 0) {
				return workload.BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.peak_ratio must be a finite value > 0, got %v", *block.PeakRatio)
			}
			peakRatio = *block.PeakRatio
		}
		if block.PeakRatioBand != nil {
			if *block.PeakRatioBand < 0 || math.IsNaN(*block.PeakRatioBand) || math.IsInf(*block.PeakRatioBand, 0) {
				return workload.BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.peak_ratio_band must be >= 0, got %v", *block.PeakRatioBand)
			}
			peakRatioBand = *block.PeakRatioBand
		}
		if block.ConfidenceCI != nil {
			if *block.ConfidenceCI <= 0 || *block.ConfidenceCI >= 1 || math.IsNaN(*block.ConfidenceCI) || math.IsInf(*block.ConfidenceCI, 0) {
				return workload.BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.confidence_ci must be in (0, 1), got %v", *block.ConfidenceCI)
			}
			confidenceCI = *block.ConfidenceCI
		}
		if block.WarmupWindows != nil {
			if *block.WarmupWindows < 0 {
				return workload.BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.warmup_windows must be >= 0, got %d", *block.WarmupWindows)
			}
			warmupWindows = *block.WarmupWindows
		}
		if block.TailWindows != nil {
			if *block.TailWindows < 0 {
				return workload.BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.tail_windows must be >= 0, got %d", *block.TailWindows)
			}
			tailWindows = *block.TailWindows
		}
		if block.SaturatedDrainRatio != nil {
			if *block.SaturatedDrainRatio <= 0 || *block.SaturatedDrainRatio > 1 || math.IsNaN(*block.SaturatedDrainRatio) || math.IsInf(*block.SaturatedDrainRatio, 0) {
				return workload.BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.saturated_drain_ratio must be in (0, 1], got %v", *block.SaturatedDrainRatio)
			}
			saturatedDrainRatio = *block.SaturatedDrainRatio
		}
		if block.TransientDrainRatio != nil {
			if *block.TransientDrainRatio <= 0 || *block.TransientDrainRatio > 1 || math.IsNaN(*block.TransientDrainRatio) || math.IsInf(*block.TransientDrainRatio, 0) {
				return workload.BacklogDriftConfig{}, fmt.Errorf("saturation config: backlog_drift.transient_drain_ratio must be in (0, 1], got %v", *block.TransientDrainRatio)
			}
			transientDrainRatio = *block.TransientDrainRatio
		}
	}

	// Cross-field invariant (mirrors NewBacklogDriftConfig): the two drain-ratio
	// thresholds must not overlap. Checked here so we return an error instead of
	// letting NewBacklogDriftConfig panic.
	if saturatedDrainRatio > transientDrainRatio {
		return workload.BacklogDriftConfig{}, fmt.Errorf(
			"saturation config: backlog_drift.saturated_drain_ratio (%v) must be <= transient_drain_ratio (%v); regions would overlap",
			saturatedDrainRatio, transientDrainRatio)
	}

	return workload.NewBacklogDriftConfig(
		windowSize, minWindows, peakRatio, peakRatioBand, confidenceCI,
		warmupWindows, tailWindows, saturatedDrainRatio, transientDrainRatio,
	), nil
}

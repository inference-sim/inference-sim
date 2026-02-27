package workload

import (
	"fmt"
	"math"
	"math/rand"
)

// ExpandCohorts transforms cohort population descriptions into explicit client specs.
// Pure function: same (cohorts, seed) always produces identical output (INV-W3).
// Each cohort derives an independent RNG sub-stream from seed and its index.
// Note: reordering cohorts in the input changes their index-derived seeds.
func ExpandCohorts(cohorts []CohortSpec, seed int64) []ClientSpec {
	var expanded []ClientSpec
	for i, cohort := range cohorts {
		if cohort.Population <= 0 {
			continue // R11: guard against div-by-zero when called without Validate()
		}

		// Derive independent RNG per cohort using multiplicative hash.
		// Avoids XOR collision (seed=1^2 == seed=2^1) and zero-seed
		// (seed==i+1 → XOR=0). Knuth multiplicative hash spreads entropy.
		cohortSeed := seed*2654435761 + int64(i)
		cohortRNG := rand.New(rand.NewSource(cohortSeed))

		perMemberFraction := cohort.RateFraction / float64(cohort.Population)

		for j := 0; j < cohort.Population; j++ {
			clientID := fmt.Sprintf("%s-%d", cohort.ID, j)

			client := ClientSpec{
				ID:           clientID,
				TenantID:     cohort.TenantID,
				SLOClass:     cohort.SLOClass,
				Model:        cohort.Model,
				RateFraction: perMemberFraction,
				Arrival:      cohort.Arrival,
				InputDist:    cohort.InputDist,
				OutputDist:   cohort.OutputDist,
				PrefixGroup:  cohort.PrefixGroup,
				Streaming:    cohort.Streaming,
			}

			// Build lifecycle windows from cohort patterns
			var windows []ActiveWindow
			if cohort.Diurnal != nil {
				windows = append(windows, diurnalWindows(cohort.Diurnal, cohortRNG)...)
			}
			if cohort.Spike != nil {
				windows = append(windows, spikeWindow(cohort.Spike))
			}
			if cohort.Drain != nil {
				windows = append(windows, drainWindows(cohort.Drain)...)
			}

			if len(windows) > 0 {
				client.Lifecycle = &LifecycleSpec{Windows: windows}
			}

			expanded = append(expanded, client)
		}
	}
	return expanded
}

// diurnalWindows creates 24 one-hour lifecycle windows with sinusoidal rate modulation.
// Rate modulation is achieved by varying the active duration within each hour:
// at peak hours the client is active for the full hour; at trough hours,
// only for a fraction (1/R) of the hour. This produces discrete on/off
// patterns within each hour (square-wave), not smooth inter-arrival rate
// changes. For smooth diurnal profiles, use a RateMultiplier on ActiveWindow
// (not yet implemented).
//
// Rate multiplier at hour h: baseRate * modulation(h), where modulation produces:
//   - At peakHour: multiplier = 1.0 (full hour active)
//   - At trough (peakHour ± 12): multiplier = 1/R (fraction of hour active)
//
// The formula: multiplier = (1 + cos(2π(h - peakHour)/24)) / 2 * (1 - 1/R) + 1/R
func diurnalWindows(d *DiurnalSpec, rng *rand.Rand) []ActiveWindow {
	_ = rng // reserved for future session-duration jitter

	const hoursPerDay = 24
	const usPerHour = int64(3600 * 1e6) // 1 hour in microseconds

	R := d.PeakToTroughRatio
	windows := make([]ActiveWindow, hoursPerDay)

	for h := 0; h < hoursPerDay; h++ {
		// Cosine modulation: peak at peakHour, trough at peakHour ± 12
		angle := 2.0 * math.Pi * float64(h-d.PeakHour) / float64(hoursPerDay)
		// multiplier ranges from 1/R (trough) to 1.0 (peak)
		multiplier := (1.0+math.Cos(angle))/2.0*(1.0-1.0/R) + 1.0/R

		// Use multiplier to scale the window duration within each hour.
		// A full hour is the maximum; reduced hours model lower rates.
		// Window covers [h*usPerHour, h*usPerHour + duration]
		durationUs := int64(float64(usPerHour) * multiplier)
		if durationUs > usPerHour {
			durationUs = usPerHour
		}
		if durationUs < 1 {
			durationUs = 1 // minimum 1µs window
		}

		windows[h] = ActiveWindow{
			StartUs: int64(h) * usPerHour,
			EndUs:   int64(h)*usPerHour + durationUs,
		}
	}
	return windows
}

// spikeWindow creates a single lifecycle window for a traffic spike.
func spikeWindow(s *SpikeSpec) ActiveWindow {
	return ActiveWindow{
		StartUs: s.StartTimeUs,
		EndUs:   s.StartTimeUs + s.DurationUs,
	}
}

// drainWindows creates lifecycle windows that approximate a linear ramp-down to zero.
// Divides the ramp into 10 segments with linearly decreasing durations per interval.
func drainWindows(d *DrainSpec) []ActiveWindow {
	const segments = 10
	segmentDuration := d.RampDurationUs / int64(segments)
	if segmentDuration < 1 {
		segmentDuration = 1
	}

	// Also include the pre-drain window (full rate from 0 to start_time)
	var windows []ActiveWindow
	if d.StartTimeUs > 0 {
		windows = append(windows, ActiveWindow{
			StartUs: 0,
			EndUs:   d.StartTimeUs,
		})
	}

	// Ramp-down segments: each segment covers a fraction of the interval
	// proportional to (segments - i) / segments, approximating linear decrease
	for i := 0; i < segments; i++ {
		fraction := float64(segments-i) / float64(segments)
		segStart := d.StartTimeUs + int64(i)*segmentDuration
		activeDuration := int64(float64(segmentDuration) * fraction)
		if activeDuration < 1 {
			activeDuration = 1
		}
		windows = append(windows, ActiveWindow{
			StartUs: segStart,
			EndUs:   segStart + activeDuration,
		})
	}
	return windows
}

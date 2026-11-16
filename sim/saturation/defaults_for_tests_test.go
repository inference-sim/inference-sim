package saturation

// Constructors at the default configuration, and a sink that discards every record, which only
// this package's tests use: a run builds its detectors from the saturation configuration
// (config.go) and its sink from the report flag. Moved here unchanged (design LIF-7).

// NewBacklogDriftDetector creates a BacklogDriftDetector with default configuration.
func NewBacklogDriftDetector() Detector {
	return newBacklogDriftDetector(DefaultBacklogDriftConfig())
}

// NewCompositeDetector creates a composite detector at its default sensitivity
// (1.0), which is byte-identical to the pre-#1614 detector.
func NewCompositeDetector() Detector {
	return NewCompositeDetectorWithSensitivity(defaultCompositeSensitivity)
}

// NewPeakRateDetector creates a peak-rate detector with the campaign-validated
// defaults.
func NewPeakRateDetector() Detector {
	return newPeakRateDetector(peakRateConfig{
		Threshold:        defaultPeakRateThreshold,
		MinObservations:  defaultPeakRateMinObservations,
		ConsecutiveK:     defaultPeakRateConsecutiveK,
		OverloadMultiple: defaultPeakRateOverloadMultiple,
		WarmupUs:         defaultPeakRateWarmupUs,
	})
}

// NoOpSink discards every record. It is the default sink when no report path is
// requested, so the replay loop can run unconditionally without allocating.
type NoOpSink struct{}

// NewNoOpSink returns a sink that discards all records.
func NewNoOpSink() *NoOpSink { return &NoOpSink{} }

func (n *NoOpSink) Record(int64, string, Result) {}
func (n *NoOpSink) Close()                       {}

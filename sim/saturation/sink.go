// sim/saturation/sink.go
package saturation

// TraceRecord is one per-event saturation verdict: the event timestamp (µs, the
// timestamp of the event that produced this verdict), the detector's name, and
// the detector's Result at that point in the stream.
type TraceRecord struct {
	Timestamp int64  `json:"timestamp"`
	Detector  string `json:"detector"`
	Result    Result `json:"result"`
}

// TraceSink receives one verdict record per streamed event and a terminal Close.
// It is the seam that lets the replay loop stay agnostic about whether verdicts
// are discarded (production hot path) or collected for later writing.
type TraceSink interface {
	// Record appends one verdict for the event at the given timestamp.
	Record(timestamp int64, detector string, result Result)
	// Close signals the end of the stream. Implementations may flush here.
	Close()
}

// NoOpSink discards every record. It is the default sink when no report path is
// requested, so the replay loop can run unconditionally without allocating.
type NoOpSink struct{}

// NewNoOpSink returns a sink that discards all records.
func NewNoOpSink() *NoOpSink { return &NoOpSink{} }

func (n *NoOpSink) Record(int64, string, Result) {}
func (n *NoOpSink) Close()                        {}

// InMemoryCollector keeps verdict records in the order they were recorded, for
// later serialization by WriteCombinedReport.
type InMemoryCollector struct {
	records []TraceRecord
}

// NewInMemoryCollector returns an empty collector.
func NewInMemoryCollector() *InMemoryCollector {
	return &InMemoryCollector{records: make([]TraceRecord, 0)}
}

// Record appends a verdict, preserving stream order.
func (c *InMemoryCollector) Record(timestamp int64, detector string, result Result) {
	c.records = append(c.records, TraceRecord{
		Timestamp: timestamp,
		Detector:  detector,
		Result:    result,
	})
}

// Close is a no-op for the in-memory collector (nothing to flush).
func (c *InMemoryCollector) Close() {}

// Records returns the collected verdicts in stream order. The returned slice
// aliases internal state; callers must not mutate it.
func (c *InMemoryCollector) Records() []TraceRecord { return c.records }

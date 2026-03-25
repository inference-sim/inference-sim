package trace

// TraceLevel controls the verbosity of decision tracing.
type TraceLevel string

const (
	// TraceLevelNone disables tracing (zero overhead).
	TraceLevelNone TraceLevel = "none"
	// TraceLevelDecisions captures all admission and routing policy decisions.
	TraceLevelDecisions TraceLevel = "decisions"
)

// validTraceLevels maps accepted trace level strings.
var validTraceLevels = map[TraceLevel]bool{
	TraceLevelNone:      true,
	TraceLevelDecisions: true,
	"":                  true, // empty defaults to none
}

// IsValidTraceLevel returns true if the given level string is a recognized trace level.
func IsValidTraceLevel(level string) bool {
	return validTraceLevels[TraceLevel(level)]
}

// TraceConfig controls trace collection behavior.
type TraceConfig struct {
	Level           TraceLevel
	CounterfactualK int // number of counterfactual candidates per routing decision
}

// SimulationTrace collects decision records during a cluster simulation.
type SimulationTrace struct {
	Config          TraceConfig
	Admissions      []AdmissionRecord
	Routings        []RoutingRecord
	Disaggregations []DisaggregationRecord
	PrefillRoutings []PrefillRoutingRecord
	DecodeRoutings  []DecodeRoutingRecord
	KVTransfers     []KVTransferRecord
}

// NewSimulationTrace creates a SimulationTrace ready for recording.
func NewSimulationTrace(config TraceConfig) *SimulationTrace {
	return &SimulationTrace{
		Config:          config,
		Admissions:      make([]AdmissionRecord, 0),
		Routings:        make([]RoutingRecord, 0),
		Disaggregations: make([]DisaggregationRecord, 0),
		PrefillRoutings: make([]PrefillRoutingRecord, 0),
		DecodeRoutings:  make([]DecodeRoutingRecord, 0),
		KVTransfers:     make([]KVTransferRecord, 0),
	}
}

// RecordAdmission appends an admission decision record.
func (st *SimulationTrace) RecordAdmission(record AdmissionRecord) {
	st.Admissions = append(st.Admissions, record)
}

// RecordRouting appends a routing decision record.
func (st *SimulationTrace) RecordRouting(record RoutingRecord) {
	st.Routings = append(st.Routings, record)
}

// RecordDisaggregation appends a disaggregation decision record.
func (st *SimulationTrace) RecordDisaggregation(record DisaggregationRecord) {
	st.Disaggregations = append(st.Disaggregations, record)
}

// RecordPrefillRouting appends a prefill pool routing decision record.
func (st *SimulationTrace) RecordPrefillRouting(record PrefillRoutingRecord) {
	st.PrefillRoutings = append(st.PrefillRoutings, record)
}

// RecordDecodeRouting appends a decode pool routing decision record.
func (st *SimulationTrace) RecordDecodeRouting(record DecodeRoutingRecord) {
	st.DecodeRoutings = append(st.DecodeRoutings, record)
}

// RecordKVTransfer appends a KV transfer event record.
func (st *SimulationTrace) RecordKVTransfer(record KVTransferRecord) {
	st.KVTransfers = append(st.KVTransfers, record)
}

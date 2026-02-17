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
	Config     TraceConfig
	Admissions []AdmissionRecord
	Routings   []RoutingRecord
}

// NewSimulationTrace creates a SimulationTrace ready for recording.
func NewSimulationTrace(config TraceConfig) *SimulationTrace {
	return &SimulationTrace{
		Config:     config,
		Admissions: make([]AdmissionRecord, 0),
		Routings:   make([]RoutingRecord, 0),
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

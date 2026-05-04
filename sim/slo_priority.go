package sim

// SLOPriorityMap maps SLOClass strings to integer priorities.
// Higher = more important. Negative = sheddable (matches GAIE's IsSheddable contract).
// Unexported map field prevents external mutation (R8).
//
// Convention: BLIS and llm-d use higher integers for more-urgent requests.
// At the cluster→instance boundary, priorities are inverted via InvertForVLLM
// to match vLLM's lower-integer = more-urgent convention for instance-level scheduling.
type SLOPriorityMap struct {
	priorities map[string]int
	defaultPri int
	maxPri     int // computed once at construction for InvertForVLLM
}

// DefaultSLOPriorityMap returns the GAIE-compatible default priority mapping.
// critical=4, standard=3, batch=-1, sheddable=-2, background=-3.
// Empty or unknown class → 3 (Standard).
func DefaultSLOPriorityMap() *SLOPriorityMap {
	priorities := map[string]int{
		"critical":   4,
		"standard":   3,
		"batch":      -1,
		"sheddable":  -2,
		"background": -3,
	}
	defaultPri := 3
	maxPri := defaultPri
	for _, p := range priorities {
		if p > maxPri {
			maxPri = p
		}
	}
	return &SLOPriorityMap{
		priorities: priorities,
		defaultPri: defaultPri,
		maxPri:     maxPri,
	}
}

// NewSLOPriorityMap creates a priority map with defaults, then applies overrides.
// Nil or empty overrides → pure defaults. Override keys replace defaults for those classes.
func NewSLOPriorityMap(overrides map[string]int) *SLOPriorityMap {
	m := DefaultSLOPriorityMap()
	for k, v := range overrides {
		m.priorities[k] = v
	}
	// Recompute maxPri after applying overrides
	m.maxPri = m.defaultPri
	for _, p := range m.priorities {
		if p > m.maxPri {
			m.maxPri = p
		}
	}
	return m
}

// Priority returns the integer priority for the given SLO class.
// Unknown or empty class → default priority (3 = Standard).
func (m *SLOPriorityMap) Priority(class string) int {
	if p, ok := m.priorities[class]; ok {
		return p
	}
	return m.defaultPri
}

// IsSheddable returns true iff the class has priority < 0.
// Matches GAIE's util/request/sheddable.go:21 contract.
func (m *SLOPriorityMap) IsSheddable(class string) bool {
	return m.Priority(class) < 0
}

// InvertForVLLM converts a BLIS SLO class to a vLLM priority value.
// vLLM uses lower integers for higher urgency (min-heap), opposite of BLIS/llm-d.
// Returns maxPri - Priority(class), where maxPri is precomputed at construction.
//
// Example with defaults (maxPri=4):
//
//	critical (4) → 0, standard (3) → 1, batch (-1) → 5, sheddable (-2) → 6, background (-3) → 7
//
// Handles custom overrides: if an override sets critical=10, maxPri becomes 10
// and all inversions adjust accordingly.
//
// Used at two call sites:
//   - Simulator.EnqueueRequest / EnqueueDecodeSubRequest: instance-level pre-processor
//   - cmd/observe.go: HTTP body injection for real vLLM servers
func (m *SLOPriorityMap) InvertForVLLM(class string) int {
	return m.maxPri - m.Priority(class)
}

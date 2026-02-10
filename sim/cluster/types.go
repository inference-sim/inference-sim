package cluster

// Identity types
type InstanceID string
type ConfigID string
type ModelID string

// Pool and architecture types
type PoolType string

const (
	PoolMonolithic PoolType = "monolithic"
	PoolPrefill    PoolType = "prefill"
	PoolDecode     PoolType = "decode"
)

type ArchitectureType string

const (
	ArchitectureMonolithic       ArchitectureType = "monolithic"
	ArchitectureDisaggregatedPD  ArchitectureType = "disaggregated_pd"
)

// Event types with priority ordering
type EventType string

const (
	EventTypeRequestArrival   EventType = "RequestArrival"
	EventTypeRouteDecision    EventType = "RouteDecision"
	EventTypeInstanceStep     EventType = "InstanceStep"
	EventTypeRequestCompleted EventType = "RequestCompleted"
)

// EventTypePriority defines ordering for simultaneous events
// Lower values are processed first
var EventTypePriority = map[EventType]int{
	EventTypeRequestArrival:   1,
	EventTypeRouteDecision:    2,
	EventTypeInstanceStep:     3,
	EventTypeRequestCompleted: 4,
}

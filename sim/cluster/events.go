package cluster

import "sync/atomic"

// Global event ID counter for deterministic tie-breaking
var globalEventID uint64

// Event represents a simulation event
type Event interface {
	Timestamp() int64
	EventID() uint64
	Type() EventType
	Execute(sim *ClusterSimulator)
}

// BaseEvent provides common event fields
type BaseEvent struct {
	timestamp int64
	eventID   uint64
	eventType EventType
}

func newBaseEvent(timestamp int64, eventType EventType) BaseEvent {
	return BaseEvent{
		timestamp: timestamp,
		eventID:   atomic.AddUint64(&globalEventID, 1),
		eventType: eventType,
	}
}

func (e *BaseEvent) Timestamp() int64 {
	return e.timestamp
}

func (e *BaseEvent) EventID() uint64 {
	return e.eventID
}

func (e *BaseEvent) Type() EventType {
	return e.eventType
}

// RequestArrivalEvent represents a request arriving at the cluster
type RequestArrivalEvent struct {
	BaseEvent
	Request *Request
}

func NewRequestArrivalEvent(timestamp int64, req *Request) *RequestArrivalEvent {
	return &RequestArrivalEvent{
		BaseEvent: newBaseEvent(timestamp, EventTypeRequestArrival),
		Request:   req,
	}
}

func (e *RequestArrivalEvent) Execute(sim *ClusterSimulator) {
	sim.handleRequestArrival(e)
}

// RouteDecisionEvent represents routing a request to an instance
type RouteDecisionEvent struct {
	BaseEvent
	Request        *Request
	TargetInstance InstanceID
}

func NewRouteDecisionEvent(timestamp int64, req *Request, target InstanceID) *RouteDecisionEvent {
	return &RouteDecisionEvent{
		BaseEvent:      newBaseEvent(timestamp, EventTypeRouteDecision),
		Request:        req,
		TargetInstance: target,
	}
}

func (e *RouteDecisionEvent) Execute(sim *ClusterSimulator) {
	sim.handleRouteDecision(e)
}

// InstanceStepEvent represents an instance processing a batch
type InstanceStepEvent struct {
	BaseEvent
	InstanceID InstanceID
}

func NewInstanceStepEvent(timestamp int64, instanceID InstanceID) *InstanceStepEvent {
	return &InstanceStepEvent{
		BaseEvent:  newBaseEvent(timestamp, EventTypeInstanceStep),
		InstanceID: instanceID,
	}
}

func (e *InstanceStepEvent) Execute(sim *ClusterSimulator) {
	sim.handleInstanceStep(e)
}

// RequestCompletedEvent represents a request completing
type RequestCompletedEvent struct {
	BaseEvent
	Request    *Request
	InstanceID InstanceID
}

func NewRequestCompletedEvent(timestamp int64, req *Request, instanceID InstanceID) *RequestCompletedEvent {
	return &RequestCompletedEvent{
		BaseEvent:  newBaseEvent(timestamp, EventTypeRequestCompleted),
		Request:    req,
		InstanceID: instanceID,
	}
}

func (e *RequestCompletedEvent) Execute(sim *ClusterSimulator) {
	sim.handleRequestCompleted(e)
}

// Request represents a request in the system
// This extends the existing sim.Request with cluster-level fields
type Request struct {
	// Identity
	ID string

	// Timing
	ArrivalTime    int64
	RouteTime      int64
	EnqueueTime    int64
	ScheduleTime   int64
	FirstTokenTime int64
	CompletionTime int64

	// Routing
	TargetInstance InstanceID

	// State
	State RequestState

	// Request properties
	PromptTokens int
	OutputTokens int
}

// RequestState represents the lifecycle state of a request
type RequestState string

const (
	RequestStatePending   RequestState = "PENDING"
	RequestStateRouted    RequestState = "ROUTED"
	RequestStateQueued    RequestState = "QUEUED"
	RequestStateRunning   RequestState = "RUNNING"
	RequestStateCompleted RequestState = "COMPLETED"
)

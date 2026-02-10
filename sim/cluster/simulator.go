package cluster

import "fmt"

// ClusterSimulator represents the multi-replica cluster simulator
type ClusterSimulator struct {
	// Configuration
	Models      []*Model
	Deployments map[ConfigID]*DeploymentConfig
	Instances   map[InstanceID]*InstanceSimulator

	// Simulation state
	EventQueue *EventHeap
	Clock      int64
	Horizon    int64

	// Request tracking
	Requests          map[string]*Request
	PendingRequests   int
	CompletedRequests int

	// Determinism
	RNG           *PartitionedRNG
	nextEventID   uint64 // Per-simulator event counter for deterministic event ordering (BC-9)

	// Round-robin routing state (Phase 1 simple routing)
	nextInstanceIdx int
}

// NewClusterSimulator creates a new cluster simulator
func NewClusterSimulator(horizon int64) *ClusterSimulator {
	return &ClusterSimulator{
		Models:      make([]*Model, 0),
		Deployments: make(map[ConfigID]*DeploymentConfig),
		Instances:   make(map[InstanceID]*InstanceSimulator),
		EventQueue:  NewEventHeap(),
		Clock:       0,
		Horizon:     horizon,
		Requests:    make(map[string]*Request),
	}
}

// AddDeployment adds a deployment configuration to the cluster
func (c *ClusterSimulator) AddDeployment(config *DeploymentConfig) error {
	if config == nil {
		return fmt.Errorf("deployment config cannot be nil")
	}
	if _, exists := c.Deployments[config.ConfigID]; exists {
		return fmt.Errorf("deployment %s already exists", config.ConfigID)
	}

	c.Deployments[config.ConfigID] = config

	// Add instances from the replica pool
	if config.ReplicaPool != nil {
		for _, inst := range config.ReplicaPool.Instances {
			c.Instances[inst.ID] = inst
		}
	}

	return nil
}

// GetInstance retrieves an instance by ID
func (c *ClusterSimulator) GetInstance(id InstanceID) *InstanceSimulator {
	return c.Instances[id]
}

// ListInstances returns all instance IDs
func (c *ClusterSimulator) ListInstances() []InstanceID {
	ids := make([]InstanceID, 0, len(c.Instances))
	for id := range c.Instances {
		ids = append(ids, id)
	}
	return ids
}

// ScheduleEvent adds an event to the event queue
func (c *ClusterSimulator) ScheduleEvent(e Event) {
	c.EventQueue.Schedule(e)
}

// newEventID generates the next event ID for this simulator (BC-9 determinism)
func (c *ClusterSimulator) newEventID() uint64 {
	c.nextEventID++
	return c.nextEventID
}

// NewRequestArrivalEvent creates a new request arrival event
func (c *ClusterSimulator) NewRequestArrivalEvent(timestamp int64, req *Request) *RequestArrivalEvent {
	return NewRequestArrivalEvent(timestamp, req, c.newEventID())
}

// NewRouteDecisionEvent creates a new route decision event
func (c *ClusterSimulator) NewRouteDecisionEvent(timestamp int64, req *Request, targetInstance InstanceID) *RouteDecisionEvent {
	return NewRouteDecisionEvent(timestamp, req, targetInstance, c.newEventID())
}

// NewInstanceStepEvent creates a new instance step event
func (c *ClusterSimulator) NewInstanceStepEvent(timestamp int64, instanceID InstanceID) *InstanceStepEvent {
	return NewInstanceStepEvent(timestamp, instanceID, c.newEventID())
}

// NewRequestCompletedEvent creates a new request completed event
func (c *ClusterSimulator) NewRequestCompletedEvent(timestamp int64, req *Request, instanceID InstanceID) *RequestCompletedEvent {
	return NewRequestCompletedEvent(timestamp, req, instanceID, c.newEventID())
}

// Run executes the simulation until horizon or all requests complete
func (c *ClusterSimulator) Run() *ClusterMetrics {
	for c.EventQueue.Len() > 0 {
		event := c.EventQueue.PopNext()

		// Check horizon
		if event.Timestamp() > c.Horizon {
			break
		}

		// Update clock (BC-5: monotonicity)
		if event.Timestamp() < c.Clock {
			panic(fmt.Sprintf("Clock went backwards: %d < %d", event.Timestamp(), c.Clock))
		}
		c.Clock = event.Timestamp()

		// Execute event
		event.Execute(c)
	}

	return c.ComputeMetrics()
}

// Event handlers

func (c *ClusterSimulator) handleRequestArrival(e *RequestArrivalEvent) {
	req := e.Request
	req.State = RequestStatePending
	req.ArrivalTime = e.Timestamp()

	c.Requests[req.ID] = req
	c.PendingRequests++

	// Simple round-robin routing (Phase 1)
	if len(c.Instances) > 0 {
		instances := c.ListInstances()
		targetInstance := instances[c.nextInstanceIdx%len(instances)]
		c.nextInstanceIdx++

		// Schedule route decision immediately
		routeEvent := c.NewRouteDecisionEvent(e.Timestamp(), req, targetInstance)
		c.ScheduleEvent(routeEvent)
	}
}

func (c *ClusterSimulator) handleRouteDecision(e *RouteDecisionEvent) {
	req := e.Request
	req.State = RequestStateRouted
	req.RouteTime = e.Timestamp()
	req.TargetInstance = e.TargetInstance

	// Enqueue to target instance
	instance := c.GetInstance(e.TargetInstance)
	if instance != nil {
		instance.EnqueueRequest(req)
		req.State = RequestStateQueued
		req.EnqueueTime = e.Timestamp()

		// Schedule instance step if not already scheduled
		// (In full implementation, this would check if instance is idle)
		stepEvent := c.NewInstanceStepEvent(e.Timestamp()+1, e.TargetInstance)
		c.ScheduleEvent(stepEvent)
	}

	c.PendingRequests--
}

func (c *ClusterSimulator) handleInstanceStep(e *InstanceStepEvent) {
	instance := c.GetInstance(e.InstanceID)
	if instance == nil {
		return
	}

	// Stub: In full implementation, this would call instance.Step()
	// and process completed requests
	_, _ = instance.Step(e.Timestamp())
}

func (c *ClusterSimulator) handleRequestCompleted(e *RequestCompletedEvent) {
	req := e.Request
	req.State = RequestStateCompleted
	req.CompletionTime = e.Timestamp()

	c.CompletedRequests++

	// Verify causality (BC-8)
	if req.ArrivalTime > req.RouteTime ||
		req.RouteTime > req.EnqueueTime ||
		req.EnqueueTime > req.CompletionTime {
		panic(fmt.Sprintf("Causality violated for request %s", req.ID))
	}
}

// ComputeMetrics calculates final simulation metrics
func (c *ClusterSimulator) ComputeMetrics() *ClusterMetrics {
	metrics := &ClusterMetrics{
		CompletedRequests: c.CompletedRequests,
		TotalRequests:     len(c.Requests),
		SimDuration:       c.Clock,
		PerInstance:       make(map[InstanceID]*InstanceMetrics),
	}

	// Aggregate instance metrics
	for id, inst := range c.Instances {
		metrics.PerInstance[id] = &InstanceMetrics{
			CompletedRequests:  inst.CompletedRequests,
			TotalInputTokens:   inst.TotalInputTokens,
			TotalOutputTokens:  inst.TotalOutputTokens,
			PeakWaitQueueDepth: inst.PeakWaitQueueDepth,
			PeakBatchSize:      inst.PeakBatchSize,
		}

		metrics.TotalInputTokens += inst.TotalInputTokens
		metrics.TotalOutputTokens += inst.TotalOutputTokens
	}

	return metrics
}

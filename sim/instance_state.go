package sim

// InstanceState represents the lifecycle state of an instance (model server).
type InstanceState string

const (
	InstanceStateScheduling InstanceState = "Scheduling"
	InstanceStateLoading    InstanceState = "Loading"
	InstanceStateWarmingUp  InstanceState = "WarmingUp"
	InstanceStateActive     InstanceState = "Active"
	InstanceStateDraining   InstanceState = "Draining"
	InstanceStateTerminated InstanceState = "Terminated"
)

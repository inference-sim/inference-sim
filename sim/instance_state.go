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

// validInstanceStates is the unexported validation map (R8).
var validInstanceStates = map[InstanceState]struct{}{
	InstanceStateScheduling: {},
	InstanceStateLoading:    {},
	InstanceStateWarmingUp:  {},
	InstanceStateActive:     {},
	InstanceStateDraining:   {},
	InstanceStateTerminated: {},
}

// IsValidInstanceState returns true if s is a known InstanceState value.
func IsValidInstanceState(s string) bool {
	_, ok := validInstanceStates[InstanceState(s)]
	return ok
}

package cluster

// The list of control loops (SIM-14). A control loop runs beside the router and acts on the
// cluster: here, the model autoscaler. Each is a component (design section 8.1): it acts only
// on values published to it (DEV-1, DEV-4), and each is listed here with the upstream controller
// it models and its validation status and evidence (design 8.9), so a reader can tell a policy
// the model verified from one it validated.
//
// docs/reference/control-loops.md is generated from this list (TestControlLoopsPage).

// ValidationStatus says how far a control loop's behaviour has been checked against the system
// it models (design section 8.9).
type ValidationStatus string

const (
	// ValidationVerified: the code was checked against the upstream's code or a written rule, and
	// no measured cluster run was compared with it. Behaviour that appears only at cluster scale
	// stays here until measured cluster runs exist.
	ValidationVerified ValidationStatus = "verified, not validated"
	// ValidationValidated: held-out measured runs agree with it within a stated bound.
	ValidationValidated ValidationStatus = "validated"
)

// ControlLoop is one entry of the list.
type ControlLoop struct {
	// Name identifies the loop in a Result and in this list.
	Name string
	// PolicyField is the field of sim.PolicyBundle that configures it.
	PolicyField string
	// Component says which types make up the loop.
	Component string
	// Published lists the values the loop reads, all of which the cluster hands it as a copy
	// taken at the tick (sim.ClusterState).
	Published string
	// Upstream names the controller it models, or says it models none.
	Upstream string
	Status   ValidationStatus
	// Evidence says what supports the status.
	Evidence string
}

// ControlLoops returns the list, in name order.
func ControlLoops() []ControlLoop {
	return []ControlLoop{{
		Name:        "model-autoscaler",
		PolicyField: "Autoscaler",
		Component:   "Collector, Analyzer, Engine and Actuator, run by the scaling tick (autoscaler.go)",
		Published:   "per-instance status copied into sim.ClusterState at each tick (queue depth, batch size, KV use, in-flight count), not the instances themselves",
		Upstream:    "the llm-d workload-variant-autoscaler (its saturation model); its README (https://github.com/llm-d/llm-d-workload-variant-autoscaler#readme, read for this list in November 2026) says it is deprecated in favour of a target-value rule on a router metric",
		Status:      ValidationVerified,
		Evidence:    "the pipeline is tested against its written contracts (autoscaler_test.go, autoscaler_invariant_test.go); no measured autoscaling run has been compared with it, and the upstream it follows is deprecated",
	}}
}

// controlLoopByPolicyField returns the entry for a PolicyBundle field, if the field configures
// a control loop.
func controlLoopByPolicyField(field string) (ControlLoop, bool) {
	for _, l := range ControlLoops() {
		if l.PolicyField == field {
			return l, true
		}
	}
	return ControlLoop{}, false
}

package cluster

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// policyFieldsThatAreNotControlLoops are the PolicyBundle fields that configure no control loop,
// each with the reason. A new field of the bundle is in neither this map nor the list of control
// loops until someone decides which it is (SIM-14).
var policyFieldsThatAreNotControlLoops = map[string]string{
	"Priority":          "the engine's request priority, applied inside a step",
	"NodePools":         "the infrastructure the cluster runs on, not a policy that acts on it",
	"InstanceLifecycle": "how long an instance takes to load, a timing of the infrastructure",
}

// Every configurable control policy has an entry in the list of control loops (SIM-14): each
// field of the policy bundle is either a listed control loop or declared not to be one, and each
// listed loop names a field the bundle has.
func TestControlLoops_EveryConfigurablePolicyIsListedOrDeclaredNotALoop(t *testing.T) {
	bundle := reflect.TypeOf(sim.PolicyBundle{})
	fields := map[string]bool{}
	for i := 0; i < bundle.NumField(); i++ {
		name := bundle.Field(i).Name
		fields[name] = true
		_, listed := controlLoopByPolicyField(name)
		_, declared := policyFieldsThatAreNotControlLoops[name]
		switch {
		case listed && declared:
			t.Errorf("PolicyBundle.%s is both a listed control loop and declared not to be one", name)
		case !listed && !declared:
			t.Errorf("PolicyBundle.%s is neither in the list of control loops (control_loops.go) nor declared not to be one: "+
				"a policy that acts on the cluster beside the router is a control loop (SIM-14)", name)
		}
	}
	for _, l := range ControlLoops() {
		if !fields[l.PolicyField] {
			t.Errorf("control loop %s names PolicyBundle.%s, which does not exist", l.Name, l.PolicyField)
		}
		if l.Name == "" || l.Component == "" || l.Published == "" || l.Upstream == "" || l.Evidence == "" || l.Status == "" {
			t.Errorf("control loop %q leaves a column empty: the list states the upstream, the status and the evidence for each", l.Name)
		}
		if l.Status != ValidationVerified && l.Status != ValidationValidated {
			t.Errorf("control loop %s has the unknown validation status %q", l.Name, l.Status)
		}
	}
	for name := range policyFieldsThatAreNotControlLoops {
		if !fields[name] {
			t.Errorf("policyFieldsThatAreNotControlLoops names PolicyBundle.%s, which does not exist", name)
		}
	}
}

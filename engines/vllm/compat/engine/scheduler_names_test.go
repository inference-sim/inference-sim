// Original to blis-vllm.

package engine

import "testing"

// The name tables are read-only to every caller: editing the table a NameTable call returns
// changes neither the next call's table nor which names are valid (R8).
func TestNameTablesCannotBeEditedFromOutside(t *testing.T) {
	s := SchedulerNameTable()
	s["made-up"] = true
	delete(s, "fcfs")
	if IsValidScheduler("made-up") || !IsValidScheduler("fcfs") || SchedulerNameTable()["made-up"] || !SchedulerNameTable()["fcfs"] {
		t.Error("editing SchedulerNameTable's result changed the scheduler names")
	}
	p := PreemptionPolicyNameTable()
	p["made-up"] = true
	delete(p, "priority")
	if PreemptionPolicyNameTable()["made-up"] || !PreemptionPolicyNameTable()["priority"] {
		t.Error("editing PreemptionPolicyNameTable's result changed the preemption policy names")
	}
}

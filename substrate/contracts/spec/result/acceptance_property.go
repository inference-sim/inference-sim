package result

import (
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
)

// SpeculativeAcceptance is the name of the workload property a run's speculative acceptance is
// recorded under.
const SpeculativeAcceptance = "speculative_acceptance"

// AcceptanceProperty builds the Result's record of a run's speculative acceptance from the
// resolution that supplied it (design D8, SIM-10). It is here, beside the record's type, so the
// simulator that writes it and the harness that reads it agree on every field.
//
// The values are the acceptance length and the acceptance rate. The source follows the layer:
// the command line, the scenario or the workload when the value was stated, or profile when a
// registry or scenario profile supplied it. A profile lookup records the profile it used, as
// set and key; and, because it can borrow, how it found the value: the step, the key fields that
// differed, how K was brought to the run's, and every profile it read. A stated value records
// none of those.
func AcceptanceProperty(r acceptance.Resolution) WorkloadProperty {
	w := WorkloadProperty{
		Name: SpeculativeAcceptance,
		Values: map[string]float64{
			"acceptance_length": r.AcceptanceLength,
			"acceptance_rate":   r.AcceptanceRate(),
		},
	}
	switch r.Layer {
	case acceptance.LayerCLI:
		w.Source = PropertyCommandLine
	case acceptance.LayerScenario:
		w.Source = PropertyScenario
	case acceptance.LayerWorkload:
		w.Source = PropertyWorkload
	default:
		w.Source = PropertyProfile
	}
	if r.Profile == nil {
		return w // a stated value: it came from no profile
	}
	w.Source = PropertyProfile
	w.Profile = r.Set + " " + r.Profile.Key().String()
	w.Step = string(r.Step)
	for _, u := range r.Used {
		w.Profiles = append(w.Profiles, u.Set+" "+u.Profile.Key().String())
	}
	if r.Borrowed() {
		w.Differed = append([]string(nil), r.Differed...)
		w.KRule = string(r.KRule)
	}
	return w
}

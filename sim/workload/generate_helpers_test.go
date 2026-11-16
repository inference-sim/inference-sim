package workload

import "github.com/inference-sim/inference-sim/sim"

// generatedWorkload is everything the workload generator yields for a spec: the requests its
// streaming source emits, in emission order, with the session blueprints and the follow-up budget.
type generatedWorkload struct {
	Requests       []*sim.Request
	Sessions       []SessionBlueprint
	FollowUpBudget int64
}

// generateWorkload drains the streaming source GenerateWorkloadLazy builds for spec, so a test
// can assert on the whole request sequence. Streaming is the only generation path (#1451); a
// sampler error the source records during the drain is returned as the generator's error.
func generateWorkload(spec *WorkloadSpec, horizon, maxRequests int64) (*generatedWorkload, error) {
	src, sessions, budget, err := GenerateWorkloadLazy(spec, horizon, maxRequests)
	if err != nil {
		return nil, err
	}
	var reqs []*sim.Request
	for {
		r, ok := src.Next()
		if !ok {
			break
		}
		reqs = append(reqs, r)
	}
	if err := src.Err(); err != nil {
		return nil, err
	}
	return &generatedWorkload{Requests: reqs, Sessions: sessions, FollowUpBudget: budget}, nil
}

// generateRequests is the request sequence generateWorkload yields for spec.
func generateRequests(spec *WorkloadSpec, horizon, maxRequests int64) ([]*sim.Request, error) {
	wl, err := generateWorkload(spec, horizon, maxRequests)
	if err != nil {
		return nil, err
	}
	return wl.Requests, nil
}

package cluster

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// completionEvent is one completion as an instance's own OnRequestDone callback saw it.
type completionEvent struct {
	id, instance string
	tick         int64
}

// recordCompletions wraps every instance's completion callback (after the cluster installed
// its own) so the test sees completions in the global order the cluster processed them.
func recordCompletions(cs *ClusterSimulator) *[]completionEvent {
	var seen []completionEvent
	for _, inst := range cs.instances {
		inst := inst
		prev := inst.sim.OnRequestDone
		inst.sim.OnRequestDone = func(req *sim.Request, tick int64) []*sim.Request {
			if req.State == sim.StateCompleted {
				seen = append(seen, completionEvent{req.ID, string(inst.ID()), tick})
			}
			if prev != nil {
				return prev(req, tick)
			}
			return nil
		}
	}
	return &seen
}

// completionIndexOf writes the run's metrics file and returns each request's completion_index.
func completionIndexOf(t *testing.T, cs *ClusterSimulator) map[string]int {
	t.Helper()
	path := filepath.Join(t.TempDir(), "metrics.json")
	m := cs.AggregatedMetrics()
	devnull, err := os.Open(os.DevNull)
	if err != nil {
		t.Fatal(err)
	}
	orig := os.Stdout
	os.Stdout = devnull
	emitErr := m.EmitOutput(m.BuildOutput("cluster"), path)
	os.Stdout = orig
	_ = devnull.Close()
	if emitErr != nil {
		t.Fatal(emitErr)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var out sim.MetricsOutput
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	idx := map[string]int{}
	for _, r := range out.Requests {
		idx[r.ID] = r.CompletionIndex
	}
	return idx
}

// sameShapeRequests is n requests arriving together in waves of three, with IDs that sort
// opposite to their arrival order, so round-robin spreads each wave over three instances that
// complete it at one clock -- a tie an ID or completion-time sort would get wrong.
func sameShapeRequests(n int) []*sim.Request {
	var reqs []*sim.Request
	for i := 0; i < n; i++ {
		reqs = append(reqs, &sim.Request{
			ID: fmt.Sprintf("r%02d", 99-i), ArrivalTime: int64(i/3) * 5000,
			InputTokens: make([]sim.TokenID, 64), OutputTokens: make([]sim.TokenID, 4+(i/3)%3),
			State: sim.StateQueued,
		})
	}
	return reqs
}

// Across a multi-instance cluster completion_index is the global order the cluster processed
// completions in: each instance's callback order, interleaved by the global clock. Non-vacuity:
// some same-clock completions on different instances complete in an order other than ID order.
func TestCompletionIndex_IsTheClustersCompletionOrder(t *testing.T) {
	cfg := newTestDeploymentConfig(3)
	cfg.RoutingPolicy = "round-robin"
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(sameShapeRequests(12)), nil)
	seen := recordCompletions(cs)
	mustRun(t, cs)
	idx := completionIndexOf(t, cs)

	if len(*seen) != 12 {
		t.Fatalf("%d of 12 requests completed", len(*seen))
	}
	instances := map[string]bool{}
	reordered := false
	for i, e := range *seen {
		if got := idx[e.id]; got != i+1 {
			t.Errorf("%s (completed %d-th, on %s at %d) has completion_index %d", e.id, i+1, e.instance, e.tick, got)
		}
		instances[e.instance] = true
		if i > 0 {
			p := (*seen)[i-1]
			if p.tick > e.tick {
				t.Errorf("the cluster completed %s at %d after %s at %d", e.id, e.tick, p.id, p.tick)
			}
			if p.tick == e.tick && p.instance != e.instance && p.id > e.id {
				reordered = true
			}
		}
	}
	if len(instances) < 2 {
		t.Fatalf("only %d instance served; the case does not interleave instances", len(instances))
	}
	if !reordered {
		t.Error("no same-clock completions on different instances ran against ID order; an ID sort would pass")
	}
}

// A disaggregated parent takes its decode completion's place in the order: every served parent
// carries a completion_index, the indices are 1..n, and they follow the order the decode
// instance completed the decode sub-requests in.
func TestCompletionIndex_ADisaggregatedParentIsOrderedByItsDecode(t *testing.T) {
	cfg := newTestDisaggDeploymentConfig(4, 2, 2)
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(sameShapeRequests(9)), nil)
	seen := recordCompletions(cs)
	mustRun(t, cs)
	idx := completionIndexOf(t, cs)

	parentOf := map[string]string{}
	for _, p := range cs.ParentRequests() {
		if p.DecodeSubReq != nil {
			parentOf[p.DecodeSubReq.ID] = p.ID
		}
	}
	var order []string
	for _, e := range *seen {
		if pid, ok := parentOf[e.id]; ok {
			order = append(order, pid)
		}
	}
	if len(order) != 9 {
		t.Fatalf("%d of 9 parents completed through decode", len(order))
	}
	for i, pid := range order {
		if got := idx[pid]; got != i+1 {
			t.Errorf("parent %s completed %d-th through decode, completion_index %d", pid, i+1, got)
		}
	}
}

// Original to blis-vllm.

package engine

import (
	"reflect"
	"testing"
)

// route sends the ids through the balancer's clients in turn and returns each one's rank.
func route(lb *LoadBalancer, ids ...string) []int {
	out := make([]int, len(ids))
	for i, id := range ids {
		out[i] = lb.Choose(id)
	}
	return out
}

// With api_server_count API servers, each balances with its own scan start, n*client_index/
// client_count, and bumps its own snapshot: on four idle ranks, two clients take ranks 0 and 2 on
// their first requests, then the next rank of each scan (0, 2, 1, 3). One client takes 0, 1, 2, 3.
func TestTwoClientsStartTheirScansApart(t *testing.T) {
	two := NewLoadBalancers(4, 2)
	if got, want := route(two, "a", "b", "c", "d"), []int{0, 2, 1, 3}; !reflect.DeepEqual(got, want) {
		t.Errorf("two clients over four idle ranks routed %v, want %v", got, want)
	}
	one := NewLoadBalancer(4)
	if got, want := route(one, "a", "b", "c", "d"), []int{0, 1, 2, 3}; !reflect.DeepEqual(got, want) {
		t.Errorf("one client over four idle ranks routed %v, want %v", got, want)
	}
}

// A client scales its own in-flight count by the client count, to estimate the group's total
// (max(client_count x in_flight, waiting + running), core_client.py:1786@4ba787ec). Two ranks, two
// clients, every request through client 0: a goes to rank 0, b to rank 1 (rank 0 now scores 2),
// then b ends and the coordinator publishes rank 1 running one request. Client 0 holds one
// request on rank 0, so it scores rank 0 at 2 x 1 and rank 1 at 1, and picks rank 1. A lone
// client scores both at 1 and keeps its scan's first, rank 0.
func TestAClientScalesItsInFlightCountByTheClientCount(t *testing.T) {
	run := func(clients int) int {
		lb := NewLoadBalancers(2, clients)
		if r := lb.ChooseVia(0, "a"); r != 0 {
			t.Fatalf("a went to rank %d, want 0", r)
		}
		if r := lb.ChooseVia(0, "b"); r != 1 {
			t.Fatalf("b went to rank %d, want 1", r)
		}
		lb.Finished("b")
		lb.Publish([][3]float64{{0, 0, 0}, {0, 1, 0}})
		return lb.ChooseVia(0, "c")
	}
	if got := run(2); got != 1 {
		t.Errorf("with two clients the scaled in-flight count put c on rank %d, want rank 1", got)
	}
	if got := run(1); got != 0 {
		t.Errorf("with one client c went to rank %d, want rank 0 (equal scores keep the scan's first)", got)
	}
}

// Each client's counts, in-flight totals and scan start are its own: a request a client routed
// leaves only that client's in-flight count when it finishes, and a publish rebinds every client.
func TestFinishingARequestLeavesOnlyItsClientsCount(t *testing.T) {
	lb := NewLoadBalancers(2, 2)
	lb.ChooseVia(0, "a") // rank 0, client 0
	lb.ChooseVia(1, "b") // client 1 starts at rank 1
	lb.Finished("a")
	lb.Publish([][3]float64{{}, {}})
	// Client 1 still holds b on rank 1 (scaled 2); client 0 holds nothing. Client 1 scans from
	// its rotated start, rank 0 (1+1 mod 2), where it scores 0 against rank 1's 2.
	if got := lb.ChooseVia(1, "c"); got != 0 {
		t.Errorf("client 1 routed c to rank %d, want rank 0 (its own in-flight on rank 1 is 1 x 2)", got)
	}
}

// Requests reach the clients in turn, so a group of three clients over three ranks starts its
// scans at ranks 0, 1 and 2 and spreads three simultaneous arrivals over the three ranks.
func TestRequestsReachTheClientsInTurn(t *testing.T) {
	lb := NewLoadBalancers(3, 3)
	if got, want := route(lb, "a", "b", "c"), []int{0, 1, 2}; !reflect.DeepEqual(got, want) {
		t.Errorf("three clients routed %v, want %v", got, want)
	}
	if lb.ClientCount() != 3 {
		t.Errorf("ClientCount %d, want 3", lb.ClientCount())
	}
}

// The count is set before the first request; afterwards the clients' state would be orphaned.
func TestTheClientCountIsSetBeforeTheFirstRequest(t *testing.T) {
	lb := NewLoadBalancer(2)
	if err := lb.SetClientCount(2); err != nil {
		t.Fatal(err)
	}
	lb.Choose("a")
	if err := lb.SetClientCount(3); err == nil {
		t.Error("the client count changed after a request was routed")
	}
}

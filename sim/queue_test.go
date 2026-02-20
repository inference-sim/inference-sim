package sim

import (
	"sort"
	"testing"
)

func TestWaitQueue_Peek_NonEmpty_ReturnsFront(t *testing.T) {
	// GIVEN a queue with requests [A, B]
	wq := &WaitQueue{}
	reqA := &Request{ID: "A"}
	reqB := &Request{ID: "B"}
	wq.Enqueue(reqA)
	wq.Enqueue(reqB)

	// WHEN Peek() is called
	got := wq.Peek()

	// THEN it returns the front element without removing it
	if got != reqA {
		t.Errorf("Peek: got request %v, want %v", got.ID, reqA.ID)
	}
	if wq.Len() != 2 {
		t.Errorf("Peek modified queue length: got %d, want 2", wq.Len())
	}
}

func TestWaitQueue_Peek_Empty_ReturnsNil(t *testing.T) {
	// GIVEN an empty queue
	wq := &WaitQueue{}

	// WHEN Peek() is called
	got := wq.Peek()

	// THEN it returns nil
	if got != nil {
		t.Errorf("Peek on empty queue: got %v, want nil", got)
	}
}

func TestWaitQueue_PrependFront_InsertsAtFront(t *testing.T) {
	// GIVEN a queue with requests [A, B, C]
	wq := &WaitQueue{}
	reqA := &Request{ID: "A"}
	reqB := &Request{ID: "B"}
	reqC := &Request{ID: "C"}
	wq.Enqueue(reqA)
	wq.Enqueue(reqB)
	wq.Enqueue(reqC)

	// WHEN PrependFront(X) is called
	reqX := &Request{ID: "X"}
	wq.PrependFront(reqX)

	// THEN Peek() returns X and Len() increased by 1
	if wq.Peek() != reqX {
		t.Errorf("PrependFront: Peek() got %v, want X", wq.Peek().ID)
	}
	if wq.Len() != 4 {
		t.Errorf("PrependFront: Len() got %d, want 4", wq.Len())
	}

	// Verify full order by dequeuing all
	ids := make([]string, 0, 4)
	for wq.Len() > 0 {
		ids = append(ids, wq.DequeueBatch().ID)
	}
	want := []string{"X", "A", "B", "C"}
	for i, id := range ids {
		if id != want[i] {
			t.Errorf("PrependFront order[%d]: got %s, want %s", i, id, want[i])
		}
	}
}

func TestWaitQueue_PrependFront_OnEmpty(t *testing.T) {
	// GIVEN an empty queue
	wq := &WaitQueue{}

	// WHEN PrependFront(X) is called
	reqX := &Request{ID: "X"}
	wq.PrependFront(reqX)

	// THEN Peek() returns X and Len() is 1
	if wq.Peek() != reqX {
		t.Errorf("PrependFront on empty: Peek() got %v, want X", wq.Peek())
	}
	if wq.Len() != 1 {
		t.Errorf("PrependFront on empty: Len() got %d, want 1", wq.Len())
	}
}

func TestWaitQueue_Items_ReturnsContents(t *testing.T) {
	// GIVEN a queue with requests [A, B, C]
	wq := &WaitQueue{}
	reqA := &Request{ID: "A"}
	reqB := &Request{ID: "B"}
	reqC := &Request{ID: "C"}
	wq.Enqueue(reqA)
	wq.Enqueue(reqB)
	wq.Enqueue(reqC)

	// WHEN Items() is called
	items := wq.Items()

	// THEN it returns [A, B, C] in order
	if len(items) != 3 {
		t.Fatalf("Items: got %d elements, want 3", len(items))
	}
	wantIDs := []string{"A", "B", "C"}
	for i, req := range items {
		if req.ID != wantIDs[i] {
			t.Errorf("Items[%d]: got %s, want %s", i, req.ID, wantIDs[i])
		}
	}
}

func TestWaitQueue_Items_EmptyQueue(t *testing.T) {
	// GIVEN an empty queue
	wq := &WaitQueue{}

	// WHEN Items() is called
	items := wq.Items()

	// THEN it returns an empty (or nil) slice
	if len(items) != 0 {
		t.Errorf("Items on empty queue: got %d elements, want 0", len(items))
	}
}

func TestWaitQueue_Reorder_AppliesFunction(t *testing.T) {
	// GIVEN a queue with requests [C, A, B] (arrival order)
	wq := &WaitQueue{}
	wq.Enqueue(&Request{ID: "C", ArrivalTime: 300})
	wq.Enqueue(&Request{ID: "A", ArrivalTime: 100})
	wq.Enqueue(&Request{ID: "B", ArrivalTime: 200})

	// WHEN Reorder is called with a function that sorts by arrival time
	wq.Reorder(func(reqs []*Request) {
		sort.SliceStable(reqs, func(i, j int) bool {
			return reqs[i].ArrivalTime < reqs[j].ArrivalTime
		})
	})

	// THEN the queue order is [A, B, C] and length is preserved
	if wq.Len() != 3 {
		t.Fatalf("Reorder changed length: got %d, want 3", wq.Len())
	}
	items := wq.Items()
	wantIDs := []string{"A", "B", "C"}
	for i, req := range items {
		if req.ID != wantIDs[i] {
			t.Errorf("Reorder result[%d]: got %s, want %s", i, req.ID, wantIDs[i])
		}
	}
}

func TestWaitQueue_Reorder_EmptyQueue_NoOp(t *testing.T) {
	// GIVEN an empty queue
	wq := &WaitQueue{}
	called := false

	// WHEN Reorder is called
	wq.Reorder(func(reqs []*Request) {
		called = true
	})

	// THEN the function is still called (with empty slice) and queue remains empty
	if !called {
		t.Error("Reorder did not call the function on empty queue")
	}
	if wq.Len() != 0 {
		t.Errorf("Reorder on empty queue changed length: got %d, want 0", wq.Len())
	}
}

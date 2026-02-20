package sim

import (
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

package sim

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// completionIndexRun drives a small single-instance run in which several requests finish in
// the same step, recording the order the simulator completed them in through its own
// OnRequestDone seam, and returns that order with the --metrics-path file's requests and the
// stdout EmitOutput printed.
func completionIndexRun(t *testing.T) (done []string, file []RequestMetrics, stdout string) {
	t.Helper()
	cfg := newTestSimConfig()
	cfg.BatchConfig = NewBatchConfig(4, 2048, 0) // a cap of 4 makes requests queue and retire in waves
	s := mustNewSimulator(t, cfg)
	s.OnRequestDone = func(req *Request, _ int64) []*Request {
		if req.State == StateCompleted {
			done = append(done, req.ID)
		}
		return nil
	}
	// Equal output lengths in pairs, so pairs finish in the same step: a completion-time sort
	// alone cannot order them.
	for i, out := range []int{3, 3, 5, 5, 2, 2, 7, 7, 4, 4, 6, 6} {
		s.InjectArrival(newTestRequest(string(rune('a'+i)), int64(i%3), 16, out))
	}
	s.Run()

	path := filepath.Join(t.TempDir(), "metrics.json")
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	orig := os.Stdout
	os.Stdout = w
	emitErr := s.Metrics.EmitOutput(s.Metrics.BuildOutput("test"), path)
	os.Stdout = orig
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if _, err := io.Copy(&buf, r); err != nil {
		t.Fatal(err)
	}
	if emitErr != nil {
		t.Fatalf("EmitOutput: %v", emitErr)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var out MetricsOutput
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return done, out.Requests, buf.String()
}

// completion_index reproduces the order the simulator completed requests in -- the order a
// warm-up cut by completion needs (#1902) -- including among requests that finished in the
// same step.
func TestCompletionIndexIsTheSimulatorsCompletionOrder(t *testing.T) {
	done, reqs, _ := completionIndexRun(t)
	if len(done) != 12 {
		t.Fatalf("%d of 12 requests completed", len(done))
	}
	byIndex := append([]RequestMetrics(nil), reqs...)
	sort.Slice(byIndex, func(i, j int) bool { return byIndex[i].CompletionIndex < byIndex[j].CompletionIndex })
	got := make([]string, 0, len(byIndex))
	for i, r := range byIndex {
		if r.CompletionIndex != i+1 {
			t.Fatalf("completion indices are not 1..n: position %d holds %d", i+1, r.CompletionIndex)
		}
		got = append(got, r.ID)
	}
	for i := range done {
		if got[i] != done[i] {
			t.Fatalf("completion_index order %v, simulator completed %v", got, done)
		}
	}
}

// The field is file-only: stdout stays byte-identical to a build without it (INV-6).
func TestCompletionIndexStaysOffStdout(t *testing.T) {
	_, _, stdout := completionIndexRun(t)
	if bytes.Contains([]byte(stdout), []byte("completion_index")) {
		t.Error("completion_index reached stdout")
	}
}

// Across instances the processing clock decides; simultaneous completions are ordered by
// instance index and then by that instance's own sequence, and an unstamped request gets no
// index.
func TestCompletionIndicesAcrossInstances(t *testing.T) {
	m := NewMetrics()
	rs := []RequestMetrics{
		{ID: "a", HandledBy: "instance_0", completionSeq: 2, completionClock: 20},
		{ID: "b", HandledBy: "instance_10", completionSeq: 1, completionClock: 10},
		{ID: "c", HandledBy: "instance_2", completionSeq: 1, completionClock: 10},
		{ID: "d", HandledBy: "instance_10", completionSeq: 2, completionClock: 10},
		{ID: "e", HandledBy: "instance_0"}, // never completed
	}
	m.assignCompletionIndices(rs)
	want := map[string]int{"c": 1, "b": 2, "d": 3, "a": 4, "e": 0}
	for _, r := range rs {
		if r.CompletionIndex != want[r.ID] {
			t.Errorf("%s: completion_index %d, want %d", r.ID, r.CompletionIndex, want[r.ID])
		}
	}
}

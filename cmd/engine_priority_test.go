package cmd

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// priorityOrderRequests are the trace's requests by id (the trace format numbers them).
var priorityOrderRequests = []struct {
	name, class string
	at          int
}{{"first", "standard", 0}, {"standard", "standard", 1000}, {"background", "background", 2000}, {"critical", "critical", 3000}}

// priorityOrderTrace is a trace of four requests that contend for a one-sequence engine: "first"
// arrives alone and holds the sequence slot, and the other three arrive while it runs, in the
// order standard, background, critical (arrival order is the opposite of urgency order). Every
// request has the same shape, so only the queue order can separate their service order.
func priorityOrderTrace(t *testing.T) (header, data string) {
	t.Helper()
	dir := t.TempDir()
	header, data = filepath.Join(dir, "t.yaml"), filepath.Join(dir, "t.csv")
	if err := os.WriteFile(header, []byte("trace_version: 4\ntime_unit: microseconds\nmode: generated\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	body := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length," +
		"streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio," +
		"model,deadline_us,arrival_time_us,first_chunk_time_us,last_chunk_time_us,status\n"
	for i, r := range priorityOrderRequests {
		body += fmt.Sprintf("%d,c-%s,t1,%s,,0,,0,false,512,64,512,0,0,0,0.0,,0,%d,0,0,ok\n", i, r.name, r.class, r.at)
	}
	if err := os.WriteFile(data, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return header, data
}

// serviceOrder replays the trace on the one-sequence engine stating policy (with an optional
// --policy-config bundle) and returns the request ids by the time the engine finished them.
func serviceOrder(t *testing.T, policy, bundle string) []string {
	t.Helper()
	header, data := priorityOrderTrace(t)
	dir := writeColocatedVariant(t, "      max_model_len: 32768\n",
		"      max_model_len: 32768\n      max_num_seqs: 1\n      scheduling_policy: "+policy+"\n")
	prefix := filepath.Join(t.TempDir(), "out")
	args := []string{"replay", "--scenarios", dir, "--scenario", "variant.yaml", "--trace-header", header,
		"--trace-data", data, "--seed", "1", "--trace-output", prefix}
	if bundle != "" {
		args = append(args, "--policy-config", bundle)
	}
	stdout, stderr, err := runKernelCLI(t, args...)
	if err != nil {
		t.Fatalf("scheduling_policy %s: %v\n%s", policy, err, lastLines(stderr, 4))
	}
	if !completedIs(stdout, 4) {
		t.Fatalf("scheduling_policy %s: want all four requests completed\n%s", policy, lastLines(stdout, 3))
	}
	f, err := os.Open(prefix + ".csv")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	rows, err := csv.NewReader(f).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	col := map[string]int{}
	for i, h := range rows[0] {
		col[h] = i
	}
	type done struct {
		id string
		at int64
	}
	var finished []done
	for _, r := range rows[1:] {
		at, err := strconv.ParseInt(r[col["last_chunk_time_us"]], 10, 64)
		if err != nil || at <= 0 {
			t.Fatalf("request %s has no finish time in the exported trace: %q", r[col["request_id"]], r[col["last_chunk_time_us"]])
		}
		n, err := strconv.Atoi(r[col["request_id"]])
		if err != nil || n < 0 || n >= len(priorityOrderRequests) {
			t.Fatalf("exported request id %q is not one of the trace's", r[col["request_id"]])
		}
		finished = append(finished, done{priorityOrderRequests[n].name, at})
	}
	slices.SortFunc(finished, func(a, b done) int { return int(a.at - b.at) })
	var ids []string
	for _, d := range finished {
		ids = append(ids, d.id)
	}
	return ids
}

// Request priority reaches the engine (#2052). Under engine.scheduling_policy priority the queue
// is served by SLO class, most urgent first, not by arrival: critical, then standard, then
// background (the class map inverted for vLLM, where lower is more urgent). Under fcfs the same
// trace is served by arrival, so the order above is the policy's doing.
func TestReplayCmd_PrioritySchedulingServesTheQueueBySLOClass(t *testing.T) {
	if got, want := strings.Join(serviceOrder(t, "fcfs", ""), ","), "first,standard,background,critical"; got != want {
		t.Errorf("fcfs: service order %s, want arrival order %s", got, want)
	}
	if got, want := strings.Join(serviceOrder(t, "priority", ""), ","), "first,critical,standard,background"; got != want {
		t.Errorf("priority: service order %s, want most urgent class first: %s", got, want)
	}
}

// A --policy-config priority.slo_priorities override re-ranks the classes the engine orders by:
// raising background above critical serves background first among the waiting requests, and the
// same override does nothing under fcfs.
func TestReplayCmd_PrioritySchedulingFollowsAnSLOPriorityOverride(t *testing.T) {
	bundle := filepath.Join(t.TempDir(), "policy.yaml")
	if err := os.WriteFile(bundle, []byte("priority:\n  slo_priorities:\n    background: 9\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, want := strings.Join(serviceOrder(t, "priority", bundle), ","), "first,background,critical,standard"; got != want {
		t.Errorf("priority with background raised to 9: service order %s, want %s", got, want)
	}
	if got, want := strings.Join(serviceOrder(t, "fcfs", bundle), ","), "first,standard,background,critical"; got != want {
		t.Errorf("fcfs with an override: service order %s, want arrival order %s", got, want)
	}
}

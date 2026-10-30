package cmd

// Test helpers for the Result document (#1871): stdout of run and replay is one blis-schemas
// Result, so every stdout-reading test parses it with blis-schemas' own reader rather than
// scanning text.

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/result"
)

// parseResult reads stdout as one Result document, failing the test when it is anything else.
func parseResult(t *testing.T, stdout string) *result.Result {
	t.Helper()
	var r result.Result
	if err := result.Unmarshal([]byte(stdout), &r); err != nil {
		t.Fatalf("stdout is not one Result document: %v\n%s", err, stdout)
	}
	return &r
}

// clusterFigures are the whole-run, cluster-wide figures a run reported under the names the
// MetricsOutput JSON used, read from the Result: the INV-1 ledger and the engine's cluster
// figures. Tests that compared these names keep reading them by name.
func clusterFigures(t *testing.T, stdout string) map[string]float64 {
	t.Helper()
	r := parseResult(t, stdout)
	q := r.Requests
	out := map[string]float64{
		"completed_requests":        float64(q.Completed),
		"injected_requests":         float64(q.Injected),
		"still_queued":              float64(q.StillQueued),
		"still_running":             float64(q.StillRunning),
		"dropped_unservable":        float64(q.DroppedUnservable),
		"timed_out_requests":        float64(q.TimedOut),
		"length_capped_requests":    float64(q.LengthCapped),
		"rejected_requests":         float64(q.Rejected),
		"routing_rejections":        float64(q.RoutingRejections),
		"gateway_queue_depth":       float64(q.GatewayQueueDepth),
		"gateway_queue_rejected":    float64(q.GatewayQueueRejected),
		"gateway_evicted":           float64(q.GatewayEvicted),
		"gateway_expired":           float64(q.GatewayExpired),
		"encode_routing_rejections": float64(q.EncodeRoutingRejections),
	}
	// The MetricsOutput name of the engine's preemption count.
	if c, ok := r.Engine["cluster"].(map[string]any); ok {
		if v, ok := asFloat(c["preemptions"]); ok {
			out["preemption_count"] = v
		}
	}
	if c, ok := r.Engine["cluster"].(map[string]any); ok {
		for k, v := range c {
			if f, ok := asFloat(v); ok {
				out[k] = f
			}
		}
	}
	for k, v := range r.Metrics {
		out["metric:"+k] = v
	}
	return out
}

// instanceFigures are each engine instance's own figures (its whole-run MetricsOutput), keyed
// by instance id; nil for a single-instance run, which reports only the cluster.
func instanceFigures(t *testing.T, stdout string) map[string]map[string]any {
	t.Helper()
	v, ok := resultValue(resultDoc(t, stdout), "engine.instances")
	if !ok {
		return nil
	}
	insts, ok := v.(map[string]any)
	if !ok {
		return nil
	}
	out := make(map[string]map[string]any, len(insts))
	for id, v := range insts {
		if m, ok := v.(map[string]any); ok {
			out[id] = m
		}
	}
	return out
}

// metric is a Result metric by name, failing when the run does not report it.
func metric(t *testing.T, stdout, name string) float64 {
	t.Helper()
	r := parseResult(t, stdout)
	v, ok := r.Metrics[name]
	if !ok {
		t.Fatalf("the Result reports no metric %q (it reports %v)", name, r.Metrics)
	}
	return v
}

// withoutRuntime returns stdout with the runtime block removed, the one block INV-6 exempts.
// The document is re-encoded with blis-schemas' writer, so two documents equal outside
// runtime compare byte-identical.
func withoutRuntime(t *testing.T, stdout string) string {
	t.Helper()
	r := parseResult(t, stdout)
	r.Runtime = nil
	data, err := result.Marshal(r, result.FormatJSON)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// asFloat reads a decoded number.
func asFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case uint64:
		return float64(n), true
	}
	return 0, false
}

// resultDoc decodes stdout as a generic document, for reading a value by its path.
func resultDoc(t *testing.T, stdout string) map[string]any {
	t.Helper()
	parseResult(t, stdout) // stdout must be one valid Result
	var doc map[string]any
	if err := json.Unmarshal([]byte(stdout), &doc); err != nil {
		t.Fatalf("stdout is not JSON: %v", err)
	}
	return doc
}

// resultValue is the value at a dotted path of the document ("requests.completed",
// "metrics.ttft_mean_ms", "engine.instances.instance_0.completed_requests"), and whether
// the document has it.
func resultValue(doc map[string]any, path string) (any, bool) {
	var cur any = doc
	for _, seg := range strings.Split(path, ".") {
		m, ok := cur.(map[string]any)
		if !ok {
			return nil, false
		}
		if cur, ok = m[seg]; !ok {
			return nil, false
		}
	}
	return cur, true
}

// resultNum is the number at path in stdout's Result, failing when it is absent.
func resultNum(t *testing.T, stdout, path string) float64 {
	t.Helper()
	v, ok := resultValue(resultDoc(t, stdout), path)
	if !ok {
		t.Fatalf("the Result has no %s:\n%s", path, lastLines(stdout, 40))
	}
	f, ok := v.(float64)
	if !ok {
		t.Fatalf("the Result's %s is %T, not a number", path, v)
	}
	return f
}

// inv13Sections are the parts of a Result that INV-13 holds equal between a run and the
// replay of its trace: everything the simulation computed. The input record, the name (which
// states the load), the artifacts, the revisions and the runtime describe how each command
// was invoked and are not compared.
var inv13Sections = []string{"metric_defs", "window", "metrics", "requests", "goodput", "breakdowns",
	"engine", "router", "kernel", "saturation", "overrides"}

// inv13View is stdout's INV-13 sections, re-encoded canonically, so a run and its replay
// compare as strings.
func inv13View(t *testing.T, stdout string) string {
	t.Helper()
	doc := resultDoc(t, stdout)
	view := map[string]any{}
	for _, k := range inv13Sections {
		if v, ok := doc[k]; ok {
			view[k] = v
		}
	}
	data, err := json.MarshalIndent(view, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// completedIs reports whether stdout is one Result that completed exactly n requests.
func completedIs(stdout string, n int) bool {
	var r result.Result
	return result.Unmarshal([]byte(stdout), &r) == nil && r.Requests.Completed == n
}

// completedAny reports whether stdout is one Result that completed at least one request.
func completedAny(stdout string) bool {
	var r result.Result
	return result.Unmarshal([]byte(stdout), &r) == nil && r.Requests.Completed > 0
}

// withoutRevisionsAndRuntime is stdout's Result with runtime and the input revisions removed:
// what two runs of the same resolved input must agree on byte for byte when only an input's
// revision differs (a catalog edited without committing is a new revision, INV-6).
func withoutRevisionsAndRuntime(t *testing.T, stdout string) string {
	t.Helper()
	r := parseResult(t, stdout)
	r.Runtime = nil
	r.Revisions = result.Revisions{BLIS: "x", BLISSchemas: "x"}
	r.Artifacts = nil
	data, err := result.Marshal(r, result.FormatJSON)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// simulated is stdout's Result reduced to what the simulation computed, for a metamorphic test
// whose two runs state one configuration two ways (a scenario field against the flag that sets
// it): their input records differ by construction, and so do the sections that describe those
// inputs -- the kernel's provenance of the deployment it was handed, and the overrides -- while
// every simulated figure must not.
func simulated(t *testing.T, stdout string) string {
	t.Helper()
	doc := resultDoc(t, stdout)
	view := map[string]any{}
	for _, k := range inv13Sections {
		if k == "kernel" || k == "overrides" {
			continue
		}
		if v, ok := doc[k]; ok {
			view[k] = v
		}
	}
	data, err := json.MarshalIndent(view, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// withoutRuntimeOrArtifactPaths is withoutRuntime with each artifact's ref cleared, for two
// runs that write their files to different temporary directories: the digests and row counts,
// what the files hold, still compare.
func withoutRuntimeOrArtifactPaths(t *testing.T, stdout string) string {
	t.Helper()
	r := parseResult(t, stdout)
	r.Runtime = nil
	for k, a := range r.Artifacts {
		a.Ref = ""
		r.Artifacts[k] = a
	}
	data, err := result.Marshal(r, result.FormatJSON)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// firstDiff is the first differing line of two documents and its neighbours, for a failure
// message that would otherwise print two whole documents.
func firstDiff(a, b string) string {
	al, bl := strings.Split(a, "\n"), strings.Split(b, "\n")
	for i := 0; i < len(al) || i < len(bl); i++ {
		var x, y string
		if i < len(al) {
			x = al[i]
		}
		if i < len(bl) {
			y = bl[i]
		}
		if x != y {
			lo := max(0, i-3)
			ctx := strings.Join(al[lo:min(i, len(al))], "\n")
			return fmt.Sprintf("line %d:\n%s\n- %s\n+ %s", i+1, ctx, x, y)
		}
	}
	return ""
}

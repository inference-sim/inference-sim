package workload

import (
	"math"
	"testing"
)

func chatbot() *Shape {
	return &Shape{Name: "chatbot",
		Prompt: Distribution{Mean: 256, StdDev: 100, Min: 2, Max: 800},
		Output: Distribution{Mean: 256, StdDev: 100, Min: 1, Max: 1024}}
}

// agentic is the long-context shape the published report corpus exercises: a
// request carrying tens of thousands of input tokens against a few hundred output.
func agentic() *Shape {
	return &Shape{Name: "agentic", PrefixTokens: 4096,
		Prompt: Distribution{Mean: 52393, Min: 8000, Max: 262144},
		Output: Distribution{Mean: 486, Min: 1, Max: 2048}}
}

func TestValidShapesPass(t *testing.T) {
	for _, s := range []*Shape{chatbot(), agentic()} {
		if p := s.Validate(); !p.OK() {
			t.Fatalf("%s rejected:\n%s", s.Name, p.Error())
		}
	}
}

func TestRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Shape)
	}{
		{"no name", func(s *Shape) { s.Name = "" }},
		{"negative prefix", func(s *Shape) { s.PrefixTokens = -1 }},
		{"prefix longer than the prompt", func(s *Shape) { s.PrefixTokens = 9999 }},
		{"zero prompt mean", func(s *Shape) { s.Prompt.Mean = 0 }},
		{"zero output mean", func(s *Shape) { s.Output.Mean = 0 }},
		{"min above max", func(s *Shape) { s.Prompt.Min = 900; s.Prompt.Max = 800 }},
		{"mean above max", func(s *Shape) { s.Prompt.Max = 10 }},
		{"mean below min", func(s *Shape) { s.Prompt.Min = 9000 }},
		{"negative stdev", func(s *Shape) { s.Prompt.StdDev = -1 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := chatbot()
			tc.mutate(s)
			if p := s.Validate(); p.OK() {
				t.Fatal("expected rejection, got none")
			}
		})
	}
}

func seed(v int64) *int64 { return &v }

// traceRef is a well-formed concrete-arm reference: a data path, integrity and provenance
// fields, and the small header metadata including a server block and one SLO class.
func traceRef() *TraceRef {
	return &TraceRef{
		Data:   "traces/agentic-run.csv",
		SHA256: strRepeat("a", 64),
		Rows:   1048576,
		Header: TraceHeader{
			Version: 3, TimeUnit: "microseconds", Mode: ModeReal, WorkloadSeed: seed(0),
			Server: &TraceServer{Type: "vllm", Model: "granite-5-230b",
				TensorParallel: 8, MaxNumSeqs: 1024, BlockSize: 16,
				GPUMemoryUtilization: 0.9, MaxModelLen: 131072},
			GoodputSLOTargets: map[string]SLODimTargets{
				"critical": {TTFTMs: 500, ITLMs: 50, E2EMs: 30000}},
		},
	}
}

func TestValidBindingsPass(t *testing.T) {
	cases := map[string]*Binding{
		"shape arm": {Shape: "chatbot"},
		"trace arm": {Trace: traceRef()},
		"minimal trace: path + required header only": {Trace: &TraceRef{
			Data:   "t.csv",
			Header: TraceHeader{Version: 1, TimeUnit: "us", Mode: ModeGenerated}}},
	}
	for name, b := range cases {
		t.Run(name, func(t *testing.T) {
			if p := b.Validate(); !p.OK() {
				t.Fatalf("valid binding rejected:\n%s", p.Error())
			}
		})
	}
}

// TestBindingIsExactlyOneArm is the sum-type law: a present binding chooses a shape or a
// trace, never both and never neither. A scenario with no traffic omits the binding
// entirely, so neither-arm is an error here rather than a silent "no traffic".
func TestBindingIsExactlyOneArm(t *testing.T) {
	cases := map[string]*Binding{
		"both arms set":   {Shape: "chatbot", Trace: traceRef()},
		"neither arm set": {},
	}
	for name, b := range cases {
		t.Run(name, func(t *testing.T) {
			if p := b.Validate(); p.OK() {
				t.Fatal("expected rejection, got none")
			}
		})
	}
}

func TestTraceRefRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*TraceRef)
	}{
		{"no data path", func(tr *TraceRef) { tr.Data = "" }},
		{"short sha256", func(tr *TraceRef) { tr.SHA256 = "abc" }},
		{"non-hex sha256", func(tr *TraceRef) { tr.SHA256 = strRepeat("g", 64) }},
		{"negative rows", func(tr *TraceRef) { tr.Rows = -1 }},
		{"token blocks without a path", func(tr *TraceRef) {
			tr.TokenBlocks = &TokenBlocks{SHA256: strRepeat("a", 64)}
		}},
		{"token blocks without a digest", func(tr *TraceRef) {
			tr.TokenBlocks = &TokenBlocks{Path: "blocks.jsonl.zst"}
		}},
		{"token blocks with an uppercase digest", func(tr *TraceRef) {
			tr.TokenBlocks = &TokenBlocks{Path: "blocks.jsonl.zst", SHA256: strRepeat("A", 64)}
		}},
		{"zero trace version", func(tr *TraceRef) { tr.Header.Version = 0 }},
		{"no time unit", func(tr *TraceRef) { tr.Header.TimeUnit = "" }},
		{"unknown time unit", func(tr *TraceRef) { tr.Header.TimeUnit = "fortnights" }},
		{"no mode", func(tr *TraceRef) { tr.Header.Mode = "" }},
		{"unknown mode", func(tr *TraceRef) { tr.Header.Mode = "simulated" }},
		{"negative tensor_parallel", func(tr *TraceRef) { tr.Header.Server.TensorParallel = -1 }},
		{"gpu util above one", func(tr *TraceRef) { tr.Header.Server.GPUMemoryUtilization = 1.5 }},
		{"empty slo class key", func(tr *TraceRef) {
			tr.Header.GoodputSLOTargets = map[string]SLODimTargets{"": {TTFTMs: 1}}
		}},
		{"slo class with no target", func(tr *TraceRef) {
			tr.Header.GoodputSLOTargets = map[string]SLODimTargets{"critical": {}}
		}},
		{"negative slo target", func(tr *TraceRef) {
			tr.Header.GoodputSLOTargets = map[string]SLODimTargets{"critical": {TTFTMs: -1}}
		}},
		{"non-finite gpu util", func(tr *TraceRef) {
			tr.Header.Server.GPUMemoryUtilization = math.NaN()
		}},
		{"non-finite slo target", func(tr *TraceRef) {
			tr.Header.GoodputSLOTargets = map[string]SLODimTargets{"critical": {TTFTMs: math.Inf(1)}}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tr := traceRef()
			tc.mutate(tr)
			if p := (&Binding{Trace: tr}).Validate(); p.OK() {
				t.Fatal("expected rejection, got none")
			}
		})
	}
}

// strRepeat avoids importing strings for one call; the sha256 fixtures need a 64-char
// string and nothing else from it.
func strRepeat(s string, n int) string {
	out := make([]byte, 0, len(s)*n)
	for i := 0; i < n; i++ {
		out = append(out, s...)
	}
	return string(out)
}

func TestTraceRefWithTokenBlocksPasses(t *testing.T) {
	tr := traceRef()
	tr.TokenBlocks = &TokenBlocks{Path: "blocks.jsonl.zst", SHA256: strRepeat("a", 64)}
	if p := (&Binding{Trace: tr}).Validate(); !p.OK() {
		t.Fatalf("a trace with a block-hash sidecar was rejected:\n%s", p.Error())
	}
}

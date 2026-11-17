package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/result"
)

// Pipeline parallelism (inference-sim#1535): a pool states a stage count; each stage is priced as its own step over
// its share of the layers, with the boundary send, and the stages may sit on different nodes.

func ppVariant(t *testing.T, edits ...string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(testScenarios, "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	for i := 0; i+1 < len(edits); i += 2 {
		if !strings.Contains(body, edits[i]) {
			t.Fatalf("fixture has no %q to edit", edits[i])
		}
		body = strings.Replace(body, edits[i], edits[i+1], 1)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "variant.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func ppRun(t *testing.T, dir string) result.Result {
	t.Helper()
	out, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml",
		"--num-requests", "16", "--rate", "4", "--seed", "1", "--num-instances", "1")
	if err != nil || !completedIs(out, 16) {
		t.Fatalf("pp run: %v\n%s", err, lastLines(stderr, 3))
	}
	var r result.Result
	if err := result.Unmarshal([]byte(out), &r); err != nil {
		t.Fatal(err)
	}
	return r
}

func TestRunCmd_KernelBackend_PipelineStagesSplitTheLayers(t *testing.T) {
	one := ppRun(t, ppVariant(t))
	// Two stages on the same node: each stage is a step over half the layers, so the inter-token interval of one
	// request falls below the single stage's, and the cluster still completes every request.
	two := ppRun(t, ppVariant(t, "      pp: 1", "      pp: 2", "  nodes: 1", "  nodes: 1"))
	a, b := one.Metrics["icl_p50_ms"].Value, two.Metrics["icl_p50_ms"].Value
	if !(b > 0 && a > 0) {
		t.Fatalf("no inter-token latency: pp1 %v pp2 %v", a, b)
	}
	if b == a {
		t.Errorf("a two-stage pool priced exactly like one stage (%v ms): the stage count is ignored", a)
	}
}

func TestRunCmd_KernelBackend_PipelineStagesAcrossNodes(t *testing.T) {
	// tp8 x pp2 over two H200 nodes: a stage per node; the boundary send crosses the fabric.
	dir := ppVariant(t, "      tp: 4", "      tp: 8", "      pp: 1", "      pp: 2",
		"  hardware: h200\n", "  hardware: h200\n  fabric: ib-400g\n", "  nodes: 1", "  nodes: 2", "    nodes: 1", "    nodes: 2")
	r := ppRun(t, dir)
	if r.Metrics["icl_p50_ms"].Value <= 0 {
		t.Errorf("no inter-token latency across nodes")
	}
}

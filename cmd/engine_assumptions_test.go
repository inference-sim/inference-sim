package cmd

import (
	"slices"
	"strings"
	"testing"
)

// kernelReport is the Result's reports.kernel.<pool> object.
func kernelReport(t *testing.T, stdout, pool string) map[string]any {
	t.Helper()
	reports, _ := resultDoc(t, stdout)["reports"].(map[string]any)
	kernels, _ := reports["kernel"].(map[string]any)
	p, ok := kernels[pool].(map[string]any)
	if !ok {
		t.Fatalf("the Result carries no reports.kernel.%s", pool)
	}
	return p
}

func stringsOf(t *testing.T, v any) []string {
	t.Helper()
	list, _ := v.([]any)
	var out []string
	for _, e := range list {
		s, ok := e.(string)
		if !ok {
			t.Fatalf("%v is not a string", e)
		}
		out = append(out, s)
	}
	return out
}

// What the engine says it assumed reaches the Result and not only stderr. A prefill group of two
// four-GPU trays' worth of ranks spans two nodes, and the engine states every handoff from the
// group's first node as an assumption of that pool (it is on stderr too). The Result lists it under
// the prefill pool, with the pool's other assumptions, sorted and once each, and not under the
// decode pool, whose group does not make it.
func TestResult_CarriesTheEnginesDisclosedAssumptions(t *testing.T) {
	const span = "a prefill group of 2 data-parallel ranks spans 2 nodes"
	dir, name := dpPoolScenarios(t, 4, 4)
	stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", name,
		"--num-instances", "4", "--prefill-instances", "2", "--decode-instances", "2",
		"--rate", "5", "--num-requests", "4", "--seed", "3")
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 5))
	}
	if !strings.Contains(stderr, span) {
		t.Fatalf("non-vacuity: the engine did not state the span assumption on stderr:\n%s", lastLines(stderr, 4))
	}
	has := func(list []string) bool {
		return slices.ContainsFunc(list, func(a string) bool { return strings.HasPrefix(a, span) })
	}
	pre := stringsOf(t, kernelReport(t, stdout, "prefill")["engine_assumptions"])
	if !has(pre) {
		t.Errorf("the prefill pool's engine_assumptions lack the span assumption: %q", pre)
	}
	if !slices.IsSorted(pre) || len(slices.Compact(slices.Clone(pre))) != len(pre) {
		t.Errorf("engine_assumptions are not sorted and distinct (two pods state the same text): %q", pre)
	}
	if dec := stringsOf(t, kernelReport(t, stdout, "decode")["engine_assumptions"]); has(dec) {
		t.Errorf("the decode pool lists the prefill group's span assumption: %q", dec)
	}
}

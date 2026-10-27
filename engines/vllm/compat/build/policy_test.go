// Original to blis-vllm.

package build

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// vendoredRepos are the kernel repos with this module's vendored scenarios (testdata/scenarios,
// copies of inference-sim's, testdata/README.md).
func vendoredRepos(t *testing.T) kernelmodel.Repos {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	repos := kernelmodel.DefaultRepos()
	repos.Scenarios = filepath.Join(filepath.Dir(filepath.Dir(wd)), "testdata", "scenarios")
	return repos
}

// pdVariant writes a copy of the vendored 3P1D GLM-5 scenario whose prefill and decode
// engines state the given extra lines, and returns the repos to open it with.
func pdVariant(t *testing.T, prefill, decode string) kernelmodel.Repos {
	t.Helper()
	repos := vendoredRepos(t)
	raw, err := os.ReadFile(filepath.Join(repos.Scenarios, "glm-5-h200-3p1d-ib.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	const marker = "      gpu_memory_utilization: 0.9\n"
	s := string(raw)
	if strings.Count(s, marker) != 2 {
		t.Fatalf("fixture has %d engine blocks to edit, want 2", strings.Count(s, marker))
	}
	i := strings.Index(s, marker) + len(marker)
	s = s[:i] + prefill + s[i:]
	j := strings.LastIndex(s, marker) + len(marker)
	s = s[:j] + decode + s[j:]
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "variant.yaml"), []byte(s), 0o644); err != nil {
		t.Fatal(err)
	}
	repos.Scenarios = dir
	return repos
}

func openRole(t *testing.T, repos kernelmodel.Repos, role deployment.Role) *kernelmodel.Model {
	t.Helper()
	m, err := kernelmodel.OpenRole("variant.yaml", repos, role)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// Each pool's scheduling_policy maps to its own queue order and preemption policy; an
// unstated one is vLLM's default fcfs, never the other pool's; anything else is refused.
func TestEnginePolicyOf_EachPoolMapsItsOwnPolicy(t *testing.T) {
	repos := pdVariant(t, "      scheduling_policy: priority\n", "")
	p, err := EnginePolicyOf("variant.yaml", openRole(t, repos, deployment.RolePrefill))
	if err != nil || p != (EnginePolicy{Scheduler: "priority-fcfs", Preemption: "priority"}) {
		t.Errorf("prefill (priority): %+v, %v", p, err)
	}
	d, err := EnginePolicyOf("variant.yaml", openRole(t, repos, deployment.RoleDecode))
	if err != nil || d != (EnginePolicy{Scheduler: "fcfs", Preemption: "fcfs"}) {
		t.Errorf("decode (unstated): %+v, %v; want vLLM's default fcfs, not the prefill pool's", d, err)
	}
	fcfs := pdVariant(t, "      scheduling_policy: fcfs\n", "")
	if p, err := EnginePolicyOf("variant.yaml", openRole(t, fcfs, deployment.RolePrefill)); err != nil ||
		p != (EnginePolicy{Scheduler: "fcfs", Preemption: "fcfs"}) {
		t.Errorf("prefill (fcfs): %+v, %v", p, err)
	}
}

// A pool's knobs are the flag's when one is given, else what the pool states, else vLLM's
// defaults (no threshold, the internal balancer).
func TestPoolEngineKnobs_FlagElsePoolElseDefault(t *testing.T) {
	repos := pdVariant(t,
		"      long_prefill_token_threshold: 512\n",
		"")
	prefill, decode := openRole(t, repos, deployment.RolePrefill), openRole(t, repos, deployment.RoleDecode)
	if th, mode, err := PoolEngineKnobs("variant.yaml", prefill, EngineKnobFlags{}); err != nil || th != 512 || mode != "internal" {
		t.Errorf("prefill, no flag: %d %q %v; want its own 512 and internal", th, mode, err)
	}
	if th, mode, err := PoolEngineKnobs("variant.yaml", decode, EngineKnobFlags{}); err != nil || th != 0 || mode != "internal" {
		t.Errorf("decode, no flag: %d %q %v; want vLLM's defaults", th, mode, err)
	}
	flags := EngineKnobFlags{DPLoadBalancing: "external", DPLoadBalancingFromFlag: true}
	if th, mode, err := PoolEngineKnobs("variant.yaml", decode, flags); err != nil || th != 0 || mode != "external" {
		t.Errorf("decode, flag: %d %q %v; want its own threshold and the flag's mode", th, mode, err)
	}
	if th, _, err := PoolEngineKnobs("variant.yaml", prefill, flags); err != nil || th != 512 {
		t.Errorf("prefill, flag: %d %v; a balancing flag must leave the pool's own threshold", th, err)
	}
}

// The scenario is the source: a flag may only restate what a pool states, and a
// disagreement names the pool and both values. A colocated scenario's one pool sets the
// run-wide value; a P/D run keeps the flags' (each pool resolves its own).
func TestRunEngineKnobs_FlagsMayOnlyRestate(t *testing.T) {
	shapeOf := func(repos kernelmodel.Repos) kernelmodel.Shape {
		sh, err := kernelmodel.ShapeOf("variant.yaml", repos)
		if err != nil {
			t.Fatal(err)
		}
		return sh
	}
	sh := shapeOf(pdVariant(t, "", "      long_prefill_token_threshold: 256\n"))
	if th, mode, err := RunEngineKnobs("variant.yaml", sh, EngineKnobFlags{DPLoadBalancing: "internal"}); err != nil || th != 0 || mode != "internal" {
		t.Errorf("a P/D run with no flag: %d %q %v; each pool resolves its own threshold, the run-wide mode stays the default", th, mode, err)
	}
	dsh := shapeOf(pdVariant(t, "", ""))
	dsh.DPLoadBalancings[1] = deployment.DPLoadBalancing("external")
	if _, mode, err := RunEngineKnobs("variant.yaml", dsh, EngineKnobFlags{DPLoadBalancing: "external", DPLoadBalancingFromFlag: true}); err != nil || mode != "external" {
		t.Errorf("an agreeing flag: %q, %v", mode, err)
	}
	_, _, err := RunEngineKnobs("variant.yaml", dsh, EngineKnobFlags{DPLoadBalancing: "hybrid", DPLoadBalancingFromFlag: true})
	if err == nil {
		t.Fatal("a flag disagreeing with the decode pool must be refused")
	}
	for _, frag := range []string{"--dp-load-balancing hybrid", "decode pool's parallel.data_parallel_load_balancing external", `"variant.yaml"`} {
		if !strings.Contains(err.Error(), frag) {
			t.Errorf("refusal %q must name %q", err, frag)
		}
	}

	csh, err := kernelmodel.ShapeOf("glm-5-h200-tp8-mtp3.yaml", vendoredRepos(t))
	if err != nil {
		t.Fatal(err)
	}
	csh.LongPrefillTokenThresholds[0] = 1024
	if th, _, err := RunEngineKnobs("glm-5-h200-tp8-mtp3.yaml", csh, EngineKnobFlags{}); err != nil || th != 1024 {
		t.Errorf("a colocated pool's stated threshold: %d, %v; want it as the run-wide value", th, err)
	}
	if th, _, err := RunEngineKnobs("glm-5-h200-tp8-mtp3.yaml", csh, EngineKnobFlags{DPLoadBalancing: "internal", DPLoadBalancingFromFlag: true}); err != nil || th != 1024 {
		t.Errorf("a colocated pool's stated threshold with a balancing flag: %d, %v; want it kept", th, err)
	}
}

// The routing flags become the command line's layer; an expert ratio needs a token ratio,
// and an invalid ratio is refused.
func TestRoutingOverridesOf_TheFlagsAreTheCLILayer(t *testing.T) {
	o, err := RoutingOverridesOf(RoutingFlags{TokenImbalance: 1.6, TokenSet: true, ExpertImbalance: 0.8, ExpertSet: true})
	if err != nil || o.CLI == nil || o.CLI.TokenImbalance != 1.6 || o.CLI.Expert() != 0.8 || o.CLI.Source != "--routing-token-imbalance" {
		t.Fatalf("both flags: %+v, %v", o.CLI, err)
	}
	if o, err := RoutingOverridesOf(RoutingFlags{}); err != nil || o.CLI != nil {
		t.Errorf("no flag: %+v, %v; want no cli layer", o.CLI, err)
	}
	if _, err := RoutingOverridesOf(RoutingFlags{ExpertImbalance: 0.8, ExpertSet: true}); err == nil ||
		!strings.Contains(err.Error(), "--routing-expert-imbalance needs --routing-token-imbalance") {
		t.Errorf("an expert ratio alone: %v", err)
	}
	if _, err := RoutingOverridesOf(RoutingFlags{TokenImbalance: 0.5, TokenSet: true}); err == nil {
		t.Error("a token imbalance below 1 must be refused")
	}
}

// A run that does not speculate has no acceptance: a stated length is refused, and the
// selector flags are reported as ignored rather than silently dropped.
func TestResolveAcceptance_WithoutSpeculation(t *testing.T) {
	in := AcceptanceInputs{Scenario: "s.yaml", Flags: AcceptanceFlags{ThinkingSet: true, FallbackSet: true}}
	a, speculating, err := ResolveAcceptance(in, 0, "")
	if err != nil || speculating || strings.Join(a.Ignored, ",") != AcceptanceThinkingFlag+","+AcceptanceFallbackFlag {
		t.Errorf("selectors without speculation: %+v %v %v; want both reported as ignored", a, speculating, err)
	}
	in.Flags = AcceptanceFlags{Length: 2.5, LengthSet: true}
	if _, _, err := ResolveAcceptance(in, 0, ""); err == nil || !strings.Contains(err.Error(), "drafts no tokens") {
		t.Errorf("a length without speculation: %v", err)
	}
}

// With speculation, the command line is the highest layer: a stated rate is the simulated
// rate itself, and a length resolves through the resolver; with no layer stating one, the
// refusal says how to state it.
func TestResolveAcceptance_TheCommandLineIsTheHighestLayer(t *testing.T) {
	sh, err := kernelmodel.ShapeOf("glm-5-h200-tp8-mtp3.yaml", vendoredRepos(t))
	if err != nil {
		t.Fatal(err)
	}
	in := AcceptanceInputs{Scenario: "glm-5-h200-tp8-mtp3.yaml", Shape: sh, Registry: t.TempDir(), Hardware: "H200"}
	in.Flags = AcceptanceFlags{Rate: 0.5, RateSet: true}
	a, speculating, err := ResolveAcceptance(in, 3, "mtp")
	if err != nil || !speculating || a.CLIRate == nil || *a.CLIRate != 0.5 {
		t.Fatalf("a stated rate: %+v %v %v", a, speculating, err)
	}
	in.Flags = AcceptanceFlags{Length: 2.5, LengthSet: true}
	a, _, err = ResolveAcceptance(in, 3, "mtp")
	if err != nil || a.CLIRate != nil || a.Res.AcceptanceLength != 2.5 {
		t.Errorf("a stated length: %+v %v; want AL 2.5 and no rate override", a, err)
	}
	in.Flags = AcceptanceFlags{}
	if _, _, err := ResolveAcceptance(in, 3, "mtp"); err == nil || !strings.Contains(err.Error(), "--"+AcceptanceLengthFlag) {
		t.Errorf("no layer states it (empty registry): %v; want a refusal naming the flags", err)
	}
}

// ShapeOf reads each pool's long_prefill_token_threshold from the block blis-vllm decodes, per pool.
func TestShapeOfReadsEachPoolsLongPrefillThreshold(t *testing.T) {
	sh, err := kernelmodel.ShapeOf("variant.yaml",
		pdVariant(t, "      long_prefill_token_threshold: 512\n", "      long_prefill_token_threshold: 256\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(sh.LongPrefillTokenThresholds) != 2 || sh.LongPrefillTokenThresholds[0] != 512 ||
		sh.LongPrefillTokenThresholds[1] != 256 {
		t.Errorf("ShapeOf's thresholds are %v, want each pool's own [512 256]", sh.LongPrefillTokenThresholds)
	}
}

package routertable

import "testing"

// InferenceX runs its AgentX definitions behind vllm-router 0.1.14 with the consistent_hash
// policy. The table knows that build and that no llm-d policy is close enough: the outcome is
// no_policy, so the point stays router_unsupported with the reason on the entry, and it is not
// the silence of no_entry. Any other build of the router, any other policy and a recipe that names
// no version are not covered by what was read at 0.1.14.
func TestVllmRouterConsistentHashAtItsRecipeBuildHasNoCloseLlmdPolicy(t *testing.T) {
	entries := committed(t)
	for _, c := range []struct {
		name, mode, build string
		want              Outcome
	}{
		{"the build the 10 definitions install", "consistent_hash", "0.1.14", OutcomeNoPolicy},
		{"the dated nightly of the one disaggregated AMD run", "consistent_hash", "nightly-20260913-83944c4", OutcomeRevisionNotCovered},
		{"the next release", "consistent_hash", "0.1.15", OutcomeRevisionNotCovered},
		{"no version stated", "consistent_hash", "", OutcomeRevisionNotCovered},
		{"a policy no definition uses", "cache_aware", "0.1.14", OutcomeNoEntry},
	} {
		e, got := ResolveBuild(entries, "vllm-router", c.mode, nil, c.build)
		if got != c.want || got.Scored() {
			t.Errorf("%s: %s, want %s and never scored", c.name, got, c.want)
		}
		if c.want == OutcomeNoPolicy && (e == nil || e.Policy != PolicyNone || e.Reason == "") {
			t.Errorf("%s: the entry must say why no policy is close enough, got %+v", c.name, e)
		}
	}
	// The same build by commit: the tag's commit resolves, an adjacent release's does not.
	if _, got := Resolve(entries, "vllm-router", "consistent_hash", nil, "0a6f74ec88c87b5477dac8e24466ab760b9f1f35"); got != OutcomeNoPolicy {
		t.Errorf("the commit of tag v0.1.14: %s, want %s", got, OutcomeNoPolicy)
	}
	if _, got := Resolve(entries, "vllm-router", "consistent_hash", nil, "1fbcde7443d75b36befb61bc081f64c2a1f13a4b"); got != OutcomeRevisionNotCovered {
		t.Errorf("the commit of tag v0.1.15: %s, want %s", got, OutcomeRevisionNotCovered)
	}
}

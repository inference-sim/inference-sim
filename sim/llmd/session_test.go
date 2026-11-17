package llmd

import (
	"fmt"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// sessionConfig scores the endpoint a session is pinned to above every other, over a load tie-breaker.
func sessionConfig(sessionPlugin string) string {
	return `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
` + sessionPlugin + `- type: queue-scorer
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: session-affinity-scorer
    weight: 100
  - pluginRef: queue-scorer
  - pluginRef: max-score-picker
`
}

const sessionIDScorer = "- type: session-affinity-scorer\n  parameters:\n    strategy: session_id\n"

// sessionRequests is sessions x turns requests: turn k of session s arrives at (k*sessions+s)*gapUs, so
// the sessions interleave, and each turn carries its session's identifier when tagged.
func sessionRequests(sessions, turns int, gapUs int64, tagged bool) []*sim.Request {
	reqs := testRequests(sessions*turns, 3, gapUs, 0)
	for i, r := range reqs {
		r.ID = fmt.Sprintf("s%d_t%d", i%sessions, i/sessions)
		if tagged {
			r.SessionID = fmt.Sprintf("session_%d", i%sessions)
		}
	}
	return reqs
}

// endpointsBySession is the set of instances that served each session's turns.
func endpointsBySession(reqs []*sim.Request) map[string]map[string]bool {
	out := map[string]map[string]bool{}
	for _, r := range reqs {
		s := r.ID[:strings.Index(r.ID, "_")]
		if out[s] == nil {
			out[s] = map[string]bool{}
		}
		out[s][r.AssignedInstance] = true
	}
	return out
}

func runSessions(t *testing.T, reqs []*sim.Request, plugin string, seed int64) {
	t.Helper()
	cs := newCluster(testDeployment(4), cluster.NewSliceRequestSource(reqs), nil)
	cs.SetGateway(New(Options{Model: "test-model", Seed: &seed, EPPFlags: append(DefaultEPPFlags(), "--allow-experimental-plugins", "--config-text="+sessionConfig(plugin))}))
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
}

// TestSessionAffinity_TurnsStayOnTheirEndpoint: with the session_id strategy, every turn of a session is
// served by the endpoint its first turn was routed to, though the load scorer alone would move it; the same
// workload without session identifiers is spread (so the check is not vacuous).
func TestSessionAffinity_TurnsStayOnTheirEndpoint(t *testing.T) {
	const sessions, turns = 8, 6
	for seed := int64(1); seed <= 3; seed++ {
		tagged := sessionRequests(sessions, turns, 20_000, true)
		runSessions(t, tagged, sessionIDScorer, seed)
		for s, set := range endpointsBySession(tagged) {
			if len(set) != 1 {
				t.Errorf("seed %d: session %s was served by %d endpoints, want 1", seed, s, len(set))
			}
		}
		plain := sessionRequests(sessions, turns, 20_000, false)
		runSessions(t, plain, sessionIDScorer, seed)
		moved := 0
		for _, set := range endpointsBySession(plain) {
			if len(set) > 1 {
				moved++
			}
		}
		if moved == 0 {
			t.Errorf("seed %d: no session moved without identifiers; the stickiness check proves nothing", seed)
		}
	}
}

// TestSessionAffinity_BindingsExpire: a binding unused for longer than evictionTtlSeconds is dropped, so
// a session that returns after that is routed afresh; with a TTL longer than the gaps it is not.
func TestSessionAffinity_BindingsExpire(t *testing.T) {
	const sessions, turns = 8, 6
	plugin := func(ttl string) string {
		return sessionIDScorer + "    sessionIdConfig:\n      evictionTtlSeconds: " + ttl + "\n      evictionSweepSeconds: 0.1\n"
	}
	moved := func(ttl string) int {
		n := 0
		for seed := int64(1); seed <= 3; seed++ {
			reqs := sessionRequests(sessions, turns, 500_000, true) // 0.5 s between requests: a session returns after 4 s
			runSessions(t, reqs, plugin(ttl), seed)
			for _, set := range endpointsBySession(reqs) {
				if len(set) > 1 {
					n++
				}
			}
		}
		return n
	}
	if n := moved("60"); n != 0 {
		t.Errorf("with a 60 s TTL, %d sessions moved; a binding used every 4 s must hold", n)
	}
	if n := moved("1"); n == 0 {
		t.Error("with a 1 s TTL no session moved; bindings unused for 4 s must expire")
	}
}

// TestSessionAffinity_IsDeterministic: two runs of one seed route every turn alike (INV-6).
func TestSessionAffinity_IsDeterministic(t *testing.T) {
	a := sessionRequests(8, 5, 20_000, true)
	b := sessionRequests(8, 5, 20_000, true)
	runSessions(t, a, sessionIDScorer, 7)
	runSessions(t, b, sessionIDScorer, 7)
	for i := range a {
		if a[i].AssignedInstance != b[i].AssignedInstance {
			t.Fatalf("%s: run one %s, run two %s", a[i].ID, a[i].AssignedInstance, b[i].AssignedInstance)
		}
	}
}

// TestSessionAffinity_RefusesIdentitiesTheClientDoesNotSend: the default encoded_endpoint_header strategy
// needs the client to echo a token, a cookie and another header are not sent; each is refused at startup
// naming what is read, rather than running with a plugin that never sees a session.
func TestSessionAffinity_RefusesIdentitiesTheClientDoesNotSend(t *testing.T) {
	for name, tc := range map[string]struct{ plugin, want string }{
		"default strategy": {"- type: session-affinity-scorer\n", "encoded_endpoint_header"},
		"other header":     {sessionIDScorer + "    sessionIdConfig:\n      sources:\n      - header: x-tenant\n", `header "x-tenant"`},
		"cookie":           {"- type: session-id-producer\n  parameters:\n    cookieName: sid\n- type: session-affinity-scorer\n  parameters:\n    strategy: session_id\n", `cookie "sid"`},
	} {
		cs := newCluster(testDeployment(2), cluster.NewSliceRequestSource(sessionRequests(2, 2, 1000, true)), nil)
		cs.SetGateway(New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--allow-experimental-plugins", "--config-text="+sessionConfig(tc.plugin))}))
		if err := cs.Run(); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: Run() = %v, want a refusal naming %s", name, err, tc.want)
		}
	}
}

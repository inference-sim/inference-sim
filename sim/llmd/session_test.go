package llmd

import (
	"fmt"
	"math/rand"
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

// TestSessionAffinity_EncodedEndpointHeaderSticks: under the scorer's default strategy the router writes
// base64(pod) into x-session-token on each response (util/sessionaffinity WriteResponseHeader) and the
// client sends it back on the session's next request, so every turn of a session after the first goes to
// the endpoint its previous turn was served by. A request of no session echoes nothing and is spread.
func TestSessionAffinity_EncodedEndpointHeaderSticks(t *testing.T) {
	const encoded = "- type: session-affinity-scorer\n"
	for seed := int64(1); seed <= 3; seed++ {
		tagged := sessionRequests(8, 6, 20_000, true)
		runSessions(t, tagged, encoded, seed)
		for s, set := range endpointsBySession(tagged) {
			if len(set) != 1 {
				t.Errorf("seed %d: session %s was served by %d endpoints, want 1", seed, s, len(set))
			}
		}
		plain := sessionRequests(8, 6, 20_000, false)
		runSessions(t, plain, encoded, seed)
		moved := 0
		for _, set := range endpointsBySession(plain) {
			if len(set) > 1 {
				moved++
			}
		}
		if moved == 0 {
			t.Errorf("seed %d: no session moved without identifiers; the check proves nothing", seed)
		}
	}
}

// TestSessionAffinity_StaleTokenFallsBack: a token naming an endpoint that is gone scores no candidate
// (DecodePodName names a pod no candidate has), so the session is routed by the other scorers, and the
// response then carries the new endpoint's token, which the client echoes from then on.
func TestSessionAffinity_StaleTokenFallsBack(t *testing.T) {
	rebindsOnceAfterRemoval(t, "- type: session-affinity-scorer\n")
}

// TestSessionAffinity_CookieAndNamedHeaderAreSent: the session-id producer reading a cookie, and a
// session_id source naming another header, both receive the session's identifier from the client; the
// sessions stick.
func TestSessionAffinity_CookieAndNamedHeaderAreSent(t *testing.T) {
	for name, plugin := range map[string]string{
		"cookie": "- type: session-id-producer\n  parameters:\n    cookieName: sid\n" +
			"- type: session-affinity-scorer\n  parameters:\n    strategy: session_id\n    sessionIdConfig:\n      sources:\n      - attribute: SessionIDDataKey\n        producer: session-id-producer\n",
		"named header": sessionIDScorer + "    sessionIdConfig:\n      sources:\n      - header: x-tenant-session\n",
	} {
		tagged := sessionRequests(8, 6, 20_000, true)
		runSessions(t, tagged, plugin, 2)
		for s, set := range endpointsBySession(tagged) {
			if len(set) != 1 {
				t.Errorf("%s: session %s was served by %d endpoints, want 1", name, s, len(set))
			}
		}
	}
}

// TestSessionAffinity_RebindsAfterItsEndpointIsRemoved: when the endpoint a session is pinned to
// terminates, the session's later turns go to a surviving endpoint and stay there; no turn is sent to
// the removed one.
func TestSessionAffinity_RebindsAfterItsEndpointIsRemoved(t *testing.T) {
	rebindsOnceAfterRemoval(t, sessionIDScorer)
}

func rebindsOnceAfterRemoval(t *testing.T, plugin string) {
	t.Helper()
	reqs := sessionRequests(4, 8, 100_000, true)
	cs := newCluster(testDeployment(4), cluster.NewSliceRequestSource(reqs), nil)
	seed := int64(5)
	gw := New(Options{Model: "test-model", Seed: &seed, EPPFlags: append(DefaultEPPFlags(), "--allow-experimental-plugins", "--config-text="+sessionConfig(plugin))})
	var victim string
	removedAt := 0
	gw.observe = func(kind, detail string) {
		if kind == "request" && detail == "s1_t3" && victim == "" { // session 0 has been served three turns by now
			victim = reqs[0].AssignedInstance
			for _, inst := range cs.Instances() {
				if string(inst.ID()) == victim {
					inst.TransitionTo(sim.InstanceStateTerminated)
				}
			}
			removedAt = 3
		}
	}
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if victim == "" || reqs[0].AssignedInstance == "" {
		t.Fatalf("the endpoint was never removed (victim %q)", victim)
	}
	after := map[string]bool{}
	for _, r := range reqs {
		if !strings.HasPrefix(r.ID, "s0_") || r.AssignedInstance == "" {
			continue
		}
		var turn int
		if _, err := fmt.Sscanf(r.ID, "s0_t%d", &turn); err != nil {
			t.Fatalf("request %q: its ID does not name a turn: %v", r.ID, err)
		}
		if turn > removedAt {
			after[r.AssignedInstance] = true
			if r.AssignedInstance == victim {
				t.Errorf("%s was sent to the removed endpoint %s", r.ID, victim)
			}
		}
	}
	if len(after) != 1 {
		t.Errorf("session 0 used %d endpoints after the removal, want one new binding: %v", len(after), after)
	}
}

// TestSessionAffinity_ReplayRoutesAsTheRun (INV-13): a replayed trace carries the run's lengths, sessions and
// shared prefixes but draws new placeholder token values. Under session affinity composed with the precise
// prefix scorer (which reads the prompt's content and the engines' KV events), every request goes to the
// endpoint it went to in the run.
func TestSessionAffinity_ReplayRoutesAsTheRun(t *testing.T) {
	plugins := "- type: token-producer\n  parameters:\n    modelName: test-model\n    vllm:\n      url: http://render.blis.local:8000\n" +
		"- type: precise-prefix-cache-producer\n  parameters:\n    tokenProcessorConfig:\n      blockSizeTokens: 16\n    kvEventsConfig:\n      zmqEndpoint: tcp://0.0.0.0:5557\n" +
		"- type: prefix-cache-scorer\n  parameters:\n    prefixMatchInfoProducerName: precise-prefix-cache-producer\n" +
		"- type: session-affinity-scorer\n  parameters:\n    strategy: session_id\n"
	config := "apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\nplugins:\n" + plugins +
		"- type: queue-scorer\n- type: max-score-picker\n- type: single-profile-handler\nschedulingProfiles:\n- name: default\n  plugins:\n" +
		"  - pluginRef: session-affinity-scorer\n    weight: 100\n  - pluginRef: prefix-cache-scorer\n    weight: 10\n  - pluginRef: queue-scorer\n  - pluginRef: max-score-picker\n"
	route := func(tokenSeed int64) []string {
		reqs := sessionRequests(8, 5, 30_000, true)
		rng := rand.New(rand.NewSource(tokenSeed))
		shared := sim.GenerateRandomTokenIDs(rng, 32)
		for _, r := range reqs {
			n := len(r.InputTokens)
			r.InputTokens = append(append([]sim.TokenID{}, shared...), sim.GenerateRandomTokenIDs(rng, n-len(shared))...)
		}
		cs := newCluster(testDeployment(4), cluster.NewSliceRequestSource(reqs), nil)
		seed := int64(9)
		cs.SetGateway(New(Options{Model: "test-model", Seed: &seed, EPPFlags: append(DefaultEPPFlags(), "--allow-experimental-plugins", "--config-text="+config)}))
		if err := cs.Run(); err != nil {
			t.Fatalf("Run: %v", err)
		}
		out := make([]string, len(reqs))
		for i, r := range reqs {
			out[i] = r.AssignedInstance
		}
		return out
	}
	run, replay := route(1), route(2)
	for i := range run {
		if run[i] != replay[i] {
			t.Fatalf("request %d went to %s in the run and %s in its replay", i, run[i], replay[i])
		}
	}
}

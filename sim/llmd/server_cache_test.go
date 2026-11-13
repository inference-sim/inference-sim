package llmd

import (
	"math/rand"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// A model server's rendered text is cached by the publications it was rendered from (DEV-4).
// An engine can publish twice at one simulated instant: a scrape at a step boundary sees the
// state of the step ending at T, an eviction at T removes the request, and the empty step after
// the abort publishes the corrected state at T. The two states share a time and differ in
// sequence, so the second must not be served the first's text.
func TestPublicationKey_SeparatesTwoStatesPublishedAtOneInstant(t *testing.T) {
	first := []sim.EngineMetrics{{PublishedAt: 500, Seq: 7, Running: 1}}
	second := []sim.EngineMetrics{{PublishedAt: 500, Seq: 8, Running: 0}}
	if publicationKey(first) == publicationKey(second) {
		t.Fatal("two states published at one instant share a cache key; a scrape after the second would be served the first")
	}
	if publicationKey(first) != publicationKey([]sim.EngineMetrics{{PublishedAt: 500, Seq: 7, Running: 1}}) {
		t.Error("the same publication gets two keys, so nothing is ever served from the cache")
	}
}

// A data-parallel endpoint is keyed by every rank: a change in one rank alone changes the key.
func TestPublicationKey_ChangesWhenOneRankPublishes(t *testing.T) {
	a := []sim.EngineMetrics{{PublishedAt: 10, Seq: 1}, {PublishedAt: 10, Seq: 1}}
	b := []sim.EngineMetrics{{PublishedAt: 10, Seq: 1}, {PublishedAt: 10, Seq: 2}}
	if publicationKey(a) == publicationKey(b) {
		t.Fatal("a second rank's new publication does not change the endpoint's key")
	}
}

// The issue's reproduction, end to end through the model server: a scrape at the instant a step
// ends sees one request running; the request is evicted at that instant; the engine's empty step
// after the abort publishes the corrected state at the same instant; the next scrape must show
// the request gone. A cache keyed on the publication time alone served the first text again.
func TestMetricsText_SecondScrapeSeesTheStateAPostAbortStepPublishesAtTheSameInstant(t *testing.T) {
	reqs := testRequests(1, 1, 0, 0)
	reqs[0].OutputTokens = sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(1)), 3000) // long-running
	reqs[0].MaxOutputLen = len(reqs[0].OutputTokens)
	cs := newCluster(testDeployment(1), cluster.NewSliceRequestSource(reqs), nil)
	gw := New(Options{Model: "test-model"})
	var first, second string
	var evictedAt int64
	var probe func()
	scrape := func() string {
		return string(gw.rt.server.metricsText(gw.host.Instances()[0]))
	}
	running := func(text string) bool {
		return strings.Contains(text, "vllm:num_requests_running{model_name=\"test-model\"} 1")
	}
	polls := 0
	probe = func() {
		now, inst := gw.host.Now(), gw.host.Instances()[0]
		if polls++; polls > 200000 {
			return
		}
		// Wait for an instant at which a step has just ended with the request running.
		if em := inst.EngineMetricsAt(now); em.PublishedAt != now || em.Running != 1 {
			gw.host.Schedule(now+10, cluster.PriorityGatewayBackground, true, probe)
			return
		}
		first = scrape()
		for _, f := range gw.flights {
			gw.host.Evict(f.req)
		}
		evictedAt = now
		gw.host.Schedule(now+1, cluster.PriorityGatewayBackground, false, func() { second = scrape() })
	}
	armed := false
	gw.observe = func(kind, _ string) {
		if kind == "request" && !armed {
			armed = true
			gw.host.Schedule(gw.host.Now()+1, cluster.PriorityGatewayBackground, true, probe)
		}
	}
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if evictedAt == 0 || first == "" || second == "" {
		t.Fatalf("the scenario did not reach a step end with the request running (evictedAt %d)", evictedAt)
	}
	if !running(first) {
		t.Fatalf("the first scrape does not show the request running:\n%s", first)
	}
	if running(second) || !strings.Contains(second, "vllm:num_requests_running{model_name=\"test-model\"} 0") {
		t.Errorf("the scrape after the eviction still shows the request running: the rendered text was served from the cache for a state published at the same instant\n%s", second)
	}
	if first == second {
		t.Error("the two scrapes returned identical text")
	}
}

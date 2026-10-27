package llmd

import (
	"fmt"
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

type observed struct {
	at           int64
	kind, detail string
}

func observeRun(t *testing.T, n int, opts Options, reqGapUs int64, numReqs int) []observed {
	t.Helper()
	reqs := testRequests(numReqs, 5, reqGapUs, 0)
	cs := newCluster(testDeployment(n), cluster.NewSliceRequestSource(reqs), nil)
	opts.Model = "test-model"
	gw := New(opts)
	var log []observed
	gw.observe = func(kind, detail string) { log = append(log, observed{cs.Clock(), kind, detail}) }
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	return log
}

// TestWarmScrape_PrecedesTimeZeroRequests: every endpoint present at startup has been scraped
// before the router receives a request arriving at time zero.
func TestWarmScrape_PrecedesTimeZeroRequests(t *testing.T) {
	log := observeRun(t, 3, Options{}, 0, 5) // all five requests arrive at t=0
	scraped := map[string]bool{}
	for _, e := range log {
		switch e.kind {
		case "scrape":
			scraped[e.detail] = true
		case "request":
			if len(scraped) != 3 {
				t.Fatalf("request %s reached the router at t=%d after %d of 3 endpoints were scraped", e.detail, e.at, len(scraped))
			}
			return
		}
	}
	t.Fatal("no request reached the router")
}

// TestScrapeJitter_OffsetsPeriodicScrapes: with a scrape jitter, endpoints present at startup are
// scraped at seeded offsets within the jitter bound (still sampled once at startup); without it,
// all are scraped in phase.
func TestScrapeJitter_OffsetsPeriodicScrapes(t *testing.T) {
	firstPeriodic := func(jitter time.Duration) map[string]int64 {
		out := map[string]int64{}
		for _, e := range observeRun(t, 4, Options{ScrapeJitter: jitter}, 100000, 3) {
			if e.kind == "scrape" && e.at > 0 {
				if _, ok := out[e.detail]; !ok {
					out[e.detail] = e.at
				}
			}
		}
		return out
	}
	const interval = 50000 // the router's default --refresh-metrics-interval (50ms)
	inPhase := firstPeriodic(0)
	for ep, at := range inPhase {
		if at != interval {
			t.Errorf("no jitter: %s first periodic scrape at %d, want %d", ep, at, interval)
		}
	}
	jittered := firstPeriodic(40 * time.Millisecond)
	distinct := map[int64]bool{}
	for ep, at := range jittered {
		if at < interval || at >= interval+40000 {
			t.Errorf("jitter 40ms: %s first periodic scrape at %d, want in [%d, %d)", ep, at, interval, interval+40000)
		}
		distinct[at] = true
	}
	if len(jittered) != 4 || len(distinct) < 2 {
		t.Errorf("jitter 40ms: first periodic scrapes %v are not offset", fmt.Sprint(jittered))
	}
}

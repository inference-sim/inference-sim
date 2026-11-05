package embedtest_test

import (
	"context"
	"encoding/json"
	"errors"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/conformance/embedtest"
	"github.com/inference-sim/blis-substrate/contracts/embed"
)

func config(c stubConfig) embed.Config {
	b, _ := json.Marshal(c)
	return embed.Config{Config: b}
}

// suite is the stub router's Suite, as a router's own test would write one.
func suite(m mutation) embedtest.Suite {
	evict := config(stubConfig{FlowControl: true, Capacity: 4, Evict: true})
	unavailable := config(stubConfig{Unavailable: true})
	return embedtest.Suite{
		New:             newStub(m),
		Seed:            7,
		Route:           config(stubConfig{}),
		Hold:            config(stubConfig{FlowControl: true, TTL: "5s", Capacity: 1}),
		KV:              config(stubConfig{KV: true}),
		Evict:           &evict,
		Unavailable:     &unavailable,
		Refuse:          config(stubConfig{Plugins: []string{"scorer", "unhostable"}}),
		RefuseType:      "unhostable",
		Unaudited:       "experimental",
		UnauditedConfig: config(stubConfig{Plugins: []string{"scorer", "experimental"}, Unaudited: []string{"experimental"}}),
	}
}

// TestStubPasses: a router that keeps the contract passes.
func TestStubPasses(t *testing.T) {
	embedtest.Run(t, suite(correct))
}

// TestMutationsFail: every mutation of the stub, each breaking one rule, is caught, by the
// part of the scenario that checks that rule. This is the check's own mutation test: a check
// that let any of these pass would not be checking its rule.
func TestMutationsFail(t *testing.T) {
	for _, c := range []struct {
		m    mutation
		want string // a substring of some problem
	}{
		{saturatedAsAdmit, "hold: request-2: want a saturated rejection, got admission"},
		{ttlAsCancelled, "want a ttl-expired rejection, got context-cancelled"},
		{noEndpointsAsUnavl, "no-endpoints: request-1: want a no-endpoints rejection, got unavailable"},
		{admitNotShed, "admission: request-1: want a admission rejection, got dispatched"},
		{badBodyDispatched, "unclassified: request-1: want a unclassified rejection"},
		{randAtConstruction, "before the Constructor returned"},
		{randInStartedWork, "before the Constructor returned"},
		{wallClockSeed, "determinism: two runs on seed 7 made different decisions"},
		{leakOnStop, "hold: request-6: decided after Stop"},
		{loopAfterStop, "an Every loop ran"},
		{headersDropped, "want every header passed in"},
		{ignoresAllow, "refuse: construction succeeded although Allow refused unhostable"},
		{ignoresClear, "the other endpoint's blocks were cleared"},
		{ignoresRemoved, "the other endpoint removed the blocks it had published"},
		{neverEvicts, "evict: request-1: not evicted"},
		{routesNotReady, "after it was made not ready"},
		{removeKeeps, "after the other was removed"},
		{doneTwice, "done called 2 times"},
		{noFlowControlFacts, "hold: FlowControl() = {Enabled:false"},
		{evictSignalNoEvict, "route: request-1: an EvictionSignal from a configuration that does not evict"},
		{scoresForStrangers, `a score for "192.0.2.1:8000", which is no endpoint`},
		{failedConstructLeak, "a failed construction left an Every loop running"},
		{unavailableAsUncl, "unavailable: request-1: want a unavailable rejection, got unclassified"},
		{checkWrapped, "check: construction failed with \"stub: refused by the embedtest host check\"; want Check's error, unwrapped"},
		{checkAfterPlugins, "check: Allow was consulted"},
		{checkIgnored, "check: Check was called 0 times; want once"},
		{checkStalePool, "check: Check saw Pool"},
		{inspectorStale, "kv: KVBlocks(10.0.0.2:8000) after its blocks were removed holds 1 blocks; want 0"},
		{addressKept, "at the address it was updated to"},
		{requestTTLIgnored, "Request.TTL 1.25s did not bound its wait"},
		{unauditedAllowed, "unaudited: construction succeeded although experimental has not been audited"},
		{unauditedNoAsk, "unaudited: Allow was never asked about experimental"},
		{parkedForever, "route: 1 goroutines started with Runtime.Go are still live after Stop"},
	} {
		t.Run(string(c.m), func(t *testing.T) {
			problems := embedtest.Problems(suite(c.m))
			for _, p := range problems {
				if strings.Contains(p.Error(), c.want) {
					return
				}
			}
			t.Errorf("no problem contains %q; got %d:", c.want, len(problems))
			for _, p := range problems {
				t.Log(p)
			}
		})
	}
}

// TestRuntimeIsDeterministic: two Runtimes on one seed give the same streams, and distinct
// stream names give distinct streams.
func TestRuntimeIsDeterministic(t *testing.T) {
	a, b := embedtest.NewRuntime(3), embedtest.NewRuntime(3)
	for range 8 {
		if x, y := a.Rand("s").Float64(), b.Rand("s").Float64(); x != y {
			t.Fatalf("stream s differs between runtimes on one seed: %v, %v", x, y)
		}
	}
	if a.Rand("t").Float64() == a.Rand("u").Float64() {
		t.Error("streams t and u gave the same first draw")
	}
}

// TestDeadlineErrIsStable: a deadline context keeps the first error it reports, whichever of
// its deadline, its cancel function and its parent ends it first.
func TestDeadlineErrIsStable(t *testing.T) {
	for _, c := range []struct {
		name string
		end  func(rt *embedtest.Runtime, parentCancel, cancel func())
		want error
	}{
		{"deadline", func(rt *embedtest.Runtime, _, _ func()) { rt.Advance(time.Second) }, context.DeadlineExceeded},
		{"cancel then deadline", func(rt *embedtest.Runtime, _, cancel func()) { cancel(); rt.Advance(time.Second) }, context.Canceled},
		{"parent then deadline", func(rt *embedtest.Runtime, parent, _ func()) { parent(); rt.Advance(time.Second) }, context.Canceled},
	} {
		t.Run(c.name, func(t *testing.T) {
			rt := embedtest.NewRuntime(1)
			parent, parentCancel := context.WithCancel(context.Background())
			ctx, cancel := rt.WithTimeout(parent, time.Second)
			defer cancel()
			c.end(rt, parentCancel, cancel)
			if err := ctx.Err(); err != c.want {
				t.Fatalf("Err() = %v, want %v", err, c.want)
			}
			rt.Advance(time.Second)
			parentCancel()
			if err := ctx.Err(); err != c.want {
				t.Errorf("Err() changed to %v after it was %v", err, c.want)
			}
		})
	}
}

// TestDeadlineRaceHasOneOutcome: a deadline and a parent cancellation racing from two
// goroutines end the context once. Whichever wins, Err reports it — DeadlineExceeded exactly
// when the deadline's cause is the context's cause, Canceled exactly when the parent's is —
// and Err never changes once it has reported an error.
func TestDeadlineRaceHasOneOutcome(t *testing.T) {
	errDeadline := errors.New("deadline cause")
	errParent := errors.New("parent cause")
	wins := map[error]int{}
	for i := range 2000 {
		rt := embedtest.NewRuntime(1)
		parent, cancelParent := context.WithCancelCause(context.Background())
		ctx, cancel := rt.WithDeadlineCause(parent, rt.Now().Add(time.Millisecond), errDeadline)
		var first atomic.Pointer[error]
		var wg sync.WaitGroup
		stopWatch := make(chan struct{})
		wg.Add(3)
		go func() { // a reader: the first error Err reports, and that it never changes
			defer wg.Done()
			for {
				if err := ctx.Err(); err != nil {
					if !first.CompareAndSwap(nil, &err) && *first.Load() != err {
						t.Errorf("iteration %d: Err changed from %v to %v", i, *first.Load(), err)
					}
				}
				select {
				case <-stopWatch:
					return
				default:
				}
			}
		}()
		go func() { defer wg.Done(); rt.Advance(time.Millisecond) }()
		go func() { defer wg.Done(); cancelParent(errParent) }()
		for ctx.Err() == nil {
			runtime.Gosched()
		}
		time.Sleep(time.Microsecond)
		close(stopWatch)
		wg.Wait()
		err, cause := ctx.Err(), context.Cause(ctx)
		switch {
		case errors.Is(cause, errDeadline) && err == context.DeadlineExceeded:
		case errors.Is(cause, errParent) && err == context.Canceled:
		default:
			t.Fatalf("iteration %d: Err() = %v but the context's cause is %v", i, err, cause)
		}
		if f := first.Load(); f != nil && *f != err {
			t.Fatalf("iteration %d: Err first reported %v, now %v", i, *f, err)
		}
		wins[err]++
		cancel()
	}
	t.Logf("outcomes: %v", wins)
}

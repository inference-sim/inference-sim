package llmd

import (
	"fmt"
	"runtime/debug"
	"slices"
	"time"
)

// watchdog bounds how long (wall clock) a coroutine may run without yielding. Router work takes no
// simulated time, so a coroutine that does not yield is blocked on something the runtime cannot
// see — an unconverted blocking call — and the simulation would otherwise hang.
const watchdog = 2 * time.Minute

// scheduler runs the router's goroutines as coroutines under a single baton. Exactly one of the
// simulation goroutine and the coroutines runs at any time, so the router's shared state is never
// touched concurrently and the order of its operations depends only on the order of simulation
// events, never on the Go scheduler (INV-6).
//
// A coroutine runs from the moment it is spawned until it finishes or parks. A parked coroutine is
// resumed by settle once its ready predicate holds; predicates only read state that changes while
// the baton is held, so settle sees a consistent view.
type scheduler struct {
	current  *coroutine   // the baton holder; nil while the simulation goroutine runs
	parked   []*coroutine // parked coroutines in spawn order
	nextID   int
	live     int
	settling bool
	// settled, when set, runs after each settle on the simulation goroutine: it observes what the
	// router's goroutines did (it must not resume them; it may schedule events).
	settled func()
}

type coroutine struct {
	id      int
	resume  chan struct{}
	yielded chan struct{}
	ready   func() bool
	done    bool
	panicV  any
	stack   []byte
}

func newScheduler() *scheduler { return &scheduler{} }

// spawn starts fn as a coroutine and runs it until it finishes or parks.
func (s *scheduler) spawn(fn func()) {
	c := &coroutine{id: s.nextID, resume: make(chan struct{}), yielded: make(chan struct{})}
	s.nextID++
	s.live++
	go func() {
		<-c.resume
		defer func() {
			if r := recover(); r != nil {
				c.panicV = r
				c.stack = debug.Stack()
			}
			c.done = true
			c.yielded <- struct{}{}
		}()
		fn()
	}()
	s.run(c)
}

// run hands the baton to c and takes it back when c finishes or parks.
func (s *scheduler) run(c *coroutine) {
	prev := s.current
	s.current = c
	c.resume <- struct{}{}
	select {
	case <-c.yielded:
	case <-time.After(watchdog): //blis:wallclock-ok diagnostic watchdog, never affects results
		panic(fmt.Sprintf("llmd: router coroutine %d ran %v without yielding: it is blocked on a wait the "+
			"hosting runtime cannot observe (an unconverted blocking call in the router fork)", c.id, watchdog))
	}
	s.current = prev
	if c.done {
		s.live--
		if c.panicV != nil {
			panic(fmt.Sprintf("llmd: router coroutine %d panicked: %v\n%s", c.id, c.panicV, c.stack))
		}
	}
}

// park blocks the current coroutine until ready holds.
func (s *scheduler) park(ready func() bool) {
	if ready() {
		return
	}
	c := s.current
	if c == nil {
		panic("llmd: the router blocked on the simulation goroutine; only router coroutines may wait")
	}
	c.ready = ready
	s.parked = append(s.parked, c)
	c.yielded <- struct{}{}
	<-c.resume
}

// settle resumes parked coroutines whose predicates hold, in spawn order, until none is ready.
// It runs only on the simulation goroutine; a call from inside a coroutine is a no-op because the
// simulation goroutine settles once the coroutine yields.
func (s *scheduler) settle() {
	if s.current != nil || s.settling {
		return
	}
	s.settling = true
	defer func() {
		s.settling = false
		if s.settled != nil {
			s.settled()
		}
	}()
	for {
		progressed := false
		for i := 0; i < len(s.parked); {
			c := s.parked[i]
			if !c.ready() {
				i++
				continue
			}
			s.parked = slices.Delete(s.parked, i, i+1)
			s.run(c)
			progressed = true
		}
		if !progressed {
			return
		}
		slices.SortFunc(s.parked, func(a, b *coroutine) int { return a.id - b.id })
	}
}

// drain resumes ready coroutines newest first until none is ready. It is used at shutdown, where
// cancelling the root context readies every loop at once and a parent loop may wait for its
// children to exit.
func (s *scheduler) drain() {
	for {
		progressed := false
		for i := len(s.parked) - 1; i >= 0; i-- {
			if i >= len(s.parked) {
				continue
			}
			c := s.parked[i]
			if !c.ready() {
				continue
			}
			s.parked = slices.Delete(s.parked, i, i+1)
			s.run(c)
			progressed = true
		}
		if !progressed {
			return
		}
	}
}

// Live returns the number of coroutines that have not finished.
func (s *scheduler) Live() int { return s.live }

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/managers/cache_controller.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package hicache

import "errors"

// Clock counts step boundaries: the only time the controller sees. The caller advances it with
// Controller.Boundary.
type Clock struct {
	boundaries int
}

// Boundaries is how many step boundaries have passed.
func (c *Clock) Boundaries() int { return c.boundaries }

// Event is a device event (torch.cuda.Event) on the virtual clock, completing as the step-driven
// runner's rule says (oracle/step_runner.py RULE).
type Event struct {
	clock      *Clock
	recordedAt int
	done       bool
}

// NewEvent is an event on clock that has not been recorded.
func NewEvent(clock *Clock) *Event { return &Event{clock: clock, done: true} }

// Record marks the event at the current boundary: it completes at the next one.
func (e *Event) Record() {
	e.recordedAt = e.clock.boundaries
	e.done = false
}

// Query is whether the event has completed.
func (e *Event) Query() bool {
	if !e.done && e.clock.boundaries > e.recordedAt {
		e.done = true
	}
	return e.done
}

// Synchronize waits for the event, which completes it.
func (e *Event) Synchronize() { e.done = true }

// LayerLoadingEvent is LayerLoadingEvent (cache_controller.py:65-80@297e54fd): one event per
// layer of a load-back, each recorded when that layer's copy is submitted; the last is the
// load's finish.
type LayerLoadingEvent struct {
	Start      *Event
	LoadEvents []*Event
}

// Finish is the last layer's event.
func (l *LayerLoadingEvent) Finish() *Event { return l.LoadEvents[len(l.LoadEvents)-1] }

// Complete records layer's event (cache_controller.py:71-73).
func (l *LayerLoadingEvent) Complete(layer int) {
	if layer < 0 || layer >= len(l.LoadEvents) {
		panic("hicache: layer index out of range")
	}
	l.LoadEvents[layer].Record()
}

// NumLayerCounters is LayerDoneCounter.num_counters (cache_controller.py:87): a producer and
// consumer pair, and one more for the overlap scheduler.
const NumLayerCounters = 3

// LayerDoneCounter is LayerDoneCounter (cache_controller.py:83-111@297e54fd): the ring of load
// events a load-back is produced into and a forward pass consumes, layer by layer (SGL-7's wait
// point).
type LayerDoneCounter struct {
	NumLayers     int
	Events        [NumLayerCounters]*LayerLoadingEvent
	ProducerIndex int
	ConsumerIndex int
	// Waits is each layer index wait_until was called with, in order: the step's wait points.
	Waits []int
}

// NewLayerDoneCounter is a counter of numLayers layers with no load produced or consumed.
func NewLayerDoneCounter(clock *Clock, numLayers int) *LayerDoneCounter {
	c := &LayerDoneCounter{NumLayers: numLayers, ProducerIndex: -1, ConsumerIndex: -1}
	for i := range c.Events {
		ev := &LayerLoadingEvent{Start: NewEvent(clock)}
		for range numLayers {
			ev.LoadEvents = append(ev.LoadEvents, NewEvent(clock))
		}
		c.Events[i] = ev
	}
	return c
}

// ErrProducerBusy is SGLang's assertion that the next producer's load has finished before it is
// reused (cache_controller.py:92-96). The producer index has already advanced when it fails, as
// in SGLang.
var ErrProducerBusy = errors.New("hicache: producer finish event should be ready before being reused")

// UpdateProducer advances to the next event of the ring and returns its index
// (cache_controller.py:90-97).
func (c *LayerDoneCounter) UpdateProducer() (int, error) {
	c.ProducerIndex = (c.ProducerIndex + 1) % NumLayerCounters
	if !c.Events[c.ProducerIndex].Finish().Query() {
		return 0, ErrProducerBusy
	}
	return c.ProducerIndex, nil
}

// SetConsumer is set_consumer (cache_controller.py:99-100).
func (c *LayerDoneCounter) SetConsumer(index int) { c.ConsumerIndex = index }

// WaitUntil is wait_until (cache_controller.py:102-105): the consumer's forward pass waits for
// layer threshold's copy. The port records the wait point; the simulator times it (SGL-7).
func (c *LayerDoneCounter) WaitUntil(threshold int) {
	if c.ConsumerIndex < 0 {
		return
	}
	c.Waits = append(c.Waits, threshold)
}

// Reset is reset (cache_controller.py:107-109).
func (c *LayerDoneCounter) Reset() {
	c.ProducerIndex = -1
	c.ConsumerIndex = -1
}

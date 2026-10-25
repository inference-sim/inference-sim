package sim

// recorder is how the engine writes the host's report record: every count, latency and series
// the engine produces goes through one of these calls, at exactly the point the engine used to
// write the field (#2021 B2: inference-sim owns its reports; the engine updates them through
// the seam, PLAN-E2 E2.3b). *Metrics implements it.
type recorder interface {
	// requestArrived registers a request the instance has received; arrivedAt is its arrival
	// in seconds.
	requestArrived(req *Request, arrivedAt float64)
	// requestDropped records a request dropped at enqueue: contextOverflow when it exceeds
	// the context window.
	requestDropped(req *Request, contextOverflow bool)
	// inputReceived counts a received request's prompt tokens.
	inputReceived(tokens int64)
	// timedOut records a request its client deadline cancelled.
	timedOut(req *Request)
	// preempted counts one preemption.
	preempted()
	// preemptions is the cumulative preemption count.
	preemptions() int64
	// scheduled records a request's admission into the running batch at now: its scheduling
	// delay and its prefix-cache lookup.
	scheduled(req *Request, now int64, lookup PrefixLookup)
	// firstToken records a request's time to first token, in ticks.
	firstToken(req *Request)
	// queueSnapshot records the wait queue and running batch sizes after a schedule.
	queueSnapshot(waiting, running int)
	// kvUsage records a step's KV blocks in use over duration ticks (the time-weighted
	// integral and the peak). With weight false it updates the peak only (a free poll).
	kvUsage(used, duration int64, weight bool)
	// completed records a finished request (see recordRequestCompletion for each value).
	completed(c completion)
	// lengthCapped records a request force-completed at the context limit.
	lengthCapped(req *Request)
	// kvAllocationFailure counts a final-token KV allocation that failed.
	kvAllocationFailure()
	// adapterLoaded and adapterEvicted count LoRA residency transitions.
	adapterLoaded(id string)
	adapterEvicted(id string)
	// stillHeld records the requests the engine still holds at the end of the run.
	stillHeld(waiting, running int)
}

// completion is one finished request's report values, computed by the engine.
type completion struct {
	req          *Request
	ttft         int64 // FirstTokenTime
	outputTokens int   // emitted output tokens (>= 0)
	e2e          int64 // end-to-end latency in ticks
	itl          float64
	steps        int   // FinishedStepIdx - ScheduledStepIdx
	clock        int64 // the clock at which the completion was processed
	overhead     int64 // the post-decode overhead e2e includes
}

func (m *Metrics) requestArrived(req *Request, arrivedAt float64) {
	m.Requests[req.ID] = NewRequestMetrics(req, arrivedAt)
}

func (m *Metrics) requestDropped(req *Request, contextOverflow bool) {
	m.DroppedUnservable++
	m.RecordUnserved(req, contextOverflow)
	delete(m.Requests, req.ID)
}

func (m *Metrics) inputReceived(tokens int64) { m.TotalInputTokens += int(tokens) }

func (m *Metrics) timedOut(req *Request) {
	m.TimedOutRequests++
	m.RecordTimeout(req)
}

func (m *Metrics) preempted()         { m.PreemptionCount++ }
func (m *Metrics) preemptions() int64 { return m.PreemptionCount }

func (m *Metrics) scheduled(req *Request, now int64, lookup PrefixLookup) {
	m.RequestSchedulingDelays[req.ID] = now - req.ArrivalTime
	m.recordPrefixLookup(req, lookup)
}

func (m *Metrics) firstToken(req *Request) { m.RequestTTFTs[req.ID] = float64(req.FirstTokenTime) }

func (m *Metrics) queueSnapshot(waiting, running int) {
	m.NumWaitQRequests = append(m.NumWaitQRequests, waiting)
	m.NumRunningBatchRequests = append(m.NumRunningBatchRequests, running)
}

func (m *Metrics) kvUsage(used, duration int64, weight bool) {
	if used > m.PeakKVBlocksUsed {
		m.PeakKVBlocksUsed = used
	}
	if weight {
		m.KVBlocksUsed += float64(used) * float64(duration)
	}
}

func (m *Metrics) completed(c completion) {
	req := c.req
	m.CompletedRequests++
	m.TTFTSum += c.ttft
	if c.outputTokens > 0 {
		m.TotalOutputTokens += c.outputTokens
	}
	m.RequestE2Es[req.ID] = float64(c.e2e)
	m.RequestITLs[req.ID] = c.itl
	m.RequestStepCounters = append(m.RequestStepCounters, c.steps)
	m.RequestCompletionTimes[req.ID] = float64(c.e2e + req.ArrivalTime)
	m.recordCompletionOrder(req.ID, c.clock, c.outputTokens, c.overhead)
	m.AllITLs = append(m.AllITLs, req.ITL...)
	m.StepOwners = append(m.StepOwners,
		StepOwner{CompletionUs: c.e2e + req.ArrivalTime, Steps: len(req.ITL), Phase: req.Phase})
}

func (m *Metrics) lengthCapped(req *Request) {
	m.LengthCappedRequests++
	// Refresh Requests: NewRequestMetrics was called at enqueue before LengthCapped was known,
	// so the per-request JSON reflects the flag.
	if rm, ok := m.Requests[req.ID]; ok {
		rm.LengthCapped = true
		m.Requests[req.ID] = rm
	}
}

func (m *Metrics) kvAllocationFailure()     { m.KVAllocationFailures++ }
func (m *Metrics) adapterLoaded(id string)  { m.AdapterLoadCounts[id]++ }
func (m *Metrics) adapterEvicted(id string) { m.AdapterEvictionCounts[id]++ }

func (m *Metrics) stillHeld(waiting, running int) {
	m.StillQueued = waiting
	m.StillRunning = running
}

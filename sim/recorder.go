package sim

// The *Metrics side of the engine's recorder seam (engine_seams.go): the host's report record
// written exactly as the engine used to write it.

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

func (m *Metrics) completed(c Completion) {
	req := c.Req
	m.CompletedRequests++
	m.TTFTSum += c.TTFT
	if c.OutputTokens > 0 {
		m.TotalOutputTokens += c.OutputTokens
	}
	m.RequestE2Es[req.ID] = float64(c.E2E)
	m.RequestITLs[req.ID] = c.ITL
	m.RequestStepCounters = append(m.RequestStepCounters, c.Steps)
	m.RequestCompletionTimes[req.ID] = float64(c.E2E + req.ArrivalTime)
	m.recordCompletionOrder(req.ID, c.Clock, c.OutputTokens, c.Overhead)
	m.AllITLs = append(m.AllITLs, req.ITL...)
	m.StepOwners = append(m.StepOwners,
		StepOwner{CompletionUs: c.E2E + req.ArrivalTime, Steps: len(req.ITL), Phase: req.Phase})
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

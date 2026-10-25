package engine

// Test hooks: engine internals that package sim's tests check through the host. They existed
// as unexported identifiers when the engine was part of package sim, and those tests read them
// directly; with the engine in its own package they need exported names. Nothing outside tests
// calls them. Each is a direct forward to the internal it names.

// StepCountForTest is the engine's step counter.
func (sim *EngineCore) StepCountForTest() int { return sim.stepCount }

// ReqNumComputedTokensForTest is the engine's computed-token map (its own, not a copy).
func (sim *EngineCore) ReqNumComputedTokensForTest() map[string]int64 {
	return sim.reqNumComputedTokens
}

// SetReqNumComputedTokensForTest replaces the engine's computed-token map.
func (sim *EngineCore) SetReqNumComputedTokensForTest(m map[string]int64) {
	sim.reqNumComputedTokens = m
}

// BatchFormationForTest is the engine's batch formation strategy.
func (sim *EngineCore) BatchFormationForTest() BatchFormation { return sim.batchFormation }

// ResidentAdaptersForTest is the engine's resident-adapter set (nil when LoRA is inert).
func (sim *EngineCore) ResidentAdaptersForTest() ResidentAdapterSet { return sim.residentAdapters }

// LatencyModelForTest is the engine's latency model.
func (sim *EngineCore) LatencyModelForTest() LatencyModel { return sim.latencyModel }

// ServerSeriesForTest is the engine's server series (nil when off).
func (sim *EngineCore) ServerSeriesForTest() *serverSeries { return sim.serverSeries }

// ExecuteBatchStepForTest, ProcessCompletionsForTest, RecordRequestCompletionForTest,
// PeekDecodeTokensForTest and CommitDecodeTokensForTest run one engine step phase.
func (sim *EngineCore) ExecuteBatchStepForTest(now int64) int64 { return sim.executeBatchStep(now) }

func (sim *EngineCore) ProcessCompletionsForTest(now, advance int64) []*Request {
	return sim.processCompletions(now, advance)
}

func (sim *EngineCore) RecordRequestCompletionForTest(req *Request) { sim.recordRequestCompletion(req) }

func (sim *EngineCore) PeekDecodeTokensForTest(req *Request) int64 { return sim.peekDecodeTokens(req) }

func (sim *EngineCore) CommitDecodeTokensForTest(req *Request) { sim.commitDecodeTokens(req) }

// The request's private engine state.
func (req *Request) ScheduledTokensForTest() int64         { return req.scheduledTokensOf() }
func (req *Request) PreemptionCountForTest() int           { return req.preemptionCount() }
func (req *Request) CompletionProgressIndexForTest() int64 { return req.completionProgressIndex() }

// SetPreemptionsForTest sets how often the request was preempted.
func (req *Request) SetPreemptionsForTest(n int) { req.state().preemptions = n }

// The batch formation's preemption path.
func NewVLLMBatchFormationForTest(policy PreemptionPolicy) *VLLMBatchFormation {
	return &VLLMBatchFormation{preemptionPolicy: policy}
}

func (v *VLLMBatchFormation) SelectPriorityVictimForTest(requests []*Request) int {
	return v.selectPriorityVictim(requests)
}

// PreemptForTokensForTest runs one preemption with a fresh token budget of tokens.
func (v *VLLMBatchFormation) PreemptForTokensForTest(req *Request, numNewTokens int64, result *BatchResult,
	ctx BatchContext, tokens int64, reqIndex int) (bool, int) {
	return v.preemptForTokens(req, numNewTokens, result, ctx, &stepBudget{tokens: tokens, input: tokens}, reqIndex)
}

// The server series.
func NewServerSeriesForTest(intervalUs int64) *serverSeries { return newServerSeries(intervalUs) }

func KVCacheUsageForTest(kv KVStore) float64 { return kvCacheUsage(kv) }

func (s *serverSeries) RecordGaugesForTest(t int64, running, waiting int, usage float64) {
	s.recordGauges(t, running, waiting, usage)
}

func (s *serverSeries) RecordAdmissionForTest(t int64, req *Request, l PrefixLookup) {
	s.recordAdmission(t, req, l)
}

func (s *serverSeries) RecordFirstTokenForTest(t int64, req *Request) { s.recordFirstToken(t, req) }

func (s *serverSeries) OutputForTest(id string, end int64, bins int) ServerSeriesEngine {
	return s.output(id, end, bins)
}

// NewKVCacheStateFuncForTest and NewKVStoreFromConfigForTest read the registered KV
// constructors; SetNewKVCacheStateFunc and SetNewKVStoreFromConfig set them.
func NewKVCacheStateFuncForTest() func(totalBlocks, blockSizeTokens int64) KVStore {
	return NewKVCacheStateFunc
}

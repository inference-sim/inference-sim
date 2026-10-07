package sim

// llm-d's lora-affinity scorer, ported faithfully.
//
// Source of truth: llm-d-router at d4b8afd3,
// pkg/epp/framework/plugins/scheduling/scorer/loraaffinity/lora_affinity.go
// (LoraAffinityScorer.Score). For request target T on an endpoint with metrics
// ActiveModels (adapters with ≥1 running or queued request), WaitingModels and
// MaxActiveModels (vLLM max_lora), llm-d scores:
//
//	1.0  T ∈ ActiveModels
//	0.8  |ActiveModels ∪ WaitingModels| < MaxActiveModels
//	0.6  T ∈ WaitingModels
//	0.0  otherwise
//
// evaluated top-down (first match wins). Scores are raw tiers — llm-d applies no
// normalization, so neither does this port.
//
// Mapping to BLIS. vLLM's vllm:lora_requests_info publishes running and waiting
// adapter labels; llm-d reads them into ActiveModels and WaitingModels, and notes
// that "in current vLLM WaitingModels has the same keys". BLIS's router-observable
// RoutingSnapshot.ActiveAdapters is that same running ∪ queued set, so the scorer
// passes it as BOTH maps. Consequently the 0.6 tier is unreachable through the
// snapshot — exactly as llm-d's own comment says it is against current vLLM — but
// the tier logic is kept intact in llmdLoRAAffinityTier so it is exercised directly
// by tests. MaxActiveModels ↦ RoutingSnapshot.MaxLoras. Map VALUES are ignored
// (llm-d stores 0 for every key; only membership matters).
//
// Base-model requests. llm-d keys on request.TargetModel, which for a base-model
// request is the base model's name; vLLM's adapter labels list adapter names only,
// so a base-model request is never "active" and scores 0.8 while the endpoint has
// free adapter slots and 0.0 once it is full. In BLIS a base-model request has
// Adapter == "" and ActiveAdapters never contains "" (Simulator.ActiveAdapterCounts
// skips base-model requests), so the identical lookup-miss behaviour falls out of
// the same code path. Note this means — faithfully to llm-d — the scorer is NOT
// neutral for base-model traffic: it steers it away from instances whose adapter
// slots are full of active adapters.
//
// LoRA off: MaxLoras is 0 and ActiveAdapters nil on every instance, so every
// instance scores 0.0 (0 < 0 is false) — a uniform score that cannot change an
// argmax.
//
// Knowledge boundary: reads ONLY req.Adapter, snapshot.ActiveAdapters and
// snapshot.MaxLoras. It must never read ResidentAdapters (simulator ground truth a
// real router cannot see) — guarded by TestLLMDLoRAAffinity_IgnoresResidentAdapters.
//
// Signal freshness (R17, INV-7):
//
//	Reads: ActiveAdapters, MaxLoras (Periodic when --snapshot-refresh-interval>0,
//	else Immediate).
func scoreLLMDLoRAAffinity(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	target := ""
	if req != nil {
		target = req.Adapter
	}
	for _, snap := range snapshots {
		scores[snap.ID] = llmdLoRAAffinityTier(target, snap.ActiveAdapters, snap.ActiveAdapters, snap.MaxLoras)
	}
	return scores
}

// llmdLoRAAffinityTier is the per-endpoint body of llm-d's LoraAffinityScorer.Score,
// line for line: membership of target in active/waiting, the de-duplicated union
// count, then the four-way switch in llm-d's order.
func llmdLoRAAffinityTier(target string, active, waiting map[string]int, maxActive int) float64 {
	_, isActive := active[target]
	_, isWaiting := waiting[target]

	// ActiveModels and WaitingModels share the same source in current vLLM, so
	// take the union to count each adapter once (llm-d comment).
	unionCount := len(active)
	for k := range waiting {
		if _, ok := active[k]; !ok {
			unionCount++
		}
	}

	switch {
	case isActive:
		return 1.0
	case unionCount < maxActive:
		return 0.8
	// Unreachable against current vLLM (waiting implies active), but reachable
	// with the simulator and future backends (llm-d comment).
	case isWaiting:
		return 0.6
	default:
		return 0.0
	}
}

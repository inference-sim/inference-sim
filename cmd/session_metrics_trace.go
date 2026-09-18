package cmd

import (
	"sort"

	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// computeSessionMetricsFromTrace computes session-level metrics from TraceV2 records
// produced by blis observe. Returns nil when no records carry a non-empty SessionID.
// TTFT = FirstChunkTimeUs − SendTimeUs (wall-clock µs, converted to ms).
// SessionDuration = max(LastChunkTimeUs) − SendTimeUs[round=0] per session (ms).
func computeSessionMetricsFromTrace(records []workload.TraceRecord) *cluster.SessionMetrics {
	hasSession := false
	for _, r := range records {
		if r.SessionID != "" {
			hasSession = true
			break
		}
	}
	if !hasSession {
		return nil
	}

	// Sort by RequestID for deterministic iteration (R2, INV-6)
	sorted := make([]workload.TraceRecord, len(records))
	copy(sorted, records)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].RequestID < sorted[j].RequestID
	})

	coldTTFTs := make([]float64, 0, len(sorted))
	warmTTFTs := make([]float64, 0, len(sorted))

	type sessionData struct {
		round0SendMs float64
		hasRound0    bool
		maxCompMs    float64
	}
	sessions := make(map[string]*sessionData)

	for _, r := range sorted {
		if r.SessionID == "" {
			continue
		}

		if r.Status == "ok" && r.FirstChunkTimeUs > 0 && r.SendTimeUs > 0 {
			ttftMs := float64(r.FirstChunkTimeUs-r.SendTimeUs) / 1000.0
			if r.RoundIndex == 0 {
				coldTTFTs = append(coldTTFTs, ttftMs)
			} else {
				warmTTFTs = append(warmTTFTs, ttftMs)
			}
		}

		sd, ok := sessions[r.SessionID]
		if !ok {
			sd = &sessionData{}
			sessions[r.SessionID] = sd
		}
		if r.RoundIndex == 0 && r.SendTimeUs > 0 {
			sd.round0SendMs = float64(r.SendTimeUs) / 1000.0
			sd.hasRound0 = true
		}
		if r.LastChunkTimeUs > 0 {
			compMs := float64(r.LastChunkTimeUs) / 1000.0
			if compMs > sd.maxCompMs {
				sd.maxCompMs = compMs
			}
		}
	}

	sessionIDs := make([]string, 0, len(sessions))
	for sid := range sessions {
		sessionIDs = append(sessionIDs, sid)
	}
	sort.Strings(sessionIDs)
	var durationMs []float64
	for _, sid := range sessionIDs {
		sd := sessions[sid]
		if !sd.hasRound0 {
			continue
		}
		dur := sd.maxCompMs - sd.round0SendMs
		if dur > 0 {
			durationMs = append(durationMs, dur)
		}
	}

	return &cluster.SessionMetrics{
		SessionCount:    len(sessions),
		TTFTCold:        cluster.NewDistribution(coldTTFTs),
		TTFTWarm:        cluster.NewDistribution(warmTTFTs),
		SessionDuration: cluster.NewDistribution(durationMs),
	}
}

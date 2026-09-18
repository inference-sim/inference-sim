package workload

import "sort"

// ComputeGoodputComparison compares observed (real) and simulated SLO goodput per
// class for the matched real/sim record set used by `blis calibrate` (issue #1413, BC-9).
//
// Inputs:
//
//	realRecords — full TraceV2 record slice. Records with Status not in
//	              {ok, error, timeout} are excluded (matches INV-1 dispatched-set
//	              semantics). Only records present in matchedReqIDs contribute,
//	              so warm-up exclusion and token-mismatch filtering follow
//	              PrepareCalibrationPairs.
//	simByID     — sim-side results indexed by RequestID.
//	matchedReqIDs — IDs that appear on BOTH sides (already deduplicated).
//	itlByRequest — optional per-request ITL deltas (µs, sorted by chunk index).
//	               When nil/empty, ITL gating is skipped on the real side and
//	               the report's SkippedITL flag is set.
//	targets     — per-class SLO thresholds.
//	runtimeSec  — wall-clock duration of the real-side workload.
//
// Returns nil when targets is empty (no goodput configured). Otherwise produces
// a deterministic per-class breakdown (R2: sorted class iteration).
func ComputeGoodputComparison(
	realRecords []TraceRecord,
	simByID map[int]SimResult,
	matchedReqIDs map[int]bool,
	itlByRequest map[int][]ITLRecord,
	targets map[string]SLODimTargets,
	runtimeSec float64,
) *GoodputComparisonReport {
	if len(targets) == 0 {
		return nil
	}

	type acc struct {
		count                  int
		realGood, simGood      int
		realTTFTMet, realTTFTN int
		realITLMet, realITLN   int
		realE2EMet, realE2EN   int
		simTTFTMet, simTTFTN   int
		simITLMet, simITLN     int
		simE2EMet, simE2EN     int
	}
	accs := make(map[string]*acc, len(targets))
	for c := range targets {
		accs[c] = &acc{}
	}

	itlMissing := false

	classOf := func(rec TraceRecord) (string, bool) {
		c := rec.SLOClass
		if c == "" {
			c = "default"
		}
		if _, ok := targets[c]; !ok {
			return c, false
		}
		return c, true
	}

	itlMeanMs := func(id int) (float64, bool) {
		chunks, ok := itlByRequest[id]
		if !ok || len(chunks) < 2 {
			return 0, false
		}
		sortedChunks := make([]ITLRecord, len(chunks))
		copy(sortedChunks, chunks)
		sort.Slice(sortedChunks, func(i, j int) bool { return sortedChunks[i].ChunkIndex < sortedChunks[j].ChunkIndex })
		var sum float64
		var n int
		for i := 1; i < len(sortedChunks); i++ {
			d := sortedChunks[i].TimestampUs - sortedChunks[i-1].TimestampUs
			if d > 0 {
				sum += float64(d)
				n++
			}
		}
		if n == 0 {
			return 0, false
		}
		return (sum / float64(n)) / 1000.0, true
	}

	for _, rec := range realRecords {
		if rec.Status != "ok" && rec.Status != "error" && rec.Status != "timeout" {
			continue
		}
		if !matchedReqIDs[rec.RequestID] {
			continue
		}
		cls, gated := classOf(rec)
		if !gated {
			continue
		}
		a := accs[cls]
		a.count++

		t := targets[cls]
		sr, hasSim := simByID[rec.RequestID]

		// Real side — only "ok" records can hit goodput; failed records count
		// in denominator only.
		if rec.Status == "ok" {
			realTTFTMs := float64(rec.FirstChunkTimeUs-rec.SendTimeUs) / 1000.0
			realE2EMs := float64(rec.LastChunkTimeUs-rec.SendTimeUs) / 1000.0
			realGood := realTTFTMs > 0 && realE2EMs > 0 && realE2EMs >= realTTFTMs
			if t.TTFTMs > 0 {
				if realGood {
					a.realTTFTN++
					if realTTFTMs <= t.TTFTMs {
						a.realTTFTMet++
					} else {
						realGood = false
					}
				} else {
					realGood = false
				}
			}
			if t.E2EMs > 0 && realGood {
				a.realE2EN++
				if realE2EMs <= t.E2EMs {
					a.realE2EMet++
				} else {
					realGood = false
				}
			} else if t.E2EMs > 0 {
				realGood = false
			}
			if t.ITLMs > 0 && realGood {
				if mean, ok := itlMeanMs(rec.RequestID); ok {
					a.realITLN++
					if mean <= t.ITLMs {
						a.realITLMet++
					} else {
						realGood = false
					}
				} else {
					itlMissing = true
					realGood = false
				}
			}
			if realGood {
				a.realGood++
			}
		}

		// Sim side — only count when matched on this request (always true here).
		if hasSim {
			simTTFTMs := sr.TTFT / 1000.0
			simE2EMs := sr.E2E / 1000.0
			simGood := simTTFTMs > 0 && simE2EMs > 0 && simE2EMs >= simTTFTMs
			if t.TTFTMs > 0 {
				if simGood {
					a.simTTFTN++
					if simTTFTMs <= t.TTFTMs {
						a.simTTFTMet++
					} else {
						simGood = false
					}
				}
			}
			if t.E2EMs > 0 && simGood {
				a.simE2EN++
				if simE2EMs <= t.E2EMs {
					a.simE2EMet++
				} else {
					simGood = false
				}
			}
			if t.ITLMs > 0 && simGood {
				simITLMs := sr.ITLMeanUs / 1000.0
				if simITLMs > 0 {
					a.simITLN++
					if simITLMs <= t.ITLMs {
						a.simITLMet++
					} else {
						simGood = false
					}
				} else {
					itlMissing = true
					simGood = false
				}
			}
			if simGood {
				a.simGood++
			}
		}
	}

	classKeys := make([]string, 0, len(targets))
	for k := range targets {
		classKeys = append(classKeys, k)
	}
	sort.Strings(classKeys)

	per := make(map[string]GoodputClassComparison, len(targets))
	for _, cls := range classKeys {
		a := accs[cls]
		t := targets[cls]
		entry := GoodputClassComparison{
			Count:             a.count,
			RealSLOAttainment: ratio(a.realGood, a.count),
			SimSLOAttainment:  ratio(a.simGood, a.count),
		}
		if runtimeSec > 0 {
			entry.RealGoodputRPS = float64(a.realGood) / runtimeSec
			entry.SimGoodputRPS = float64(a.simGood) / runtimeSec
		}
		realByDim := map[string]float64{}
		simByDim := map[string]float64{}
		if t.TTFTMs > 0 {
			realByDim["ttft"] = ratioInt(a.realTTFTMet, a.realTTFTN)
			simByDim["ttft"] = ratioInt(a.simTTFTMet, a.simTTFTN)
		}
		if t.ITLMs > 0 {
			realByDim["itl"] = ratioInt(a.realITLMet, a.realITLN)
			simByDim["itl"] = ratioInt(a.simITLMet, a.simITLN)
		}
		if t.E2EMs > 0 {
			realByDim["e2e"] = ratioInt(a.realE2EMet, a.realE2EN)
			simByDim["e2e"] = ratioInt(a.simE2EMet, a.simE2EN)
		}
		if len(realByDim) > 0 {
			entry.RealAttainmentByDim = realByDim
		}
		if len(simByDim) > 0 {
			entry.SimAttainmentByDim = simByDim
		}
		per[cls] = entry
	}

	return &GoodputComparisonReport{
		Targets:    targets,
		PerClass:   per,
		SkippedITL: itlMissing,
	}
}

func ratio(num, denom int) float64 {
	if denom == 0 {
		return 0
	}
	return float64(num) / float64(denom)
}

func ratioInt(num, denom int) float64 {
	if denom == 0 {
		return 0
	}
	return float64(num) / float64(denom)
}

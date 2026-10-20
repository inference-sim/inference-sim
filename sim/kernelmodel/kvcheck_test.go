package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"pgregory.net/rapid"
)

// vLLM v0.31.0 refuses to start an engine whose KV cache cannot hold one max_model_len request
// (_check_enough_kv_cache_memory, vllm/v1/core/kv_cache_utils.py:898-935): needed <= available,
// available being the pool less the null block (:2779-2786). So a pool of exactly the request's
// blocks plus the null block starts, and one block fewer is refused with vLLM's message and its
// estimate of the longest window that fits; a pool with no room is refused as having no memory.
func TestStartupKVCheck_ExactlyEnoughStartsAndOneBlockShortIsRefused(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		perBlock := rapid.Int64Range(1, 1<<20).Draw(rt, "perBlock")
		bs := rapid.IntRange(1, 128).Draw(rt, "blockSize")
		state := rapid.Int64Range(1, 40).Draw(rt, "stateBlocks") // a request of 2+ blocks: one short still leaves usable blocks
		window := rapid.IntRange(1, 1<<16).Draw(rt, "maxModelLen")
		need := func(tokens int) int64 { return int64((tokens+bs-1)/bs) + state }
		short := rapid.Bool().Draw(rt, "oneBlockShort")
		pool := need(window) + 1
		if short {
			pool--
		}
		b := KVBudget{PerRankBlocks: pool, PerBlockBytes: perBlock, AllocatableBytes: pool * perBlock}
		err := checkEnoughKVCacheMemory(b, need, window)
		if (err != nil) != short {
			rt.Fatalf("pool %d blocks for a %d-block request (+1 null): err=%v, want refused=%v", pool, need(window), err, short)
		}
		if !short {
			return
		}
		msg := err.Error()
		if !strings.Contains(msg, "To serve at least one request with the model's max seq len") {
			rt.Fatalf("not vLLM's refusal: %s", msg)
		}
		if pool-1 >= need(1) {
			est := estimateMaxModelLen(need, pool-1, window)
			if est >= window || need(est) > pool-1 || need(est+1) <= pool-1 {
				rt.Fatalf("estimated max model len %d is not the longest that fits %d blocks", est, pool-1)
			}
			if !strings.Contains(msg, "the estimated maximum model length is ") {
				rt.Fatalf("refusal lacks the estimate: %s", msg)
			}
		}
	})
	if err := checkEnoughKVCacheMemory(KVBudget{PerRankBlocks: 1, PerBlockBytes: 1}, func(int) int64 { return 1 }, 8); err == nil ||
		!strings.Contains(err.Error(), "No available memory for the cache blocks") {
		t.Errorf("a pool of only the null block: %v", err)
	}
}

// The check applies per engine, after BLIS has sized the pool: a DEP8 rank (TP1 DP8 EP8) of
// DeepSeek-V4.1-Flash serving max_model_len 262144 starts exactly where one request -- its token
// pages and its bounded state at the admission cap -- fits the rank's pool less the null block.
// Across utilization the fixture both starts and is refused, so the check decides here, and a
// refusal is a Settings error, which the CLI turns into a start-up failure.
func TestStartupKVCheck_ADEP8RankStartsOnlyWhereOneWindowFits(t *testing.T) {
	src, err := os.ReadFile("../../testdata/scenarios/deepseek-v4.1-flash-h100-2n-dep8-dspark5.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var started, refused []string
	for _, gmu := range dep8GMUs() {
		m := openDEP8(t, src, gmu)
		b, berr := m.KVBudget()
		s, serr := m.Settings()
		if berr != nil {
			if serr == nil {
				t.Errorf("gmu %s: the budget is refused (%v) but the engine starts", gmu, berr)
			}
			refused = append(refused, gmu)
			continue
		}
		need := m.requestBlocks(m.sequenceState(b), b.PerBlockBytes)(262144)
		if fits := need <= b.PerRankBlocks-1; fits != (serr == nil) {
			t.Errorf("gmu %s: a 262144-token request needs %d blocks of the rank's %d (+1 null); started=%v, err=%v",
				gmu, need, b.PerRankBlocks, serr == nil, serr)
		}
		if serr == nil {
			if s.MaxModelLen != 262144 {
				t.Errorf("gmu %s: started serving %d, want the stated 262144", gmu, s.MaxModelLen)
			}
			started = append(started, gmu)
		} else {
			refused = append(refused, gmu)
		}
	}
	// The bounded state counts: on the vendored pins a DEP8 rank's state cap (16,425 blocks) is
	// the larger part of a 262144-token request (4,096 token blocks).
	if m := openDEP8(t, src, "0.80"); true {
		b, err := m.KVBudget()
		if err == nil {
			st := m.sequenceState(b)
			if st.Gate == nil || st.Gate(262144) <= 0 {
				t.Fatal("the DEP8 fixture has no bounded state to charge")
			}
			tokensOnly := m.requestBlocks(st, b.PerBlockBytes)(262144) - st.Gate(262144)
			if tokensOnly != (m.k.SequenceVariableBytes(262144)+b.PerBlockBytes-1)/b.PerBlockBytes {
				t.Errorf("a request's blocks less its state cap are %d, not its token pages", tokensOnly)
			}
		}
	}
	if len(started) == 0 || len(refused) == 0 {
		t.Fatalf("the sweep must cross the boundary: started %v, refused %v", started, refused)
	}
	t.Logf("started at %v, refused at %v", started, refused)
}

// The registry and catalog the KV-pool fix lands (registry c5e24a7, catalog 622175d) make the
// DEP8 rank refuse at gmu 0.85 and start at 0.90 and 0.92. Runs where BLIS_KVPOOL_PINS=1 and
// BLIS_REGISTRY / BLIS_CATALOG point at them; the vendored pins are older.
func TestStartupKVCheck_ADEP8RankOnTheKVPoolPins(t *testing.T) {
	if os.Getenv("BLIS_KVPOOL_PINS") != "1" {
		t.Skip("set BLIS_KVPOOL_PINS=1 with BLIS_REGISTRY/BLIS_CATALOG at registry c5e24a7 / catalog 622175d")
	}
	src, err := os.ReadFile("../../testdata/scenarios/deepseek-v4.1-flash-h100-2n-dep8-dspark5.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for gmu, start := range map[string]bool{"0.85": false, "0.90": true, "0.92": true} {
		_, err := openDEP8(t, src, gmu).Settings()
		if (err == nil) != start {
			t.Errorf("gmu %s: started=%v, want %v (%v)", gmu, err == nil, start, err)
		}
		if !start && (err == nil || !strings.Contains(err.Error(), "max seq len (262144)")) {
			t.Errorf("gmu %s: want vLLM's refusal for max seq len 262144, got %v", gmu, err)
		}
	}
}

func dep8GMUs() []string {
	return []string{"0.30", "0.40", "0.50", "0.55", "0.60", "0.65", "0.70", "0.75", "0.80", "0.85", "0.90", "0.92"}
}

func openDEP8(t *testing.T, src []byte, gmu string) *Model {
	t.Helper()
	body := strings.Replace(string(src), "gpu_memory_utilization: 0.85", "gpu_memory_utilization: "+gmu, 1)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "dep8.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	r := repos()
	r.Scenarios = dir
	m, err := Open("dep8.yaml", r)
	if err != nil {
		t.Fatalf("gmu %s: %v", gmu, err)
	}
	return m
}

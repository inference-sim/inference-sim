package kv

import (
	"fmt"
	"math"

	"github.com/inference-sim/blis-vllm/compat/kvkey"
	"github.com/inference-sim/blis-vllm/compat/kvtransfer"
	sim "github.com/inference-sim/inference-sim/sim/engine"
)

// secondaryTier models one secondary offload tier (an "fs" tier): the chunks it HOLDS and the
// I/O pool that moves them. It tracks keys, not payloads: like the CPU tier, content is
// addressed by key and reload is driven by the requesting request's own prefix tokens.
//
// One tier may serve several engines. vLLM's fs tier names each chunk's file from the model,
// the layout and the chunk hash (vllm/v1/kv_offload/file_mapper.py:21-144 at v0.31.0) with no
// data-parallel rank, and block hashes agree across processes by default
// (tiering/fs/manager.py:96-104), so every engine of a node that names the same root_dir
// reads and writes the same files on the same device. Such engines share one secondaryTier
// (sim.OffloadTierShares): one holdings set and one device queue. Each engine keeps its own
// thread pool in vLLM (a DualQueueThreadPool per tier manager, fs/manager.py:203-207), so a
// shared tier has the servers of every engine on it.
//
// Writes are eager parallel replicas across all tiers (BC-C7a): OffloadCache.cascade
// submits a Write job to every secondary tier, and each tier records the chunks only when
// that job completes. Lookups are ORDERED (tier 0 first). A tier's own drop never touches
// another tier -- eviction is not a spill-chain. A tier has no eviction (the fs tier deletes
// nothing): with a capacity, a write that lands on a full tier stores nothing, as a write to
// a full filesystem fails.
type secondaryTier struct {
	holds    map[kvkey.BlockKey]struct{}
	capacity int64 // chunks; 0 is unbounded

	// station serves the tier's transfers as its one tier 0; nil for a tier built bare (tests
	// that exercise holdings only).
	station *kvtransfer.TransferStation
	jobs    map[kvtransfer.JobID]tierJob
	// done holds each engine's completed jobs until that engine's SetClock applies them.
	done map[*OffloadCache][]tierJob
	// engines counts the engines on the tier: each brings its own read and write threads.
	engines int
	cfg     kvtransfer.TierConfig
	// notify holds, per engine that publishes KV events for the tier (vLLM's per-tier
	// enable_kv_events), the callback told each block that lands in (removed=false) or leaves the
	// tier. A shared tier (sim.OffloadTierShares) tells every publishing engine on it: each of
	// them can read the block from the shared files.
	notify []func(key kvkey.BlockKey, removed bool)
}

// tierJob is one in-flight transfer on a tier and the engine that submitted it. A Read
// promotes keys into that engine's CPU tier; a Write lands keys in the tier and releases the
// engine's CPU pins.
type tierJob struct {
	owner *OffloadCache
	keys  []kvkey.BlockKey
	dir   kvtransfer.Direction
}

func newSecondaryTier() *secondaryTier {
	return &secondaryTier{holds: make(map[kvkey.BlockKey]struct{})}
}

// newStationTier builds a tier served by one station tier of cfg, holding up to capacity
// chunks (0 unbounded).
func newStationTier(cfg kvtransfer.TierConfig, capacity int64) *secondaryTier {
	st, err := kvtransfer.New(kvtransfer.Config{Tiers: []kvtransfer.TierConfig{cfg}})
	if err != nil {
		panic(fmt.Sprintf("NewOffloadCache: transfer station config invalid: %v", err))
	}
	t := newSecondaryTier()
	t.capacity, t.station, t.cfg, t.engines = capacity, st, cfg, 1
	t.jobs = make(map[kvtransfer.JobID]tierJob)
	t.done = make(map[*OffloadCache][]tierJob)
	return t
}

// join adds one more engine to a shared tier: its read and write threads join the pool. The
// station is rebuilt, so engines join only at construction, before any job is submitted.
func (s *secondaryTier) join() {
	if len(s.jobs) > 0 {
		panic("secondaryTier.join: an engine joined a tier with transfers in flight")
	}
	s.engines++
	cfg := s.cfg
	cfg.NRead *= s.engines
	cfg.NWrite *= s.engines
	st, err := kvtransfer.New(kvtransfer.Config{Tiers: []kvtransfer.TierConfig{cfg}})
	if err != nil {
		panic(fmt.Sprintf("NewOffloadCache: transfer station config invalid: %v", err))
	}
	s.station = st
}

func (s *secondaryTier) store(key kvkey.BlockKey) {
	if _, ok := s.holds[key]; ok {
		return
	}
	if s.capacity > 0 && int64(len(s.holds)) >= s.capacity {
		return
	}
	s.holds[key] = struct{}{}
	for _, n := range s.notify {
		n(key, false)
	}
}

func (s *secondaryTier) drop(key kvkey.BlockKey) {
	if _, had := s.holds[key]; !had {
		return
	}
	delete(s.holds, key)
	for _, n := range s.notify {
		n(key, true)
	}
}

func (s *secondaryTier) has(key kvkey.BlockKey) bool {
	_, ok := s.holds[key]
	return ok
}

// submit queues one job for owner, returning false if the station refuses it.
func (s *secondaryTier) submit(owner *OffloadCache, job kvtransfer.TransferJob, keys []kvkey.BlockKey) bool {
	job.Tier = 0
	id, ok := s.station.Submit(job)
	if ok {
		s.jobs[id] = tierJob{owner: owner, keys: keys, dir: job.Direction}
	}
	return ok
}

// poll advances the tier to now, lands every completed Write in the holdings, and hands each
// completed job to its engine.
func (s *secondaryTier) poll(now int64) {
	for _, id := range s.station.Poll(now) {
		j, ok := s.jobs[id]
		if !ok {
			continue
		}
		delete(s.jobs, id)
		if j.dir == kvtransfer.Write {
			for _, k := range j.keys {
				s.store(k)
			}
		}
		s.done[j.owner] = append(s.done[j.owner], j)
	}
}

// takeDone returns and forgets owner's completed jobs.
func (s *secondaryTier) takeDone(owner *OffloadCache) []tierJob {
	d := s.done[owner]
	delete(s.done, owner)
	return d
}

// activeJobs is how many dir jobs are in service on the tier.
func (s *secondaryTier) activeJobs(dir kvtransfer.Direction) int { return s.station.ActiveJobs(0, dir) }

// lookupSecondary returns the index of the lowest-indexed tier that holds key, or
// (0, false) if no tier holds it. Order depends only on tier index, never on
// arrival order (BC-C6, INV-6). It is a free function over the tier slice so it
// carries no reference to the composing OffloadCache.
func lookupSecondary(tiers []*secondaryTier, key kvkey.BlockKey) (int, bool) {
	for i, tier := range tiers {
		if tier.has(key) {
			return i, true
		}
	}
	return 0, false
}

// stationTierConfig is the station tier that serves t: its thread counts, its device physics
// and its device-wide queue depth (all of a tier's threads do I/O on the one device).
func stationTierConfig(t sim.KVOffloadTier) kvtransfer.TierConfig {
	c := kvtransfer.TierConfig{
		NRead:                  int(t.NReadThreads),
		NWrite:                 int(t.NWriteThreads),
		ReadBaseTicks:          int64(math.Round(t.BaseLatency)),
		WriteBaseTicks:         int64(math.Round(t.BaseLatency)),
		ReadBytesPerTick:       t.ReadBandwidth,
		WriteBytesPerTick:      t.WriteBandwidth,
		SaturationQueueDepth:   int(t.SaturationQueueDepth),
		SingleTransferFraction: t.SingleTransferFraction,
		MaxQueueDepth:          0, // unbounded (vLLM deque default); keeps prepareStore→Submit leak-safe
		DeviceDepth:            true,
	}
	if price := t.ServiceTime; price != nil {
		c.ServiceTime = func(dir kvtransfer.Direction, bytes int64, q int) int64 {
			return price(dir == kvtransfer.Write, bytes, q)
		}
	}
	return c
}

// sharedTier is engine tier i of cfg: the node's shared tier when cfg binds one (joining it
// if another engine built it), else a private one.
func sharedTier(cfg sim.KVOffloadConfig, i int) *secondaryTier {
	t := cfg.Tiers[i]
	chunk := cfg.PerBlockBytes * cfg.BlocksPerChunk
	var capacity int64
	if t.CapacityBytes > 0 {
		capacity = max(1, t.CapacityBytes/chunk)
	}
	g := t.Group
	if g == nil {
		return newStationTier(stationTierConfig(t), capacity)
	}
	if st, ok := g.Tier.(*secondaryTier); ok {
		st.join()
		return st
	}
	st := newStationTier(stationTierConfig(t), capacity)
	g.Tier = st
	return st
}

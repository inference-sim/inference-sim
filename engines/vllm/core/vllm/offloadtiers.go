// Original to blis-vllm.

package vllm

import (
	"encoding/json"
	"fmt"
	"sort"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/units"

	"github.com/inference-sim/blis-vllm/core/connector/offloading"
	kvtier "github.com/inference-sim/blis-vllm/core/connector/tiering"
	"github.com/inference-sim/blis-vllm/core/sched"
)

// modelledTiers are the secondary tier types TieringOffloadingSpec registers
// (SecondaryTierFactory.register_tier, vllm/v1/kv_offload/tiering/factory.py:146-174@4ba787ec)
// that the engine builds beside the CPU tier, each run by the connector's TieringOffloadingManager
// (offloading.Config.Manager). buildOffload refuses every other registered type by name; a type
// vLLM does not register is refused first, as code_not_visible. surface.json marks exactly these
// modelled.
var modelledTiers = map[string]bool{"fs": true, "obj": true, "p2p": true}

// The secondary tiers through engine.Engine.
//
// TieringOffloadingSpec's manager (core/connector/tiering) runs beside the CPU tier and decides;
// it moves no bytes and reads no clock. Each job between the CPU tier and a secondary tier is
// reported as a Transfer among the next batch's TransfersStarted: a cascade as a store to the
// tier's catalog device, a promotion as a load from it, both on the node that holds the CPU tier.
// Its Tier is vLLM's tier label ("<index>:<type>", tiering/metrics.py:64-66@4ba787ec), its Move.Tier
// the device_class the tier block names, and its bytes the job's chunks at the CPU tier's chunk
// size over every worker's copy (the fs tier reads and writes the primary view's chunks whole,
// fs/manager.py:168-172). When the simulator reports the transfer done, the job runs (FSTier.Run),
// and the manager collects it at its next call, as vLLM's get_finished_jobs does.
//
// Two of the tier's threads are timed by named substitution (blis-vllm#190): a lookup batch
// flushed at a schedule's on_schedule_end resolves when that batch's output is applied, and a job
// is one transfer started with the first schedule after its submission, whatever the thread
// pool's size.

// tierLookupIssue is the fs tier's timing substitutions; tierIssue the tier keys not modelled.
const (
	tierTimingIssue = issueBase + "190"
	tierIssue       = issueBase + "63"
)

// TierPricer prices a secondary tier's job: bytes between the CPU tier and t.Move.Tier, the
// tier's catalog device, as the kernel prices that device's offload_tier_<device>_* coefficients.
// An engine whose pricer is not one prices no tier job itself (TransferTime reports false), and
// the simulator prices it from its Move.
type TierPricer interface {
	TierTransferTime(t engine.Transfer, bytes int64) engine.Duration
}

// TierStorage gives each secondary tier the files its root_dir names. Engines that name one
// root_dir through one TierStorage share its block files, as vLLM instances do over one shared
// file system (fs/manager.py:86-104@4ba787ec); the caller keeps different models' files apart.
type TierStorage func(rootDir string) kvtier.FileStore

// tierSet is an offload's secondary tiers as the engine runs them.
type tierSet struct {
	mgr   *kvtier.Manager
	tiers []*builtTier
	// chunkBytes is the CPU tier's bytes a chunk over every worker's copy; blocksPerChunk and
	// blockSize size a job in GPU blocks and tokens.
	chunkBytes     int64
	blocksPerChunk int
	blockSize      int
	// inFlight are the tier jobs started as transfers, by transfer; started the jobs reported.
	inFlight map[engine.TransferID]tierJob
	started  map[tierJob]bool
}

type tierJob struct{ tier, id int }

// builtTier is one secondary tier: its label, its catalog device, the tier as the manager calls
// it and as the host times it, and an fs tier's files or an obj tier's bucket.
type builtTier struct {
	label   string
	device  string
	rootDir string
	tier    kvtier.Tier
	host    hostTier
	// fs is the tier when it is an fs tier, nil otherwise.
	fs    *kvtier.FSTier
	files *tierFiles
	// bucket is an obj tier's store_config bucket, and objects its objects.
	bucket  string
	objects *tierBucket
	// p2p is the engine's end of a p2p tier, nil for any other.
	p2p *p2pPeer
}

// hostTier is what the engine drives of a tier's transport: the jobs waiting for a transfer, a
// job's end when its transfer is done, and the lookup worker's batches at the step boundary.
type hostTier interface {
	Jobs() []kvtier.FSJob
	Run(id int)
	ResolveLookups()
}

// peerHost is a host whose jobs are moves between pools: a WRITE to a peer (the p2p tier's), not a
// move to or from a tier. PeerEnd is the endpoint the job's bytes end at; the job starts at the
// engine's own. Such a job may fail, and Fail ends it so.
type peerHost interface {
	PeerEnd(id int) plan.Endpoint
	Fail(id int)
}

// tierFiles is the store an fs tier's files are in: its own until the engine is given a
// TierStorage.
type tierFiles struct{ kvtier.FileStore }

// fsKeys are the fs tier's keys in secondary_tiers (FileSystemTierManager.__init__,
// fs/manager.py:109-120@4ba787ec), and device_class, the simulator's: the catalog device the tier's
// moves are priced on. type and module_path are the factory's (tiering/factory.py:59-63).
var fsKeys = map[string]bool{"type": true, "root_dir": true, "n_read_threads": true, "n_write_threads": true,
	"enable_kv_events": true, "locality": true, "device_class": true}

// buildTiers builds the block's secondary tiers, refusing what it does not model; nil for a block
// with none. bpc is the CPU tier's blocks a chunk, blockSize the GPU block's tokens, chunkBytes the
// CPU chunk's bytes over every worker's copy (the primary view's stride), and ns what the fs tier's
// FileMapper reads from the spec.
func buildTiers(c *offloadConfig, bpc, blockSize int, chunkBytes int64, chunks int, ns kvtier.Namespace) (*tierSet, error) {
	if c.Spec != "TieringOffloadingSpec" || len(c.SecondaryTiers) == 0 {
		return nil, nil
	}
	ts := &tierSet{chunkBytes: chunkBytes, blocksPerChunk: bpc, blockSize: blockSize,
		inFlight: map[engine.TransferID]tierJob{}, started: map[tierJob]bool{}}
	for i, raw := range c.SecondaryTiers {
		build := buildFS
		switch raw["type"] {
		case "obj":
			build = buildObj
		case "p2p":
			if ts.peer() != nil {
				return nil, fmt.Errorf("vllm: kv_offload secondary_tiers[%d] (p2p): one p2p tier an engine: it binds one side-channel "+
					"address, the PeerAddr peers reach it at (%s)", i, p2pIssue)
			}
			build = func(i int, raw map[string]any, top map[string]any, chunkBytes int64, ns kvtier.Namespace) (*builtTier, error) {
				return buildP2P(i, raw, top, chunkBytes, chunks, ns, c)
			}
		}
		b, err := build(i, raw, c.Backpressure, chunkBytes, ns)
		if err != nil {
			return nil, err
		}
		if b.p2p != nil {
			b.p2p.mgr = func() *kvtier.Manager { return ts.mgr }
		}
		ts.tiers = append(ts.tiers, b)
	}
	return ts, nil
}

func buildFS(i int, raw map[string]any, topBackpressure map[string]any, chunkBytes int64, ns kvtier.Namespace) (*builtTier, error) {
	at := func(format string, a ...any) error {
		return fmt.Errorf("vllm: kv_offload secondary_tiers[%d] (fs): "+format, append([]any{i}, a...)...)
	}
	keys := make([]string, 0, len(raw))
	for k := range raw {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if k == "backpressure" {
			continue
		}
		if !fsKeys[k] {
			// vLLM passes every other key to FileSystemTierManager, which takes no such argument
			// (create_secondary_tier, tiering/factory.py:59-96).
			return nil, at("%q is not a key of the fs tier (fs/manager.py:109-120@4ba787ec)", k)
		}
	}
	if raw["backpressure"] != nil || topBackpressure != nil {
		// The tier's detector merges the top-level block with its own (tiering/spec.py:329-351@4ba787ec).
		return nil, at("store backpressure is not modelled (%s)", issueBase+"13")
	}
	root, ok := raw["root_dir"].(string)
	if !ok || root == "" {
		return nil, at("root_dir must be stated: FileSystemTierManager requires it (fs/manager.py:113@4ba787ec)")
	}
	for _, k := range []string{"n_read_threads", "n_write_threads"} {
		if v, stated := raw[k]; stated {
			if n, ok := v.(int); !ok || n < 0 {
				return nil, at("%s %v must be a whole number of threads", k, v)
			}
		}
	}
	if threads(raw, "n_read_threads")+threads(raw, "n_write_threads") == 0 {
		// DualQueueThreadPool with no thread runs no job (fs/thread_pool.py:60-112@4ba787ec).
		return nil, at("n_read_threads and n_write_threads are both 0: no thread would run a job")
	}
	var loc kvtier.Locality
	if v, stated := raw["locality"]; stated && v != nil {
		s, _ := v.(string)
		var err error
		if loc, err = kvtier.ParseLocality(s); err != nil {
			return nil, at("%v", err)
		}
	}
	if v, stated := raw["enable_kv_events"]; stated {
		if _, ok := v.(bool); !ok {
			return nil, at("enable_kv_events %v must be true or false", v)
		}
		// The tier's events reach vLLM's KV event publisher, as the CPU tier's do, which the
		// engine does not model (blis-vllm#60): the tier is built without them.
	}
	device, _ := raw["device_class"].(string)
	if device == "" {
		return nil, at("device_class must name the catalog device the tier's files are on: its moves are priced "+
			"on that device's offload_tier_<device>_* coefficients, and a tier that names none is not priced (%s)", tierIssue)
	}
	files := &tierFiles{FileStore: kvtier.MemFiles{}}
	fs := kvtier.NewFSTier(kvtier.FSConfig{RootDir: root, Locality: loc, BlockBytes: chunkBytes, Namespace: ns}, files)
	return &builtTier{label: fmt.Sprintf("%d:fs", i+1), device: device, rootDir: root, tier: fs, host: fs, fs: fs, files: files}, nil
}

// threads is a thread count key, vLLM's default 16 unstated (fs/manager.py:114-115@4ba787ec).
func threads(raw map[string]any, k string) int {
	if v, ok := raw[k].(int); ok {
		return v
	}
	return 16
}

// manager is the Config.Manager that runs the tiers over the connector's CPU tier; nil with none.
func (ts *tierSet) manager() func(cpu *offloading.CPUPrimary) (offloading.Manager, error) {
	if ts == nil {
		return nil
	}
	return func(cpu *offloading.CPUPrimary) (offloading.Manager, error) {
		tiers := make([]kvtier.Tier, len(ts.tiers))
		for i, b := range ts.tiers {
			tiers[i] = b.tier
		}
		// The manager's clock stamps a job's submit time, which only store backpressure reads, and
		// backpressure is refused (buildFS).
		ts.mgr = kvtier.New(kvtier.CPU(cpu), tiers, nil)
		return kvtier.Connector{M: ts.mgr}, nil
	}
}

// secondaryPlanTiers are the pool plan's devices for the secondary tiers: each tier's catalog
// device (device_class), with no capacity of its own (an fs tier has none, fs/manager.py:86-104),
// shared by the node. A tier that names none has no device; buildOffload refuses it.
func secondaryPlanTiers(c *offloadConfig) []plan.KVTier {
	if c.Spec != "TieringOffloadingSpec" {
		return nil
	}
	var out []plan.KVTier
	seen := map[string]bool{}
	for _, t := range c.SecondaryTiers {
		d, _ := t["device_class"].(string)
		if d != "" && !seen[d] {
			seen[d] = true
			out = append(out, plan.KVTier{Device: d, Shared: true})
		}
	}
	return out
}

// share gives each tier the files its root_dir names in s.
func (ts *tierSet) share(s TierStorage) {
	if ts == nil {
		return
	}
	for _, b := range ts.tiers {
		if b.files != nil {
			b.files.FileStore = s(b.rootDir)
		}
	}
}

// transfers reports the tier jobs queued since the last schedule, in tier order and each tier's
// submission order: a store to the tier's device, a load from it.
func (o *offload) tierTransfers() []engine.Transfer { return o.transfersOf(false) }

// peerTransfers are the WRITEs to peers queued since the last report: tierTransfers' share a
// schedule that ran nothing reports too.
func (o *offload) peerTransfers() []engine.Transfer { return o.transfersOf(true) }

func (o *offload) transfersOf(peersOnly bool) []engine.Transfer {
	if o == nil || o.tiers == nil {
		return nil
	}
	ts := o.tiers
	var out []engine.Transfer
	for i, b := range ts.tiers {
		if _, peer := b.host.(peerHost); peersOnly && !peer {
			continue
		}
		for _, j := range b.host.Jobs() {
			key := tierJob{tier: i, id: j.ID}
			if ts.started[key] {
				continue
			}
			ts.started[key] = true
			*o.next++
			id := *o.next
			ts.inFlight[id] = key
			blocks := j.Chunks * ts.blocksPerChunk
			kind, move := engine.TransferStore, plan.TransferToTier
			if !j.Store {
				kind, move = engine.TransferLoad, plan.TransferFromTier
			}
			t := engine.Transfer{ID: id, Request: j.Req, Kind: kind, Tier: b.label,
				Tokens: blocks * ts.blockSize, Blocks: blocks,
				Move: plan.Transfer{Tokens: units.Tokens(blocks * ts.blockSize), From: o.at, To: o.at, Tier: b.device, Kind: move}}
			if ph, ok := b.host.(peerHost); ok {
				// A move between pools names no tier (engine.Transfer.Validate).
				t.Tier = ""
				t.Move = plan.Transfer{Tokens: t.Move.Tokens, From: o.at, To: ph.PeerEnd(j.ID), Kind: plan.TransferPeer}
			}
			out = append(out, t)
		}
	}
	return out
}

// tierDone runs a tier job whose transfer the simulator reports done; false for a transfer that
// is not a tier job's. A failed transfer is refused: the port's tier fails a job only by a
// missing file.
func (o *offload) tierDone(id engine.TransferID, ok bool) (bool, error) {
	if o == nil || o.tiers == nil {
		return false, nil
	}
	key, known := o.tiers.inFlight[id]
	if !known {
		return false, nil
	}
	if ph, peer := o.tiers.tiers[key.tier].host.(peerHost); peer && !ok {
		// A WRITE the simulator reports failed is the tier's to handle (three strikes, then the
		// peer's session is down), not a refusal.
		delete(o.tiers.inFlight, id)
		delete(o.tiers.started, key)
		ph.Fail(key.id)
		return true, nil
	}
	if !ok {
		return true, fmt.Errorf("vllm: secondary tier %s transfer %d failed; a tier job fails only by a missing "+
			"block file or object, or a bucket that refuses writes, in the port", o.tiers.tiers[key.tier].label, id)
	}
	delete(o.tiers.inFlight, id)
	delete(o.tiers.started, key)
	o.tiers.tiers[key.tier].host.Run(key.id)
	return true, nil
}

// tierBytes are a tier job's bytes, for its price; false for a transfer that is not one.
func (o *offload) tierBytes(t engine.Transfer) (int64, bool) {
	if o == nil || o.tiers == nil {
		return 0, false
	}
	key, ok := o.tiers.inFlight[t.ID]
	if !ok {
		return 0, false
	}
	if _, peer := o.tiers.tiers[key.tier].host.(peerHost); peer {
		// A WRITE to a peer is priced by the simulator from its Move, over the fabric.
		return 0, false
	}
	return int64(t.Blocks/o.tiers.blocksPerChunk) * o.tiers.chunkBytes, true
}

// isPeerWrite reports whether the transfer is a WRITE the p2p tier started.
func (o *offload) isPeerWrite(id engine.TransferID) bool {
	if o == nil || o.tiers == nil {
		return false
	}
	key, ok := o.tiers.inFlight[id]
	if !ok {
		return false
	}
	_, peer := o.tiers.tiers[key.tier].host.(peerHost)
	return peer
}

// tierStep is the step boundary for the tiers' lookup workers: each lookup batch flushed so far
// resolves (blis-vllm#190).
func (o *offload) tierStep() {
	if o == nil || o.tiers == nil {
		return
	}
	for _, b := range o.tiers.tiers {
		b.host.ResolveLookups()
	}
}

// ShareTierStorage gives the engine's secondary tiers their files from s, before the engine runs:
// engines given one TierStorage share the block files of each root_dir. An engine with no
// secondary tier ignores it.
func (e *Engine) ShareTierStorage(s TierStorage) {
	if e.off != nil {
		e.off.tiers.share(s)
	}
}

// tierNamespace is what the fs tier's FileMapper reads from the spec (FileMapper.from_offloading_spec,
// vllm/v1/kv_offload/file_mapper.py@4ba787ec), as the engine knows it: the hash and chunk sizes, each
// KV cache group's block size, and the world size, collapsed by parallel_agnostic. A block file's
// name is then the same for every engine of one KV layout; the model is the TierStorage's to keep
// apart.
func tierNamespace(s *sched.Scheduler, bpc, world int) kvtier.Namespace {
	ns := kvtier.Namespace{TokensPerHash: s.KV.HashBlockSize(), BlocksPerFile: bpc, TPSize: world, PPSize: 1,
		PCPSize: 1, DCPSize: 1, ParallelAgnostic: true}
	for g := 0; g < s.KV.NumGroups(); g++ {
		ns.Groups = append(ns.Groups, kvtier.KVGroupName{TokensPerBlock: s.KV.GroupBlockSize(g),
			LayerNames: []string{fmt.Sprintf("group%d", g)}})
	}
	return ns
}

// tierTransferTime is a tier job's price, at least one microsecond, by a pricer that is a
// TierPricer; false otherwise, and the simulator prices the job from its Move.
func tierTransferTime(p Pricer, t engine.Transfer, bytes int64) (engine.Duration, bool) {
	tp, ok := p.(TierPricer)
	if !ok {
		return 0, false
	}
	return max(tp.TierTransferTime(t, bytes), 1), true
}

// TagKVTransferParams is the engine.Request.Tags key whose value is the JSON object of the request's
// kv_transfer_params, as the client gave them (Request.kv_transfer_params, from
// SamplingParams.extra_args): the offloading connector hands it to its tiers (ReqContext), which
// read their keys from it. An engine with no kv_offload ignores it. Of the offload connector's keys
// the engine models kv_load_tiers (which tiers the request may load from); max_load_tokens and
// max_offload_tokens are refused (blis-vllm#203), and a key vLLM's offload connector does not read is
// ignored, as vLLM ignores it.
const TagKVTransferParams = "kv_transfer_params"

// requestParams is the request's kv_transfer_params for the scheduler's request; nil with none, or
// with no offload.
func (o *offload) requestParams(r engine.Request) (map[string]any, error) {
	raw, ok := r.Tags[TagKVTransferParams]
	if o == nil || !ok {
		return nil, nil
	}
	var params map[string]any
	if err := json.Unmarshal([]byte(raw), &params); err != nil {
		return nil, fmt.Errorf("vllm: request %s: tag %s must be a JSON object: %w", r.ID, TagKVTransferParams, err)
	}
	for _, k := range []string{"max_load_tokens", "max_offload_tokens"} {
		if _, stated := params[k]; stated {
			return nil, fmt.Errorf("vllm: request %s: kv_transfer_params %q is not modelled: the offloading connector "+
				"would drop it (%s)", r.ID, k, issueBase+"203")
		}
	}
	return params, nil
}

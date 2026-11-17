// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/tiering/fs/manager.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).
// Also vllm/v1/kv_offload/file_mapper.py (FileMapper), tiering/fs/io.py (the existence and
// short-file rules of batch_store_block and batch_load_block) and tiering/fs/thread_pool.py
// (DualQueueThreadPool's queues, whose timing is the host's).

package tiering

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// FileStore is the file system a tier's root_dir names: the set of block files present. Two
// instances whose fs tiers name the same root_dir on a shared file system share one FileStore,
// which is how vLLM shares KV between them (FileSystemTierManager, fs/manager.py:96-104@4ba787ec).
type FileStore interface {
	Exists(path string) bool
	// Publish puts a block file in place (os.replace of the written temp file, io.py:124-132).
	Publish(path string)
}

// MemFiles is a FileStore held in memory.
type MemFiles map[string]bool

func (f MemFiles) Exists(p string) bool { return f[p] }
func (f MemFiles) Publish(p string)     { f[p] = true }

// Remove deletes a file, as another process on the shared file system may.
func (f MemFiles) Remove(p string) { delete(f, p) }

// KVGroupName is one KV cache group in FileMapper's fields (file_mapper.py:86-92).
type KVGroupName struct {
	TokensPerBlock int
	LayerNames     []string
}

// Namespace is what FileMapper.from_offloading_spec reads from the offloading config
// (file_mapper.py:76-116), already resolved: whether the layout is parallelism-agnostic or
// replicated is decided from the engine's KV layout (offloading/config.py:196-279@4ba787ec).
type Namespace struct {
	ModelName     string
	TokensPerHash int
	// BlocksPerFile is blocks_per_chunk (fs/manager.py:178).
	BlocksPerFile                    int
	TPSize, PPSize, PCPSize, DCPSize int
	Rank                             int
	// DType is the cache dtype without its "torch." prefix (offloading/config.py:289-294).
	DType  string
	Groups []KVGroupName
	// ParallelAgnostic is is_parallelism_agnostic; Replicated is replicated_layout.
	ParallelAgnostic, Replicated bool
}

// fileMapper is FileMapper (file_mapper.py:21-144) as the fs tier builds it (parallel_agnostic
// True, fs/manager.py:175-180), without a canonical format.
type fileMapper struct {
	base string
	rank int
}

func newFileMapper(root string, n Namespace) fileMapper {
	// from_offloading_spec: parallel_agnostic = True and (is_parallelism_agnostic or
	// replicated_layout); replicated_layout = True and replicated_layout (:110-114).
	agnostic := n.ParallelAgnostic || n.Replicated
	tp, pp, pcp, dcp, rank := n.TPSize, n.PPSize, n.PCPSize, n.DCPSize, n.Rank
	if agnostic {
		tp, pp, pcp, dcp, rank = 1, 1, 1, 1, 0
	}
	f := pyObject{
		{"blocks_per_file", pyInt(n.BlocksPerFile)},
		{"dcp_size", pyInt(dcp)},
		{"dtype", pyStr(n.DType)},
		{"inference_engine", pyStr("vllm")},
		{"kv_cache_groups", groupsJSON(n.Groups)},
		{"model_name", pyStr(n.ModelName)},
		{"pcp_size", pyInt(pcp)},
		{"pp_size", pyInt(pp)},
		{"tokens_per_hash", pyInt(n.TokensPerHash)},
		{"tp_size", pyInt(tp)},
	}
	if !agnostic {
		f = append(f, pyField{"parallel_agnostic", "false"})
	}
	if n.Replicated {
		f = append(f, pyField{"replicated_layout", "true"})
	}
	// _compute_base_path (:134-144): json.dumps(sort_keys=True, separators=(",", ":")).
	sum := sha256.Sum256([]byte(f.encode()))
	digest := hex.EncodeToString(sum[:])[:12]
	return fileMapper{base: root + "/" + strings.ReplaceAll(n.ModelName, "/", "_") + "_" + digest, rank: rank}
}

// name is get_file_name (:118-126).
func (m fileMapper) name(k Key) string {
	h := hex.EncodeToString(k.Hash())
	return fmt.Sprintf("%s_r%d/%s/%s_g%d/%s.bin", m.base, m.rank, h[:3], h[3:5], k.Group(), h)
}

func groupsJSON(gs []KVGroupName) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, g := range gs {
		if i > 0 {
			b.WriteByte(',')
		}
		names := make([]string, len(g.LayerNames))
		for j, n := range g.LayerNames {
			names[j] = pyStr(n)
		}
		b.WriteString(pyObject{{"layer_names", "[" + strings.Join(names, ",") + "]"},
			{"tokens_per_block", pyInt(g.TokensPerBlock)}}.encode())
	}
	b.WriteByte(']')
	return b.String()
}

// pyObject is a JSON object written as Python's json.dumps writes it with sort_keys and compact
// separators; each value is already encoded.
type pyField struct{ k, v string }
type pyObject []pyField

func (o pyObject) encode() string {
	s := append(pyObject(nil), o...)
	sort.Slice(s, func(i, j int) bool { return s[i].k < s[j].k })
	var b strings.Builder
	b.WriteByte('{')
	for i, f := range s {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(pyStr(f.k))
		b.WriteByte(':')
		b.WriteString(f.v)
	}
	b.WriteByte('}')
	return b.String()
}

func pyInt(n int) string { return strconv.Itoa(n) }

// pyStr is json.dumps of a str with ensure_ascii (the default): ", \ and control characters
// escaped, and every non-ASCII code point as \uXXXX (UTF-16 surrogates above the BMP).
func pyStr(s string) string {
	var b bytes.Buffer
	b.WriteByte('"')
	for _, r := range s {
		switch {
		case r == '"':
			b.WriteString(`\"`)
		case r == '\\':
			b.WriteString(`\\`)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r == '\b':
			b.WriteString(`\b`)
		case r == '\f':
			b.WriteString(`\f`)
		case r < 0x20 || r > 0x7e:
			if r > 0xffff {
				r -= 0x10000
				fmt.Fprintf(&b, `\u%04x\u%04x`, 0xd800+(r>>10), 0xdc00+(r&0x3ff))
			} else {
				fmt.Fprintf(&b, `\u%04x`, r)
			}
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}

// FSConfig is a secondary tier of type "fs": its keys in secondary_tiers (fs/manager.py:109-120)
// and what the spec hands it.
type FSConfig struct {
	RootDir string
	// Locality is the tier's "locality" key; "" is none.
	Locality Locality
	// Events is enable_kv_events and the global enable_kv_cache_events together (:142-152).
	Events bool
	// BlockBytes is the primary tier's bytes per chunk (primary_kv_view.strides[0], :168-172).
	BlockBytes int64
	Namespace  Namespace
}

type fsJob struct {
	id    int
	paths []string
	keys  []Key
	store bool
	// req is the request the job serves (TransferJob.req_context).
	req string
	// loaded is how many blocks a load read before the first missing file; len(paths) on success.
	loaded int
}

// FSTier is FileSystemTierManager (fs/manager.py:86-353). Its block files are in Files. Its
// DualQueueThreadPool is the host's: a submitted job runs when the host calls Run, and DrainJobs
// runs the rest (Drain, if set, chooses their order; otherwise loads, then stores, each in
// submission order, as one load-priority thread runs them).
type FSTier struct {
	cfg    FSConfig
	files  FileStore
	mapper fileMapper
	lookup *asyncLookup
	events []Event
	// queued are the submitted jobs not yet run, in submission order; ran are the ones that ran, in
	// the order they ran, until FinishedJobs collects them.
	queued []*fsJob
	ran    []*fsJob
	// Drain, if set, runs every queued job when the manager drains the tier (reset_cache).
	Drain func(t *FSTier)
}

// NewFSTier builds an fs tier over files.
func NewFSTier(cfg FSConfig, files FileStore) *FSTier {
	t := &FSTier{cfg: cfg, files: files, mapper: newFileMapper(cfg.RootDir, cfg.Namespace)}
	// FsAsyncLookupManager.batch_lookup (:76-83): whether each block file exists.
	t.lookup = newAsyncLookup(func(keys []Key, _ *ReqContext) []bool {
		out := make([]bool, len(keys))
		for i, k := range keys {
			out[i] = t.files.Exists(t.mapper.name(k))
		}
		return out
	})
	return t
}

func (t *FSTier) Type() string                    { return "fs" }
func (t *FSTier) Medium() Medium                  { return MediumStorage }
func (t *FSTier) Locality() Locality              { return t.cfg.Locality }
func (t *FSTier) HitSource() string               { return hitSourceOf(MediumStorage) }
func (t *FSTier) BlockBytes() int64               { return t.cfg.BlockBytes }
func (t *FSTier) Backpressure() Detector          { return nil }
func (t *FSTier) OffThreadPolling() bool          { return false }
func (t *FSTier) HasPendingWork() bool            { return false }
func (t *FSTier) Touch([]Key, *ReqContext)        {}
func (t *FSTier) ServeExternal(Parent)            {}
func (t *FSTier) OnNewRequest(*ReqContext) Policy { return ChunkLevel }

// FileName is the block file a key is stored in.
func (t *FSTier) FileName(k Key) string { return t.mapper.name(k) }

// Lookup is lookup (:215-220): RETRY until the key's batch has resolved.
func (t *FSTier) Lookup(k Key, c *ReqContext) Lookup {
	r := t.lookup.lookup(k, c)
	switch {
	case r == nil:
		return Retry
	case *r:
		return Hit
	}
	return Miss
}

// SubmitStore is submit_store (:222-236).
func (t *FSTier) SubmitStore(j *TransferJob) {
	t.queued = append(t.queued, &fsJob{id: j.ID, paths: t.paths(j.Keys), keys: append([]Key(nil), j.Keys...), store: true,
		req: reqOf(j)})
}

// SubmitLoad is submit_load (:238-276).
func (t *FSTier) SubmitLoad(j *TransferJob) {
	t.queued = append(t.queued, &fsJob{id: j.ID, paths: t.paths(j.Keys), keys: append([]Key(nil), j.Keys...), req: reqOf(j)})
}

func reqOf(j *TransferJob) string {
	if j.Req == nil {
		return ""
	}
	return j.Req.ReqID
}

// FSJob is a job the tier holds queued: what a host needs to time it.
type FSJob struct {
	ID int
	// Store is a store (primary → tier); otherwise a load (a promotion, tier → primary).
	Store bool
	// Chunks is the job's chunk count, each a block file of BlockBytes.
	Chunks int
	// Req is the request the job serves.
	Req string
}

// Jobs are the queued jobs, in submission order.
func (t *FSTier) Jobs() []FSJob {
	out := make([]FSJob, 0, len(t.queued))
	for _, j := range t.queued {
		out = append(out, FSJob{ID: j.id, Store: j.store, Chunks: len(j.keys), Req: j.req})
	}
	return out
}

func (t *FSTier) paths(keys []Key) []string {
	out := make([]string, len(keys))
	for i, k := range keys {
		out[i] = t.mapper.name(k)
	}
	return out
}

// Queued are the ids of the jobs submitted and not yet run, in submission order, and whether each
// is a store.
func (t *FSTier) Queued() (ids []int, stores []bool) {
	for _, j := range t.queued {
		ids, stores = append(ids, j.id), append(stores, j.store)
	}
	return ids, stores
}

// Run runs one queued job, as a pool thread does: a store publishes each block file not already
// present (_store_block skips an existing file, io.py:98-100); a load reads each file in order and
// stops at the first that is missing (batch_load_block, io.py:198-226). What the job's result
// means for the tier (events, misses) is decided when the manager collects it, in FinishedJobs.
func (t *FSTier) Run(id int) {
	idx := -1
	for i, j := range t.queued {
		if j.id == id {
			idx = i
			break
		}
	}
	if idx < 0 {
		panic(fmt.Sprintf("tiering: fs job %d is not queued", id))
	}
	j := t.queued[idx]
	t.queued = append(t.queued[:idx], t.queued[idx+1:]...)
	j.loaded = len(j.paths)
	if j.store {
		for _, p := range j.paths {
			if !t.files.Exists(p) {
				t.files.Publish(p)
			}
		}
	} else {
		for i, p := range j.paths {
			if !t.files.Exists(p) {
				j.loaded = i
				break
			}
		}
	}
	t.ran = append(t.ran, j)
}

// RunAll runs every queued job: loads, then stores, each in submission order.
func (t *FSTier) RunAll() {
	for _, store := range []bool{false, true} {
		for _, j := range append([]*fsJob(nil), t.queued...) {
			if j.store == store {
				t.Run(j.id)
			}
		}
	}
}

// ResolveLookups runs the lookup batches flushed so far, as the lookup worker thread does.
func (t *FSTier) ResolveLookups() { t.lookup.resolve() }

// PendingLookupBatches is how many flushed lookup batches have not resolved.
func (t *FSTier) PendingLookupBatches() int { return t.lookup.submitted() }

// FinishedJobs is get_finished_jobs (:278-325): the jobs that ran since the last call, in the
// order they ran. A stored job is reported as a BlockStored event when events are on; a load that
// stopped part-way keeps its first blocks and marks the rest a miss.
func (t *FSTier) FinishedJobs() []JobResult {
	var out []JobResult
	for _, j := range t.ran {
		res := JobResult{ID: j.id, Success: j.loaded == len(j.paths)}
		if n := int64(len(j.keys)); n > 0 {
			res.TransferBytes = n * t.cfg.BlockBytes
		}
		if j.store {
			if t.cfg.Events && len(j.keys) > 0 {
				t.events = append(t.events, Event{Keys: j.keys, Medium: MediumStorage, Locality: t.cfg.Locality})
			}
		} else if !res.Success {
			if j.loaded > 0 {
				res.SuccessfulKeys = append([]Key(nil), j.keys[:j.loaded]...)
			}
			t.lookup.markMiss(j.keys[j.loaded:])
		}
		out = append(out, res)
	}
	t.ran = nil
	return out
}

// TakeEvents is take_events (:327-331).
func (t *FSTier) TakeEvents() []Event {
	out := t.events
	t.events = nil
	return out
}

// DrainJobs is drain_jobs (:333-336).
func (t *FSTier) DrainJobs() {
	if t.Drain != nil {
		t.Drain(t)
	} else {
		t.RunAll()
	}
	if len(t.queued) > 0 {
		panic("tiering: fs jobs still queued after the tier drained")
	}
}

// OnRequestFinished is on_request_finished (:338-339).
func (t *FSTier) OnRequestFinished(c *ReqContext) { t.lookup.cleanup(c.ReqID) }

// OnScheduleEnd is on_schedule_end (:341-343): the step's lookups go to the worker.
func (t *FSTier) OnScheduleEnd(ScheduleEnd) { t.lookup.flush() }

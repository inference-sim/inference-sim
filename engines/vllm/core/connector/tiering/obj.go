// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/tiering/obj/manager.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).
// Also tiering/obj/config.py (ObjStoreConfig), whose connection keys reach only the transport.

package tiering

import "fmt"

// Bucket is the object store an obj tier's store_config names (bucket and endpoint_override,
// obj/config.py:18-25@4ba787ec). Two instances whose obj tiers name the same bucket share one
// Bucket, which is how vLLM shares KV between them through an object store. An object exists once
// a WRITE that put it has finished.
//
// The tier reaches the bucket only through NIXL's OBJ backend, which reports each transfer as a
// whole (obj/manager.py:393-398): Write puts every object of one transfer or none, and Read is
// whether every object of one transfer is present.
type Bucket interface {
	Exists(key string) bool
	// Write puts the objects of one WRITE transfer; false is a transfer that ended in error.
	Write(keys []string) bool
	// Read is a READ transfer of the objects; false is one that ended in error (a missing object).
	Read(keys []string) bool
}

// MemBucket is a Bucket held in memory.
type MemBucket struct {
	objects map[string]bool
	// Denied makes every WRITE end in error, as a bucket that refuses writes.
	Denied bool
}

// NewMemBucket is an empty bucket.
func NewMemBucket() *MemBucket { return &MemBucket{objects: map[string]bool{}} }

func (b *MemBucket) Exists(k string) bool { return b.objects[k] }

func (b *MemBucket) Write(keys []string) bool {
	if b.Denied {
		return false
	}
	for _, k := range keys {
		b.objects[k] = true
	}
	return true
}

func (b *MemBucket) Read(keys []string) bool {
	for _, k := range keys {
		if !b.objects[k] {
			return false
		}
	}
	return true
}

// Put stores an object, as another process writing to the bucket does.
func (b *MemBucket) Put(k string) { b.objects[k] = true }

// Remove deletes an object, as another process may.
func (b *MemBucket) Remove(k string) { delete(b.objects, k) }

// ObjConfig is a secondary tier of type "obj": its keys in secondary_tiers
// (ObjectStoreSecondaryTierManager.__init__, obj/manager.py:106-117) and what the spec hands it.
// store_config's connection keys and io_threads reach only NIXL: they change where and how fast
// the bytes move, which the host prices, not what the tier decides.
type ObjConfig struct {
	// Prefix is the "prefix" key prepended to every object key; "" is none.
	Prefix string
	// Locality is the tier's "locality" key; "" is none.
	Locality Locality
	// Events is enable_kv_events and the global enable_kv_cache_events together (:141-151).
	Events bool
	// BlockBytes is the primary tier's bytes per chunk (primary_kv_view.strides[0], :184-188).
	BlockBytes int64
	Namespace  Namespace
}

// objXfer is one NIXL transfer in flight (TransferEntry, :60-63).
type objXfer struct {
	id    int
	names []string
	write bool
	// req is the request the job serves.
	req string
	// done and ok are the transfer's state: PROC until it has run, then DONE (ok) or ERR.
	done, ok bool
}

// ObjTier is ObjectStoreSecondaryTierManager (obj/manager.py:97-464). Its objects are in a Bucket.
// NIXL's transfers are the host's: a submitted transfer is in flight until the host calls Run,
// which ends it; DrainJobs ends every one in flight.
type ObjTier struct {
	cfg    ObjConfig
	bucket Bucket
	mapper fileMapper
	lookup *asyncLookup
	events []Event
	// storeKeys are the keys of in-flight stores, kept only when events are on (:152-153);
	// loadKeys those of in-flight loads (:154-156).
	storeKeys, loadKeys map[int][]Key
	// inflight are the transfers submitted and not yet collected, in submission order (the
	// _transfers dict); pending are results awaiting FinishedJobs (_pending_results).
	inflight []*objXfer
	pending  []JobResult
	// refuse is how many of the next submissions NIXL refuses.
	refuse int
}

// NewObjTier builds an obj tier over bucket.
func NewObjTier(cfg ObjConfig, bucket Bucket) *ObjTier {
	// root_dir is "<prefix>/" or "" (:170), so a prefixed key holds "<prefix>//" and an unprefixed
	// one starts "/" (FileMapper._compute_base_path, file_mapper.py:134-144).
	root := ""
	if cfg.Prefix != "" {
		root = cfg.Prefix + "/"
	}
	t := &ObjTier{cfg: cfg, bucket: bucket, mapper: newFileMapper(root, cfg.Namespace),
		storeKeys: map[int][]Key{}, loadKeys: map[int][]Key{}}
	// ObjAsyncLookupManager.batch_lookup (:81-94): one existence probe per key.
	t.lookup = newAsyncLookup(func(keys []Key, _ *ReqContext) []bool {
		out := make([]bool, len(keys))
		for i, k := range keys {
			out[i] = t.bucket.Exists(t.mapper.name(k))
		}
		return out
	})
	return t
}

func (t *ObjTier) Type() string                    { return "obj" }
func (t *ObjTier) Medium() Medium                  { return MediumStorage }
func (t *ObjTier) Locality() Locality              { return t.cfg.Locality }
func (t *ObjTier) HitSource() string               { return hitSourceOf(MediumStorage) }
func (t *ObjTier) BlockBytes() int64               { return t.cfg.BlockBytes }
func (t *ObjTier) Backpressure() Detector          { return nil }
func (t *ObjTier) OffThreadPolling() bool          { return false }
func (t *ObjTier) HasPendingWork() bool            { return false }
func (t *ObjTier) Touch([]Key, *ReqContext)        {}
func (t *ObjTier) ServeExternal(Parent)            {}
func (t *ObjTier) OnNewRequest(*ReqContext) Policy { return ChunkLevel }

// ObjectName is the object key a key is stored under.
func (t *ObjTier) ObjectName(k Key) string { return t.mapper.name(k) }

// Lookup is lookup (:290-294): RETRY until the key's probe batch has resolved.
func (t *ObjTier) Lookup(k Key, c *ReqContext) Lookup {
	r := t.lookup.lookup(k, c)
	switch {
	case r == nil:
		return Retry
	case *r:
		return Hit
	}
	return Miss
}

// RefuseNextSubmission makes NIXL refuse the next submitted transfer (agent.transfer returns
// "ERR", :275-282). A refused transfer's failure is queued at once, ahead of any in-flight
// transfer's later completion.
func (t *ObjTier) RefuseNextSubmission() { t.refuse++ }

// SubmitStore is submit_store (:296-302).
func (t *ObjTier) SubmitStore(j *TransferJob) {
	if t.cfg.Events {
		t.storeKeys[j.ID] = append([]Key(nil), j.Keys...)
	}
	t.submit(j, true)
}

// SubmitLoad is submit_load (:304-309).
func (t *ObjTier) SubmitLoad(j *TransferJob) {
	t.loadKeys[j.ID] = append([]Key(nil), j.Keys...)
	t.submit(j, false)
}

// submit is _submit_transfer (:230-288).
func (t *ObjTier) submit(j *TransferJob, write bool) {
	if t.refuse > 0 {
		t.refuse--
		t.pending = append(t.pending, JobResult{ID: j.ID})
		return
	}
	names := make([]string, len(j.Keys))
	for i, k := range j.Keys {
		names[i] = t.mapper.name(k)
	}
	req := ""
	if j.Req != nil {
		req = j.Req.ReqID
	}
	t.inflight = append(t.inflight, &objXfer{id: j.ID, names: names, write: write, req: req})
}

// Jobs are the transfers in flight that have not run, in submission order, in the shape the fs
// tier reports its queued jobs: what a host needs to time each one (a WRITE is a store).
func (t *ObjTier) Jobs() []FSJob {
	var out []FSJob
	for _, x := range t.inflight {
		if !x.done {
			out = append(out, FSJob{ID: x.id, Store: x.write, Chunks: len(x.names), Req: x.req})
		}
	}
	return out
}

// Run ends one transfer in flight, as NIXL does when its last byte has moved: a WRITE puts its
// objects, a READ needs every one of them. Its result reaches the tier when the manager next
// collects (FinishedJobs).
func (t *ObjTier) Run(id int) {
	for _, x := range t.inflight {
		if x.id == id && !x.done {
			x.done = true
			if x.write {
				x.ok = t.bucket.Write(x.names)
			} else {
				x.ok = t.bucket.Read(x.names)
			}
			return
		}
	}
	panic(fmt.Sprintf("tiering: obj transfer %d is not in flight", id))
}

// RunAll ends every transfer in flight, in submission order.
func (t *ObjTier) RunAll() {
	for _, x := range t.inflight {
		if !x.done {
			t.Run(x.id)
		}
	}
}

// ResolveLookups runs the probe batches flushed so far, as the lookup worker thread does.
func (t *ObjTier) ResolveLookups() { t.lookup.resolve() }

// PendingLookupBatches is how many flushed probe batches have not resolved.
func (t *ObjTier) PendingLookupBatches() int { return t.lookup.submitted() }

// poll is _poll_active_transfers (:320-373): each ended transfer, in submission order, moves to
// pending; one still in flight (PROC) stays.
func (t *ObjTier) poll() {
	var still []*objXfer
	for _, x := range t.inflight {
		if !x.done {
			still = append(still, x)
			continue
		}
		t.pending = append(t.pending, JobResult{ID: x.id, Success: x.ok})
	}
	t.inflight = still
}

// FinishedJobs is get_finished_jobs (:375-404): the queued results (submission failures, then the
// transfers that ended, in submission order). A successful store is reported as a BlockStored
// event when events are on; a failed load marks all its keys a miss, since NIXL reports a
// transfer as a whole.
func (t *ObjTier) FinishedJobs() []JobResult {
	t.poll()
	out := t.pending
	t.pending = nil
	for _, r := range out {
		if t.cfg.Events {
			keys, ok := t.storeKeys[r.ID]
			delete(t.storeKeys, r.ID)
			if ok && r.Success && len(keys) > 0 {
				t.events = append(t.events, Event{Keys: keys, Medium: MediumStorage, Locality: t.cfg.Locality})
			}
		}
		if keys, ok := t.loadKeys[r.ID]; ok {
			delete(t.loadKeys, r.ID)
			if !r.Success {
				t.lookup.markMiss(failedKeys(keys, r.SuccessfulKeys))
			}
		}
	}
	return out
}

// failedKeys are the load's keys not among those that loaded (:399-403).
func failedKeys(keys, ok []Key) []Key {
	in := map[Key]bool{}
	for _, k := range ok {
		in[k] = true
	}
	var out []Key
	for _, k := range keys {
		if !in[k] {
			out = append(out, k)
		}
	}
	return out
}

// TakeEvents is take_events (:406-409).
func (t *ObjTier) TakeEvents() []Event {
	out := t.events
	t.events = nil
	return out
}

// DrainJobs is drain_jobs (:411-433): it blocks until every transfer has ended, so each one in
// flight runs, and their results wait in pending for the next FinishedJobs.
func (t *ObjTier) DrainJobs() {
	t.RunAll()
	t.poll()
}

// OnRequestFinished is on_request_finished (:311-312).
func (t *ObjTier) OnRequestFinished(c *ReqContext) { t.lookup.cleanup(c.ReqID) }

// OnScheduleEnd is on_schedule_end (:314-315): the step's probes go to the worker.
func (t *ObjTier) OnScheduleEnd(ScheduleEnd) { t.lookup.flush() }

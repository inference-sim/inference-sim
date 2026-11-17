// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/tiering/base.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).
// Also vllm/v1/kv_offload/base.py (LookupResult, OffloadPolicy, Medium, Locality, TierMatcher,
// TierFilter, ReqContext, ScheduleEndContext, OffloadingEvent).

// Package tiering is vLLM's offload tiering layer: TieringOffloadingManager
// (vllm/v1/kv_offload/tiering/manager.py:207@4ba787ec), which puts a CPU primary tier in front of
// zero or more secondary tiers, and the secondary tiers it composes (SecondaryTierManager,
// tiering/base.py:135). A chunk stored to the primary tier is cascaded to every secondary tier; a
// chunk a secondary tier holds is promoted into the primary tier before the GPU can load it.
//
// The package decides; it does not move bytes or read a clock. Every transfer between the primary
// tier and a secondary tier is a job whose completion the host reports (the transport, design §7's
// route table: "the bytes ... the oracle treats as transport"), and every asynchronous lookup a tier
// makes resolves when the host says so. The host prices each job as a move (SUB-21) and decides
// when it ends.
package tiering

import (
	"fmt"
	"strings"
)

// Key is an OffloadKey: a block hash with its KV cache group index as 4 big-endian bytes
// (make_offload_key, vllm/v1/kv_offload/base.py:30-32@4ba787ec).
type Key string

// MakeKey is make_offload_key.
func MakeKey(hash []byte, group int) Key {
	b := append(append(make([]byte, 0, len(hash)+4), hash...), byte(group>>24), byte(group>>16), byte(group>>8), byte(group))
	return Key(b)
}

// Hash is get_offload_block_hash (base.py:35-37).
func (k Key) Hash() []byte { return []byte(k[:len(k)-4]) }

// Group is get_offload_group_idx (base.py:40-42).
func (k Key) Group() int {
	n := len(k)
	return int(k[n-4])<<24 | int(k[n-3])<<16 | int(k[n-2])<<8 | int(k[n-1])
}

// Lookup is LookupResult (base.py:121-127).
type Lookup int

const (
	Miss Lookup = iota
	Hit
	HitPending // found, not yet readable
	Retry      // location not known yet; ask again later
)

func (l Lookup) String() string {
	return [...]string{"MISS", "HIT", "HIT_PENDING", "RETRY"}[l]
}

// Policy is OffloadPolicy (base.py:130-137): how a tier wants a request's chunks offloaded.
type Policy int

const (
	ChunkLevel   Policy = iota // each chunk as it is computed (the default)
	RequestLevel               // every chunk of the request, including those already in the primary tier
)

func (p Policy) String() string { return [...]string{"CHUNK_LEVEL", "REQUEST_LEVEL"}[p] }

// Medium is Medium (base.py:48-52); None is a tier that states no medium.
type Medium string

const (
	MediumNone    Medium = ""
	MediumCPU     Medium = "CPU"
	MediumStorage Medium = "STORAGE"
)

// Locality is Locality (base.py:55-59); None is a tier that states none.
type Locality string

const (
	LocalityNone   Locality = ""
	LocalityLocal  Locality = "LOCAL"
	LocalityRemote Locality = "REMOTE"
)

// ParseLocality is Locality(value) for a tier's "locality" key (fs/manager.py:140@4ba787ec). An
// Enum is looked up by its exact value, so only "LOCAL" and "REMOTE" are accepted (base.py:55-59).
func ParseLocality(s string) (Locality, error) {
	switch Locality(s) {
	case LocalityNone, LocalityLocal, LocalityRemote:
		return Locality(s), nil
	}
	return "", fmt.Errorf("tiering: %q is not a valid Locality (LOCAL, REMOTE)", s)
}

// Matcher is TierMatcher (base.py:62-72). A field left empty matches anything, and so does a
// tier that states no medium or no locality.
type Matcher struct {
	Medium   Medium
	Locality Locality
}

func (m Matcher) matches(med Medium, loc Locality) bool {
	return (m.Medium == "" || med == "" || m.Medium == med) && (m.Locality == "" || loc == "" || m.Locality == loc)
}

// Filter is TierFilter (base.py:75-89): which tiers a request may load from. All is TierFilter.ALL,
// which allows every tier; otherwise a tier is allowed when any matcher matches it, so an empty
// list allows none.
type Filter struct {
	All      bool
	Matchers []Matcher
}

// AllTiers is TierFilter.ALL.
var AllTiers = Filter{All: true}

func (f Filter) allows(med Medium, loc Locality) bool {
	if f.All {
		return true
	}
	for _, m := range f.Matchers {
		if m.matches(med, loc) {
			return true
		}
	}
	return false
}

// KVLoadTiersKey is the kv_transfer_params key that carries a request's load-tier filter
// (KV_LOAD_TIERS_KEY, offloading/scheduler.py:72@4ba787ec).
const KVLoadTiersKey = "kv_load_tiers"

// ParseTierFilter is _parse_tier_filter (offloading/scheduler.py:496-537@4ba787ec) on the JSON value of
// kv_load_tiers: a list of {"medium", "locality"} matchers. A value that is not a list allows every
// tier; an entry that is not an object, or names a medium or locality that is not one (matched
// upper-cased), is skipped; a list with no entry left allows every tier unless it was empty, which
// allows none.
func ParseTierFilter(raw any) Filter {
	list, ok := raw.([]any)
	if !ok {
		return AllTiers
	}
	var ms []Matcher
	for _, e := range list {
		entry, ok := e.(map[string]any)
		if !ok {
			continue
		}
		var m Matcher
		valid := true
		if v, stated := entry["medium"]; stated && v != nil {
			s, isStr := v.(string)
			switch med := Medium(strings.ToUpper(s)); {
			case isStr && (med == MediumCPU || med == MediumStorage):
				m.Medium = med
			default:
				valid = false
			}
		}
		if v, stated := entry["locality"]; valid && stated && v != nil {
			s, isStr := v.(string)
			switch loc := Locality(strings.ToUpper(s)); {
			case isStr && (loc == LocalityLocal || loc == LocalityRemote):
				m.Locality = loc
			default:
				valid = false
			}
		}
		if valid {
			ms = append(ms, m)
		}
	}
	if len(ms) == 0 {
		if len(list) == 0 {
			return Filter{}
		}
		return AllTiers
	}
	return Filter{Matchers: ms}
}

// ReqContext is ReqContext (base.py:92-118) as the tiers see it: one per request, the same value
// for every call about that request.
type ReqContext struct {
	ReqID  string
	Filter Filter
	// Params is kv_transfer_params (base.py:94): the request's KV-transfer parameters as the
	// proxy set them; nil is none. A tier parses what it needs from it in OnNewRequest.
	Params map[string]any
	// Primary is the primary tier's own per-request state.
	Primary any
}

// NewReqContext is a request's context with the default load filter (TierFilter.ALL).
func NewReqContext(id string) *ReqContext { return &ReqContext{ReqID: id, Filter: AllTiers} }

// ScheduleEnd is ScheduleEndContext (base.py:144-150).
type ScheduleEnd struct {
	NewReqIDs, PreemptedReqIDs []string
}

// Event is OffloadingEvent (base.py:167-178): keys stored to (or removed from) a medium.
type Event struct {
	Keys     []Key
	Medium   Medium
	Removed  bool
	Locality Locality
}

// TransferJob is TransferJob (tiering/base.py:71-80): a transfer between the primary tier's chunks
// and a secondary tier. A promotion moves secondary → primary; otherwise primary → secondary (a
// cascade).
type TransferJob struct {
	ID          int
	Keys        []Key
	ChunkIDs    []int
	IsPromotion bool
	Req         *ReqContext
	// SubmitTime is the virtual time the job was created (submit_time, time.monotonic()), from the
	// manager's clock.
	SubmitTime float64
}

// JobResult is JobResult (tiering/base.py:83-95).
type JobResult struct {
	ID      int
	Success bool
	// SuccessfulKeys, for a promotion that partly failed, are the keys that loaded; nil means every
	// key shares Success.
	SuccessfulKeys []Key
	// TransferBytes is transfer_bytes; 0 is unreported (None).
	TransferBytes int64
	// TransferTime is transfer_time in seconds, when HasTransferTime (unset is None, a tier that does
	// not time its jobs). The fs tier times its jobs; the port's is zero, as the host's pool is not
	// the simulator's (blis-vllm#190).
	TransferTime    float64
	HasTransferTime bool
}

// Detector is the part of BackpressureDetector (tiering/backpressure.py) the manager calls.
type Detector interface {
	ShouldStore(numBlocks int) bool
	Update(submitTime float64, numBytes int64)
	OnStoreFailed()
	IsUnderPressure() bool
	Reset()
}

// Parent is ParentManager (tiering/base.py:98-132): what a tier serving a remote request may call
// on the manager. The tier calling it is excluded from the fan-out.
type Parent interface {
	OnNewRequest(c *ReqContext) Policy
	Lookup(k Key, c *ReqContext) Lookup
	CreateStoreJob(keys []Key, c *ReqContext) *TransferJob
	OnRequestFinished(c *ReqContext)
}

// Tier is SecondaryTierManager (tiering/base.py:135-423): one non-primary tier.
type Tier interface {
	// Type is tier_type, the name the tier was registered under.
	Type() string
	Medium() Medium
	Locality() Locality
	// HitSource is cache_hit_source (base.py:157-165): HOST for a CPU medium, DISK for STORAGE,
	// EXTERNAL_UNSPECIFIED otherwise.
	HitSource() string
	// BlockBytes is block_size_bytes, one primary chunk's bytes (primary_kv_view.strides[0]).
	BlockBytes() int64
	// Backpressure is bp_detector; nil is none.
	Backpressure() Detector
	// OffThreadPolling is tolerates_off_thread_polling (base.py:167-175).
	OffThreadPolling() bool

	Lookup(k Key, c *ReqContext) Lookup
	SubmitStore(j *TransferJob)
	SubmitLoad(j *TransferJob)
	FinishedJobs() []JobResult
	HasPendingWork() bool
	TakeEvents() []Event
	Touch(keys []Key, c *ReqContext)
	OnNewRequest(c *ReqContext) Policy
	OnRequestFinished(c *ReqContext)
	ServeExternal(p Parent)
	OnScheduleEnd(e ScheduleEnd)
	// DrainJobs is drain_jobs (base.py:373-388): every submitted job ends before it returns, and its
	// result is queued for FinishedJobs.
	DrainJobs()
}

// hitSourceOf is cache_hit_source's derivation from medium (base.py:159-165).
func hitSourceOf(m Medium) string {
	switch m {
	case MediumCPU:
		return "HOST"
	case MediumStorage:
		return "DISK"
	}
	return "EXTERNAL_UNSPECIFIED"
}

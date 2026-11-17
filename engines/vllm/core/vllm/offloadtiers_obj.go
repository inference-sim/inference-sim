// Original to blis-vllm.

package vllm

import (
	"fmt"
	"sort"

	kvtier "github.com/inference-sim/blis-vllm/core/connector/tiering"
)

// The obj secondary tier through engine.Engine, as the fs tier is (offloadtiers.go): each job is
// one transfer between the CPU tier and the tier's catalog device, a cascade a store to it and a
// promotion a load from it, priced on that device's offload_tier_<device>_* coefficients. vLLM
// moves an obj job as one NIXL OBJ transfer through host memory with an S3 client (design §6.1:
// the storage form over tcp, to a catalog storage device reached over the network). No dataset the
// engine may fit from measures an object store's latency or bandwidth, so the device and its
// coefficients are scenario inputs: a tier that names no device_class is refused (blis-vllm#184).

// objIssue is the obj tier's pricing inputs.
const objIssue = issueBase + "184"

// objKeys are the obj tier's keys in secondary_tiers (ObjectStoreSecondaryTierManager.__init__,
// tiering/obj/manager.py:106-117@4ba787ec), and device_class, the simulator's. type is the
// factory's (tiering/factory.py:59-63).
var objKeys = map[string]bool{"type": true, "store_config": true, "prefix": true, "io_threads": true,
	"enable_kv_events": true, "locality": true, "device_class": true}

// objStoreKeys are ObjStoreConfig's fields (tiering/obj/config.py:18-25@4ba787ec); bucket and
// endpoint_override have no default.
var objStoreKeys = map[string]bool{"bucket": true, "endpoint_override": true, "access_key": true,
	"secret_key": true, "session_token": true, "region": true, "scheme": true, "ca_bundle": true}

// TierBucket is one bucket, as an obj tier sees it: whether an object exists, and a WRITE or READ
// transfer of objects (false is one that ended in error). Like TierFiles it is written as the
// interface literal (an alias, not a named type), and TierBuckets as an alias of a plain func type,
// so that a host reaches Engine.ShareTierBuckets with types of its own and no import of this
// package's tiers.
type TierBucket = interface {
	Exists(key string) bool
	Write(keys []string) bool
	Read(keys []string) bool
}

// TierBuckets gives each obj tier the bucket its store_config names. Engines given one TierBuckets
// share each bucket's objects, as vLLM instances do through one object store.
type TierBuckets = func(bucket string) TierBucket

// tierBucket is the bucket an obj tier's objects are in: its own until the engine is given a
// TierBuckets.
type tierBucket struct{ kvtier.Bucket }

func buildObj(i int, raw map[string]any, topBackpressure map[string]any, chunkBytes int64, ns kvtier.Namespace) (*builtTier, error) {
	at := func(format string, a ...any) error {
		return fmt.Errorf("vllm: kv_offload secondary_tiers[%d] (obj): "+format, append([]any{i}, a...)...)
	}
	for _, k := range sortedAnyKeys(raw) {
		if k == "backpressure" {
			continue
		}
		if !objKeys[k] {
			// vLLM passes every other key to ObjectStoreSecondaryTierManager, which takes no such
			// argument (create_secondary_tier, tiering/factory.py:59-96).
			return nil, at("%q is not a key of the obj tier (obj/manager.py:106-117@4ba787ec)", k)
		}
	}
	if raw["backpressure"] != nil || topBackpressure != nil {
		return nil, at("store backpressure is not modelled (%s)", issueBase+"13")
	}
	store, ok := raw["store_config"].(map[string]any)
	if !ok {
		return nil, at("store_config must be stated: ObjectStoreSecondaryTierManager requires it (obj/manager.py:111@4ba787ec)")
	}
	for _, k := range sortedAnyKeys(store) {
		if !objStoreKeys[k] {
			// ObjStoreConfig(**store_config) takes no such field (obj/manager.py:160).
			return nil, at("store_config %q is not a field of ObjStoreConfig (obj/config.py:18-25@4ba787ec)", k)
		}
	}
	bucket, _ := store["bucket"].(string)
	if bucket == "" {
		return nil, at("store_config bucket must be stated: ObjStoreConfig has no default (obj/config.py:18@4ba787ec)")
	}
	if e, _ := store["endpoint_override"].(string); e == "" {
		return nil, at("store_config endpoint_override must be stated: ObjStoreConfig has no default (obj/config.py:19@4ba787ec)")
	}
	prefix := ""
	if v, stated := raw["prefix"]; stated {
		if prefix, ok = v.(string); !ok {
			return nil, at("prefix %v must be a string", v)
		}
	}
	if v, stated := raw["io_threads"]; stated {
		// The NIXL backend's thread count changes how fast the bytes move, which the device's
		// coefficients price, not what the tier decides.
		if n, ok := v.(int); !ok || n < 1 {
			return nil, at("io_threads %v must be a positive whole number", v)
		}
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
		// As for the fs tier: the tier's events reach vLLM's KV event publisher, which the engine
		// does not model (blis-vllm#60), so the tier is built without them.
	}
	device, _ := raw["device_class"].(string)
	if device == "" {
		return nil, at("device_class must name the catalog device the bucket is on: an obj job is priced on that "+
			"device's offload_tier_<device>_* coefficients, which no allowed dataset measures for an object store, "+
			"so a tier that names none is not priced (%s)", objIssue)
	}
	objects := &tierBucket{Bucket: kvtier.NewMemBucket()}
	t := kvtier.NewObjTier(kvtier.ObjConfig{Prefix: prefix, Locality: loc, BlockBytes: chunkBytes, Namespace: ns}, objects)
	return &builtTier{label: fmt.Sprintf("%d:obj", i+1), device: device, tier: t, host: t, bucket: bucket, objects: objects}, nil
}

func sortedAnyKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// shareBuckets gives each obj tier the bucket its store_config names in b.
func (ts *tierSet) shareBuckets(b TierBuckets) {
	if ts == nil {
		return
	}
	for _, t := range ts.tiers {
		if t.objects != nil {
			t.objects.Bucket = b(t.bucket)
		}
	}
}

// ShareTierBuckets gives the engine's obj tiers their buckets from b, before the engine runs:
// engines given one TierBuckets share the objects of each bucket. An engine with no obj tier
// ignores it.
func (e *Engine) ShareTierBuckets(b TierBuckets) {
	if e.off != nil {
		e.off.tiers.shareBuckets(b)
	}
}

package vllm

import (
	"errors"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"

	kvtier "github.com/inference-sim/blis-vllm/core/connector/tiering"
)

// objTier is TieringOffloadingSpec beside the base block's GPU blocks: a CPU tier of 200 one-block
// chunks of 4 KiB and an obj tier in bucket kv on an object store the scenario names.
const objTier = "kv_offload: {connector: OffloadingConnector, spec: TieringOffloadingSpec, cpu_bytes: 819200, " +
	"bytes_per_block: 4096, secondary_tiers: [{type: obj, prefix: kvcache, device_class: s3_store, " +
	"store_config: {bucket: kv, endpoint_override: 'http://store:9000'}}]}\n"

// sharedBuckets is one object store: every bucket's objects in one MemBucket each.
type sharedBuckets map[string]*kvtier.MemBucket

func (s sharedBuckets) bucket(name string) TierBucket {
	if s[name] == nil {
		s[name] = kvtier.NewMemBucket()
	}
	return s[name]
}

// A prompt one engine stored reaches another engine through one bucket: the writer's CPU tier
// cascades each chunk into the obj tier (a store to the tier's device), and a reader with an empty
// CPU tier naming the same bucket promotes the chunks (a load from the device), then loads them
// from its CPU tier into the GPU, and computes only the rest of the prompt.
func TestAPromptStoredByOneEngineIsLoadedByAnotherThroughOneBucket(t *testing.T) {
	store := sharedBuckets{}
	w := mustEngine(t, objTier)
	w.ShareTierBuckets(store.bucket)
	a := req("a", 96, 2)
	if err := w.Add(a, 0); err != nil {
		t.Fatal(err)
	}
	started, _ := runTiers(t, w, nil)
	stores := 0
	for _, tr := range started {
		if tr.Tier == "1:obj" {
			if tr.Kind != engine.TransferStore || tr.Move.Tier != "s3_store" || tr.Request != "a" {
				t.Errorf("tier transfer %+v, want a's store to s3_store", tr)
			}
			stores += tr.Tokens
		}
	}
	if stores != 96 || store["kv"] == nil {
		t.Fatalf("the writer stored %d tokens to the obj tier, want a's 96", stores)
	}

	r := mustEngine(t, objTier)
	r.ShareTierBuckets(store.bucket)
	b := req("b", 0, 2)
	b.Prompt = append(append([]int32(nil), a.Prompt...), 7, 8, 9, 10)
	if err := r.Add(b, 0); err != nil {
		t.Fatal(err)
	}
	started, first := runTiers(t, r, nil)
	var promoted, loaded int
	for _, tr := range started {
		switch {
		case tr.Tier == "1:obj" && tr.Kind == engine.TransferLoad:
			if tr.Move.Tier != "s3_store" || tr.Request != "b" {
				t.Errorf("promotion %+v, want b's load from s3_store", tr)
			}
			promoted += tr.Tokens
		case tr.Tier == "cpu" && tr.Kind == engine.TransferLoad:
			loaded += tr.Tokens
		}
	}
	if promoted != 96 || loaded != 96 {
		t.Fatalf("the reader promoted %d tokens from the obj tier and loaded %d from its CPU tier, want 96 each", promoted, loaded)
	}
	if first["b"] != 4 {
		t.Fatalf("b's first step computed %d tokens, want the 4 past the shared prefix", first["b"])
	}
}

// Engines naming different buckets share no object: the reader computes the whole prompt.
func TestEnginesOnDifferentBucketsShareNoObject(t *testing.T) {
	store := sharedBuckets{}
	w := mustEngine(t, objTier)
	w.ShareTierBuckets(store.bucket)
	a := req("a", 96, 2)
	if err := w.Add(a, 0); err != nil {
		t.Fatal(err)
	}
	runTiers(t, w, nil)
	r := mustEngine(t, strings.Replace(objTier, "bucket: kv", "bucket: other", 1))
	r.ShareTierBuckets(store.bucket)
	if err := r.Add(a, 0); err != nil {
		t.Fatal(err)
	}
	started, _ := runTiers(t, r, nil)
	for _, tr := range started {
		if tr.Kind == engine.TransferLoad {
			t.Fatalf("the reader loaded %+v from another bucket", tr)
		}
	}
}

// An obj tier the engine cannot run, or cannot price, is refused at build, naming the key; one
// with no device names the issue that holds its pricing inputs.
func TestAnObjTierItCannotRunIsRefused(t *testing.T) {
	tier := func(keys string) string {
		return "kv_offload: {connector: OffloadingConnector, spec: TieringOffloadingSpec, cpu_bytes: 819200, " +
			"bytes_per_block: 4096, secondary_tiers: [{type: obj" + keys + "}]}\n"
	}
	sc := ", store_config: {bucket: kv, endpoint_override: 'http://store:9000'}"
	for name, c := range map[string]struct{ extra, want string }{
		"no device":       {tier(sc), "issues/184"},
		"no store_config": {tier(", device_class: s3_store"), "store_config"},
		"no bucket":       {tier(", device_class: s3_store, store_config: {endpoint_override: 'http://store:9000'}"), "bucket"},
		"no endpoint":     {tier(", device_class: s3_store, store_config: {bucket: kv}"), "endpoint_override"},
		"unknown field":   {tier(", device_class: s3_store, store_config: {bucket: kv, endpoint_override: e, retries: 3}"), `"retries"`},
		"unknown key":     {tier(sc + ", device_class: s3_store, root_dir: /kv"), `"root_dir"`},
		"bad locality":    {tier(sc + ", device_class: s3_store, locality: remote"), "remote"},
		"tier pressure":   {tier(sc + ", device_class: s3_store, backpressure: {backpressure_cls: EMABackpressureDetector}"), "issues/13"},
		"no io thread":    {tier(sc + ", device_class: s3_store, io_threads: 0"), "io_threads"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := build(t, c.extra)()
			if err == nil {
				t.Fatal("built, want refused")
			}
			var nv *CodeNotVisibleError
			if errors.As(err, &nv) || !strings.Contains(err.Error(), "secondary_tiers[0] (obj)") || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("refusal %v, want one naming secondary_tiers[0] (obj) and %s", err, c.want)
			}
		})
	}
}

package vllm

import (
	"reflect"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// storedBy runs the requests one after another through an engine with LoRA on and KV events
// enabled, and returns, per request ID, the BlockStored events its steps published (a request's
// events are the ones carried by the results of its own steps).
func storedBy(t *testing.T, rs ...engine.Request) map[string][]engine.KVEvent {
	t.Helper()
	eng, err := build(t, "max_loras: 2\n")()
	if err != nil {
		t.Fatal(err)
	}
	e := eng.(*Engine)
	e.EnableKVEvents()
	out := map[string][]engine.KVEvent{}
	now := engine.Time(0)
	for _, r := range rs {
		if err := e.Add(r, now); err != nil {
			t.Fatal(err)
		}
		for e.s.HasRequests() {
			b, err := e.Schedule(now)
			if err != nil {
				t.Fatal(err)
			}
			if b.Empty() {
				now += 1000
				continue
			}
			now += e.StepTime(b)
			res, err := e.Complete(b.ID, engine.Outcome{}, now)
			if err != nil {
				t.Fatal(err)
			}
			for _, ev := range res.KVEvents {
				if !ev.Removed {
					out[r.ID] = append(out[r.ID], ev)
				}
			}
		}
	}
	return out
}

// A block an adapter's request stores says so: vLLM's BlockStored carries lora_name and
// lora_id and puts the adapter's ("lora", name, path) in extra_keys, so a router can keep the
// adapter's blocks apart from the base model's. The same prompt without an adapter publishes
// none of it.
func TestKVEventsCarryTheAdaptersIdentity(t *testing.T) {
	lora := req("a", 48, 2)
	lora.Adapter, lora.AdapterPath = "sql", "/adapters/sql"
	base := req("a", 48, 2)
	base.ID = "b"
	got := storedBy(t, lora, base)
	if len(got["a"]) == 0 || len(got["b"]) == 0 {
		t.Fatalf("expected stored events from both requests, got %d and %d", len(got["a"]), len(got["b"]))
	}
	wantKey := []any{[]any{"lora", "sql", "/adapters/sql"}}
	for i, ev := range got["a"] {
		if ev.LoRAName != "sql" || ev.LoRAID == nil || *ev.LoRAID != 1 {
			t.Errorf("adapter block %d: LoRAName %q LoRAID %v, want sql and 1", i, ev.LoRAName, ev.LoRAID)
		}
		if !reflect.DeepEqual(ev.ExtraKeys, wantKey) {
			t.Errorf("adapter block %d: ExtraKeys %#v, want %#v", i, ev.ExtraKeys, wantKey)
		}
		if err := ev.Validate(); err != nil {
			t.Errorf("adapter block %d: %v", i, err)
		}
	}
	for i, ev := range got["b"] {
		if ev.LoRAName != "" || ev.LoRAID != nil || ev.ExtraKeys != nil {
			t.Errorf("base block %d carries an identity: %+v", i, ev)
		}
		if ev.Hash == got["a"][min(i, len(got["a"])-1)].Hash {
			t.Errorf("base block %d hashes as the adapter's block: the adapter must key the hash", i)
		}
	}
}

// A cache salt keys the request's first block only, and vLLM publishes it as a plain string
// among extra_keys; a salted request with an adapter has both. The events carry the salt in
// ExtraKeys and leave CacheSalt (SGLang's field) empty, as the contract says.
func TestKVEventsCarryTheCacheSaltOnTheFirstBlockOnly(t *testing.T) {
	r := req("s", 48, 2)
	r.CacheSalt = "tenant-7"
	got := storedBy(t, r)["s"]
	if len(got) < 3 {
		t.Fatalf("want at least 3 stored blocks, got %d", len(got))
	}
	if want := []any{"tenant-7"}; !reflect.DeepEqual(got[0].ExtraKeys, want) {
		t.Errorf("first block ExtraKeys %#v, want %#v", got[0].ExtraKeys, want)
	}
	for i, ev := range got[1:] {
		if ev.ExtraKeys != nil {
			t.Errorf("block %d carries ExtraKeys %#v: the salt keys the first block only", i+1, ev.ExtraKeys)
		}
	}
	for i, ev := range got {
		if ev.CacheSalt != "" {
			t.Errorf("block %d sets CacheSalt %q beside vLLM's extra keys", i, ev.CacheSalt)
		}
		if err := ev.Validate(); err != nil {
			t.Errorf("block %d: %v", i, err)
		}
	}
}

// Every published block names its KV-cache group and the group's kind, as vLLM's event does.
func TestKVEventsNameTheirGroup(t *testing.T) {
	got := storedBy(t, req("g", 48, 2))["g"]
	if len(got) == 0 {
		t.Fatal("no stored events")
	}
	for i, ev := range got {
		if ev.GroupIdx == nil || *ev.GroupIdx != 0 || ev.SpecKind != "full_attention" || ev.SlidingWindow != nil {
			t.Errorf("block %d: group %v kind %q window %v, want 0, full_attention, none", i, ev.GroupIdx, ev.SpecKind, ev.SlidingWindow)
		}
	}
}

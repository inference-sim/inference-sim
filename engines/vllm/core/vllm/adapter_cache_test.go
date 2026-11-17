// Original to blis-vllm.

package vllm

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-vllm/core/lora"
)

// loadsPerBatch runs one single-token request per adapter, one after another, and returns the
// adapters the worker's cache loaded for each request's batch.
func loadsPerBatch(t *testing.T, extra string, adapters ...string) [][]int {
	t.Helper()
	built, err := build(t, extra)()
	if err != nil {
		t.Fatal(err)
	}
	e := built.(*Engine)
	var out [][]int
	for i, a := range adapters {
		r := req("r"+string(rune('a'+i)), 16, 1)
		r.Adapter, r.AdapterPath = a, "/loras/"+a
		if err := e.Add(r, engine.Time(i)); err != nil {
			t.Fatal(err)
		}
		b, err := e.Schedule(engine.Time(i))
		if err != nil || b.Empty() {
			t.Fatalf("request %d: batch %v, err %v", i, b, err)
		}
		var loads []int
		for _, ev := range e.AdapterEvents() {
			if ev.Kind == lora.Load {
				loads = append(loads, ev.Adapter)
			}
		}
		out = append(out, loads)
		if _, err := e.Complete(b.ID, engine.Outcome{}, engine.Time(i)+1); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

// max_cpu_loras is the size of the worker's cache of loaded adapters: a request whose adapter it
// no longer holds loads it again before its batch runs. With one GPU slot and room for one
// adapter, the returning adapter is loaded a second time; with room for three, it is not. Before
// the cache was modelled the engine refused max_cpu_loras above max_loras, and reported no loads.
func TestMaxCPULoRAsDecidesWhichBatchesLoadAnAdapter(t *testing.T) {
	small := loadsPerBatch(t, "max_loras: 1\nmax_cpu_loras: 1\n", "a", "b", "a")
	if want := [][]int{{1}, {2}, {1}}; !equalLoads(small, want) {
		t.Errorf("max_cpu_loras 1: loads per batch %v, want %v (the returning adapter is loaded again)", small, want)
	}
	large := loadsPerBatch(t, "max_loras: 1\nmax_cpu_loras: 3\n", "a", "b", "a")
	if want := [][]int{{1}, {2}, nil}; !equalLoads(large, want) {
		t.Errorf("max_cpu_loras 3: loads per batch %v, want %v (the returning adapter is still cached)", large, want)
	}
}

func equalLoads(a, b [][]int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}

// A max_cpu_loras on an engine without LoRA sizes no cache, so it is refused rather than dropped,
// and one below max_loras is refused as vLLM refuses it (vllm/config/lora.py:123-126).
func TestMaxCPULoRAsRefusals(t *testing.T) {
	for _, c := range []struct{ name, extra, want string }{
		{"without max_loras", "max_cpu_loras: 4\n", "max_cpu_loras"},
		{"below max_loras", "max_loras: 4\nmax_cpu_loras: 2\n", "max_cpu_loras"},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := build(t, c.extra)()
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Errorf("want a refusal naming %q, got %v", c.want, err)
			}
		})
	}
}

// An adapter loaded through vLLM's dynamic load before any request is resident, and the request
// that then names it runs with no load of its own; an adapter nothing loaded is loaded for its
// request. Unloading removes it again. Before LoadAdapter existed the engine could not place an
// adapter ahead of its traffic.
func TestALoadedAdapterNeedsNoLoadForItsRequest(t *testing.T) {
	built, err := build(t, "max_loras: 2\nmax_cpu_loras: 3\n")()
	if err != nil {
		t.Fatal(err)
	}
	e := built.(*Engine)
	if _, err := e.LoadAdapter("a"); err != nil {
		t.Fatal(err)
	}
	if got := e.ResidentAdapterIDs(); len(got) != 1 || got[0] != "a" {
		t.Fatalf("resident after loading a = %v, want [a]", got)
	}
	if e.AdapterCapacity() != 3 {
		t.Errorf("capacity = %d, want max_cpu_loras 3", e.AdapterCapacity())
	}
	loadsFor := func(adapter string, at engine.Time) []int {
		r := req("r-"+adapter, 16, 1)
		r.Adapter, r.AdapterPath = adapter, "/loras/"+adapter
		if err := e.Add(r, at); err != nil {
			t.Fatal(err)
		}
		b, err := e.Schedule(at)
		if err != nil || b.Empty() {
			t.Fatalf("batch %v, err %v", b, err)
		}
		var loads []int
		for _, ev := range e.AdapterEvents() {
			if ev.Kind == lora.Load {
				loads = append(loads, ev.Adapter)
			}
		}
		if _, err := e.Complete(b.ID, engine.Outcome{}, at+1); err != nil {
			t.Fatal(err)
		}
		return loads
	}
	if got := loadsFor("a", 0); len(got) != 0 {
		t.Errorf("the placed adapter was loaded again for its request: %v", got)
	}
	if got := loadsFor("b", 10); len(got) != 1 {
		t.Errorf("an adapter nothing placed: loads %v, want one", got)
	}
	was, _, err := e.UnloadAdapter("a")
	if err != nil || !was {
		t.Fatalf("unload a: was %v, err %v", was, err)
	}
	for _, n := range e.ResidentAdapterIDs() {
		if n == "a" {
			t.Error("a is still resident after unloading it")
		}
	}
	if was, _, _ := e.UnloadAdapter("never"); was {
		t.Error("unloading an adapter that was never loaded reported it was loaded")
	}

	plain, err := build(t, "")()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := plain.(*Engine).LoadAdapter("a"); err == nil {
		t.Error("an engine without LoRA accepted an adapter load")
	}
}

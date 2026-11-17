package vllm

import (
	"errors"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/engine"

	kvtier "github.com/inference-sim/blis-vllm/core/connector/tiering"
)

// fsTier is TieringOffloadingSpec beside the base block's GPU blocks: a CPU tier of 200 one-block
// chunks of 4 KiB and an fs tier under /kv on a local NVMe device.
const fsTier = "kv_offload: {connector: OffloadingConnector, spec: TieringOffloadingSpec, cpu_bytes: 819200, " +
	"bytes_per_block: 4096, secondary_tiers: [{type: fs, root_dir: /kv, device_class: nvme_gen4}]}\n"

// sharedFiles is one shared file system: every root_dir's files in one map each.
type sharedFiles map[string]kvtier.MemFiles

func (s sharedFiles) storage(root string) TierFiles {
	if s[root] == nil {
		s[root] = kvtier.MemFiles{}
	}
	return s[root]
}

// A prompt one engine stored reaches another engine over shared storage: the writer's CPU tier
// cascades each chunk into the fs tier (a store to the tier's device), and a reader with an empty
// CPU tier and the same root_dir promotes the chunks from the fs tier (a load from the device),
// then loads them from its CPU tier into the GPU, and computes only the rest of the prompt.
func TestAPromptStoredByOneEngineIsLoadedByAnotherOverSharedStorage(t *testing.T) {
	files := sharedFiles{}
	w := mustEngine(t, fsTier)
	w.ShareTierStorage(files.storage)
	a := req("a", 96, 2)
	if err := w.Add(a, 0); err != nil {
		t.Fatal(err)
	}
	_, started, _ := runOne(t, w, 5)
	tierStores := 0
	for _, tr := range started {
		if tr.Tier == "1:fs" {
			if tr.Kind != engine.TransferStore || tr.Move.Tier != "nvme_gen4" || tr.Request != "a" {
				t.Errorf("tier transfer %+v, want a's store to nvme_gen4", tr)
			}
			tierStores++
		}
	}
	if tierStores == 0 || len(files["/kv"]) != 96/16 {
		t.Fatalf("the writer made %d tier stores and %d block files, want a's 6 chunks on the fs tier",
			tierStores, len(files["/kv"]))
	}

	r := mustEngine(t, fsTier)
	r.ShareTierStorage(files.storage)
	b := req("b", 0, 2)
	b.Prompt = append(append([]int32(nil), a.Prompt...), 7, 8, 9, 10)
	if err := r.Add(b, 0); err != nil {
		t.Fatal(err)
	}
	started, first := runTiers(t, r, nil)
	var promoted, loaded int
	for _, tr := range started {
		switch {
		case tr.Tier == "1:fs" && tr.Kind == engine.TransferLoad:
			if tr.Move.Tier != "nvme_gen4" || tr.Request != "b" {
				t.Errorf("promotion %+v, want b's load from nvme_gen4", tr)
			}
			promoted += tr.Tokens
		case tr.Tier == "cpu" && tr.Kind == engine.TransferLoad:
			loaded += tr.Tokens
		}
	}
	if promoted != 96 || loaded != 96 {
		t.Fatalf("the reader promoted %d tokens from the fs tier and loaded %d from its CPU tier, want 96 each", promoted, loaded)
	}
	if first["b"] != 4 {
		t.Fatalf("b's first step computed %d tokens, want the 4 past the shared prefix", first["b"])
	}
}

// A request's kv_transfer_params, carried in its Tags, reach the tiering manager: one whose kv_load_tiers
// allow only the CPU tier does not promote from the shared storage, so it computes its whole prompt; the
// same prompt without the filter promotes the shared prefix and computes only the rest.
func TestARequestsLoadTiersTagKeepsItOffTheSharedStorage(t *testing.T) {
	files := sharedFiles{}
	w := mustEngine(t, fsTier)
	w.ShareTierStorage(files.storage)
	a := req("a", 96, 2)
	if err := w.Add(a, 0); err != nil {
		t.Fatal(err)
	}
	runOne(t, w, 5)
	for name, c := range map[string]struct {
		tag       string
		wantFirst int
	}{
		"no tag":       {"", 4},
		"storage":      {`{"kv_load_tiers":[{"medium":"storage"}]}`, 4},
		"cpu only":     {`{"kv_load_tiers":[{"medium":"cpu"}]}`, 64},
		"no tier":      {`{"kv_load_tiers":[]}`, 64},
		"other key":    {`{"do_remote_decode":true}`, 4},
		"load limit":   {`{"max_load_tokens":80}`, 20},
		"load none":    {`{"max_load_tokens":0}`, 64},
		"float limit":  {`{"max_load_tokens":80.5}`, 4},
		"empty object": {`{}`, 4},
	} {
		t.Run(name, func(t *testing.T) {
			r := mustEngine(t, fsTier)
			r.ShareTierStorage(files.storage)
			b := req("b", 0, 2)
			b.Prompt = append(append([]int32(nil), a.Prompt...), 7, 8, 9, 10)
			if c.tag != "" {
				b.Tags = map[string]string{TagKVTransferParams: c.tag}
			}
			if err := r.Add(b, 0); err != nil {
				t.Fatal(err)
			}
			if _, first := runTiers(t, r, nil); first["b"] != c.wantFirst {
				t.Fatalf("b's first step computed %d tokens, want %d", first["b"], c.wantFirst)
			}
		})
	}
}

// A tag that is not a JSON object is refused; an engine with no kv_offload ignores the tag.
func TestATagThatIsNotAJSONObjectIsRefused(t *testing.T) {
	for name, tag := range map[string]string{"not json": `kv_load_tiers`, "a list": `[1]`, "null": `null`} {
		t.Run(name, func(t *testing.T) {
			r := req("a", 32, 2)
			r.Tags = map[string]string{TagKVTransferParams: tag}
			err := mustEngine(t, fsTier).Add(r, 0)
			if err == nil || !strings.Contains(err.Error(), "JSON object") {
				t.Fatalf("Add = %v, want an error naming a JSON object", err)
			}
		})
	}
	r := req("a", 32, 2)
	r.Tags = map[string]string{TagKVTransferParams: `[1]`}
	if err := mustEngine(t, "").Add(r, 0); err != nil {
		t.Fatalf("an engine with no kv_offload refused a tag it does not read: %v", err)
	}
}

// runTiers is runOne at a transfer time of 5, also returning the tokens each request's first
// scheduled step computed; onStart, if set, sees each transfer as it starts.
func runTiers(t *testing.T, e *Engine, onStart func(engine.Transfer)) ([]engine.Transfer, map[string]int) {
	t.Helper()
	first := map[string]int{}
	var started []engine.Transfer
	type due struct {
		at engine.Time
		id engine.TransferID
	}
	var transfers []due
	now := engine.Time(0)
	for steps := 0; steps < 10000; steps++ {
		for len(transfers) > 0 && transfers[0].at <= now {
			if err := e.TransferDone(transfers[0].id, true, now); err != nil {
				t.Fatal(err)
			}
			transfers = transfers[1:]
		}
		b, err := e.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		for _, tr := range b.TransfersStarted {
			if onStart != nil {
				onStart(tr)
			}
			started = append(started, tr)
			transfers = append(transfers, due{now + 5, tr.ID})
		}
		if b.Empty() {
			if !e.s.HasUnfinished() && len(transfers) == 0 {
				return started, first
			}
			next := now + 1000
			if len(transfers) > 0 {
				next = min(next, transfers[0].at)
			}
			if w, ok := e.NextWake(now); ok {
				next = min(next, w)
			}
			now = max(next, now+1)
			continue
		}
		for _, sc := range b.Reqs {
			if _, seen := first[sc.ID]; !seen {
				first[sc.ID] = sc.Tokens
			}
		}
		now += e.StepTime(b)
		if _, err := e.Complete(b.ID, engine.Outcome{}, now); err != nil {
			t.Fatal(err)
		}
	}
	t.Fatal("no progress")
	return nil, nil
}

// Engines that do not share storage share no block file: the reader's fs tier is empty and the
// whole prompt is computed.
func TestEnginesWithoutSharedStorageShareNoBlockFile(t *testing.T) {
	w := mustEngine(t, fsTier)
	a := req("a", 96, 2)
	if err := w.Add(a, 0); err != nil {
		t.Fatal(err)
	}
	runOne(t, w, 5)
	r := mustEngine(t, fsTier)
	if err := r.Add(a, 0); err != nil {
		t.Fatal(err)
	}
	_, started, _ := runOne(t, r, 5)
	for _, tr := range started {
		if tr.Kind == engine.TransferLoad {
			t.Fatalf("the reader loaded %+v with no storage shared", tr)
		}
	}
}

// tierPricer prices a tier job at its bytes, and records them.
type tierPricer struct {
	FixedPricer
	bytes []int64
}

func (p *tierPricer) TierTransferTime(t engine.Transfer, bytes int64) engine.Duration {
	p.bytes = append(p.bytes, bytes)
	return engine.Duration(bytes / 1000)
}

// A tier job is priced by a TierPricer at its chunks' bytes over every worker's copy; with no
// TierPricer the engine prices it not at all, so the simulator prices its Move.
func TestATierJobIsPricedAtItsChunkBytes(t *testing.T) {
	p := &tierPricer{FixedPricer: FixedPricer(100)}
	s := spec(t, strings.Replace(fsTier, "bytes_per_block: 4096", "bytes_per_block: 4096, workers: 2", 1))
	eng, err := Factory{Pricer: func(engine.Spec) (Pricer, error) { return p, nil }}.New(s)
	if err != nil {
		t.Fatal(err)
	}
	e := eng.(*Engine)
	if err := e.Add(req("a", 32, 2), 0); err != nil {
		t.Fatal(err)
	}
	n := 0
	runTiers(t, e, func(tr engine.Transfer) {
		if tr.Tier != "1:fs" {
			return
		}
		d, ok := e.TransferTime(tr)
		// Each chunk is one 4 KiB block, a copy per worker.
		if want := int64(tr.Blocks) * 2 * 4096; !ok || d != engine.Duration(want/1000) || p.bytes[len(p.bytes)-1] != want {
			t.Errorf("tier job %+v priced %d (%v) at %v bytes, want %d bytes", tr, d, ok, p.bytes, want)
		}
		n++
	})
	if n == 0 {
		t.Fatal("no tier job was started")
	}
	f := mustEngine(t, fsTier)
	if err := f.Add(req("a", 32, 2), 0); err != nil {
		t.Fatal(err)
	}
	runTiers(t, f, func(tr engine.Transfer) {
		if _, ok := f.TransferTime(tr); tr.Tier == "1:fs" && ok {
			t.Errorf("tier job %+v priced by a pricer that prices no tier", tr)
		}
	})
}

// An fs tier engine keeps the engine contract's laws (enginetest), tier jobs included.
func TestFSTierConformance(t *testing.T) {
	var arr []enginetest.Arrival
	at := engine.Time(0)
	for i, id := range []string{"a", "b", "a2", "c", "a3"} {
		r := req(id, 0, 3)
		base, n := 100000*(i+1), 100
		if id[0] == 'a' {
			base, n = 1000, 96
		}
		for k := 0; k < n; k++ {
			r.Prompt = append(r.Prompt, int32(base+k))
		}
		r.Prompt = append(r.Prompt, int32(i))
		arr = append(arr, enginetest.Arrival{At: at, Req: r})
		at += 3000
	}
	tiers := map[string]int{}
	enginetest.TestEngine(t, build(t, fsTier+tightGPU), arr, enginetest.Options{
		Boundary: boundary,
		TransferTime: func(tr engine.Transfer) engine.Duration {
			tiers[tr.Tier]++
			return engine.Duration(50 + 10*tr.Blocks)
		},
	})
	if tiers["1:fs"] == 0 {
		t.Errorf("transfers by tier %v: the fs tier was not exercised", tiers)
	}
}

// An fs tier the engine cannot run, or cannot price, is refused at build, naming the key.
func TestAnFSTierItCannotRunIsRefused(t *testing.T) {
	tier := func(keys string) string {
		return "kv_offload: {connector: OffloadingConnector, spec: TieringOffloadingSpec, cpu_bytes: 819200, " +
			"bytes_per_block: 4096, secondary_tiers: [{type: fs" + keys + "}]}\n"
	}
	for name, c := range map[string]struct{ extra, want string }{
		"no device":       {tier(", root_dir: /kv"), "device_class"},
		"no root":         {tier(", device_class: nvme_gen4"), "root_dir"},
		"unknown key":     {tier(", root_dir: /kv, device_class: nvme_gen4, capacity: 9"), `"capacity"`},
		"no thread":       {tier(", root_dir: /kv, device_class: nvme_gen4, n_read_threads: 0, n_write_threads: 0"), "no thread"},
		"bad locality":    {tier(", root_dir: /kv, device_class: nvme_gen4, locality: local"), "local"},
		"tier pressure":   {tier(", root_dir: /kv, device_class: nvme_gen4, backpressure: {backpressure_cls: EMABackpressureDetector}"), "issues/13"},
		"global pressure": {strings.Replace(tier(", root_dir: /kv, device_class: nvme_gen4"), "secondary_tiers", "backpressure: {backpressure_cls: EMABackpressureDetector}, secondary_tiers", 1), "issues/13"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := build(t, c.extra)()
			if err == nil {
				t.Fatal("built, want refused")
			}
			var nv *CodeNotVisibleError
			if errors.As(err, &nv) || !strings.Contains(err.Error(), "secondary_tiers[0]") || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("refusal %v, want one naming secondary_tiers[0] and %s", err, c.want)
			}
		})
	}
}

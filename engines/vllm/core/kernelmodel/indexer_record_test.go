package kernelmodel

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// openM3Indexer opens the committed MiniMax-M3 fixture with attention_config.indexer_kv_dtype
// dtype ("" leaves it unstated).
func openM3Indexer(t *testing.T, dtype string) (*Model, error) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "kerneltests", "testdata", "aisimulate", "minimax-m3-b200-fp4-vllm-tp2.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	if dtype != "" {
		body = strings.Replace(body, "      block_size:", "      attention_config: {indexer_kv_dtype: "+dtype+"}\n      block_size:", 1)
		if !strings.Contains(body, "indexer_kv_dtype") {
			t.Fatal("the fixture has no block_size line to put the key before")
		}
	}
	r := repos()
	r.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return Open("v.yaml", r)
}

func indexerElemBytes(t *testing.T, m *Model) (n, elem int) {
	t.Helper()
	runs, err := m.KVLayers()
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range runs {
		if r.Spec == "indexer" {
			n += r.Count
			elem = r.ElemBytes
		}
	}
	return n, elem
}

// vLLM stores MiniMax-M3's indexer keys as torch.float8_e4m3fn with no scale when
// attention_config.indexer_kv_dtype is fp8 (common/indexer.py:118-121, :145-154@5df63a72), one byte
// a value where bf16 takes two. The deployment opens, the pool sizes the indexer's keys at one byte,
// and the latency kernel prices the cache at the same record: the page the pool draws is smaller, the
// graph the kernel reads is the one stated, and a long-context decode is cheaper than at bf16.
func TestMiniMaxM3FP8IndexerKeysAreHalfTheBytes(t *testing.T) {
	bf16, err := openM3Indexer(t, "")
	if err != nil {
		t.Fatal(err)
	}
	nb, eb := indexerElemBytes(t, bf16)
	if nb == 0 || eb != 2 {
		t.Fatalf("unstated: %d indexer layers of %d bytes a value, want bf16's 2", nb, eb)
	}
	for _, dtype := range []string{"auto", "bf16", "fp8", "fp8_e4m3"} {
		m, err := openM3Indexer(t, dtype)
		if err != nil {
			t.Errorf("indexer_kv_dtype %s: %v", dtype, err)
			continue
		}
		n, e := indexerElemBytes(t, m)
		want := 2
		if strings.HasPrefix(dtype, "fp8") {
			want = 1
		}
		if n != nb || e != want {
			t.Errorf("indexer_kv_dtype %s: %d indexer layers of %d bytes a value, want %d layers of %d", dtype, n, e, nb, want)
		}
		if want == 1 {
			long := []ReqShape{{Scheduled: 1, Computed: 150_000, PromptLen: 100_000}}
			if a, b := m.StepTimeOfShapes(long), bf16.StepTimeOfShapes(long); a >= b {
				t.Errorf("indexer_kv_dtype %s prices a 150k-token decode at %d us, bf16 at %d us: the half-size keys must read less", dtype, a, b)
			}
		}
	}
}

// The lowering returns a graph with the half-size record and leaves the graph it was given as it
// was, so a graph shared by several deployments of one checkpoint keeps the default record. A dtype
// that is not fp8 hands back the graph itself.
func TestLoweringTheIndexerRecordLeavesTheGivenGraphAlone(t *testing.T) {
	m, err := openM3Indexer(t, "")
	if err != nil {
		t.Fatal(err)
	}
	indexEntries := func(g *model.Graph) (entries []int) {
		for _, k := range g.LayerKinds {
			for _, n := range k.Nodes {
				for _, c := range n.Caches {
					if c.Kind == model.CacheIndex {
						entries = append(entries, c.BytesPerEntry)
					}
				}
			}
		}
		return entries
	}
	before := indexEntries(m.graph)
	if len(before) == 0 || before[0] != 256 {
		t.Fatalf("the catalog graph's index entries are %v, want bf16's 256", before)
	}
	low := lowerIndexerKV(m.graph, "fp8")
	if got := indexEntries(low); len(got) != len(before) || got[0] != 128 {
		t.Errorf("lowered index entries %v, want 128 each", got)
	}
	if got := indexEntries(m.graph); !reflect.DeepEqual(got, before) {
		t.Errorf("lowering edited the graph it was given: %v, was %v", got, before)
	}
	for _, d := range []string{"", "auto", "bf16"} {
		if lowerIndexerKV(m.graph, d) != m.graph {
			t.Errorf("indexer_kv_dtype %q changed the graph", d)
		}
	}
}

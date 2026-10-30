package vllmengine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"
)

// kvOffload is the kv_offload the factory states in a prepared engine block.
type kvOffload struct {
	Connector      string `yaml:"connector"`
	CPUBytes       int64  `yaml:"cpu_bytes"`
	BytesPerBlock  int64  `yaml:"bytes_per_block"`
	Workers        int    `yaml:"workers"`
	Copies         int    `yaml:"copies"`
	EvictionPolicy string `yaml:"eviction_policy"`
	BlockSize      int    `yaml:"block_size"`
}

func preparedOffload(t *testing.T, spec engine.Spec) (kvOffload, int64) {
	t.Helper()
	done, err := prepare(spec)
	if err != nil {
		t.Fatal(err)
	}
	var b struct {
		O      *kvOffload `yaml:"kv_offload"`
		Blocks int64      `yaml:"num_gpu_blocks"`
	}
	if err := done.Config.Decode(&b); err != nil {
		t.Fatal(err)
	}
	if b.O == nil {
		t.Fatal("the prepared block states no kv_offload")
	}
	return *b.O, b.Blocks
}

// Muse Spark's arm D, PP2 x TP8 over two H100 nodes with DSpark, fp8 KV, async scheduling and the
// OffloadingConnector's CPU tier, maps the deployment's offload block onto the engine block: the
// tier's bytes as cpu_bytes_to_use, LRU, and what the kernel states of the engine (16 workers each
// holding a copy, a worker's bytes a block). It is refused, naming blis-vllm#14, until the batch
// queue and the drafter are diff-tested beside the connector. Its PP1 sync twin on one node maps
// the same way and runs: requests sharing a prefix, the GPU's prefix cache forgetting it, the CPU
// tier serving it back.
func TestMuseSparkArmDOffload(t *testing.T) {
	Register()
	armD := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, "muse-h100-roce-armD.yaml")}
	o, _ := preparedOffload(t, armD)
	if o.Connector != "OffloadingConnector" || o.CPUBytes != 1<<40 || o.Workers != 16 || o.Copies != 16 ||
		o.EvictionPolicy != "lru" || o.BytesPerBlock <= 0 {
		t.Fatalf("arm D's kv_offload %+v, want OffloadingConnector over 1 TiB, lru, 16 workers and copies", o)
	}
	_, err := engine.New(armD)
	if err == nil || !strings.Contains(err.Error(), "issues/14") {
		t.Fatalf("arm D built (%v): want the batch queue beside the connector refused with blis-vllm#14", err)
	}

	twin := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, "muse-h100-roce-armD-pp1-sync.yaml")}
	o, blocks := preparedOffload(t, twin)
	if o.CPUBytes != 512<<30 || o.Workers != 8 || o.Copies != 8 {
		t.Fatalf("the twin's kv_offload %+v, want 512 GiB over 8 workers' copies", o)
	}
	// The CPU tier holds 512 GiB over 8 copies of a block: far more blocks than the GPU pool.
	if chunks := o.CPUBytes / (o.BytesPerBlock * int64(o.Copies)); chunks <= blocks {
		t.Fatalf("the CPU tier holds %d blocks, the GPU %d: want the tier larger", chunks, blocks)
	}
	// a's 6,144-token prompt, then eight prompts of new tokens (more than the GPU pool's 2,653
	// blocks hold beside it), then b with a's prefix, which the CPU tier serves back.
	prefix := make([]int32, 6144)
	for k := range prefix {
		prefix[k] = int32(1_000_000 + k)
	}
	var arrivals []enginetest.Arrival
	ids := []string{"a", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "b"}
	for i, id := range ids {
		r := req(id, 0, 4)
		if id == "a" || id == "b" {
			r.Prompt = append(append([]int32(nil), prefix...), int32(i))
		} else {
			for k := 0; k < 6144; k++ {
				r.Prompt = append(r.Prompt, int32(2_000_000+10_000*i+k))
			}
		}
		arrivals = append(arrivals, enginetest.Arrival{At: engine.Time(i) * 2_000_000, Req: r})
	}
	kinds := map[engine.TransferKind]int{}
	tr, err := enginetest.Drive(func() (engine.Engine, error) { return engine.New(twin) }, arrivals, enginetest.Options{
		TransferTime: func(x engine.Transfer) engine.Duration {
			if x.Tier != "cpu" {
				t.Errorf("transfer %+v names tier %q, want cpu", x, x.Tier)
			}
			kinds[x.Kind]++
			return 1000
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(tr.Finished) != len(ids) {
		t.Fatalf("finished %d of %d requests", len(tr.Finished), len(ids))
	}
	if kinds[engine.TransferStore] == 0 || kinds[engine.TransferLoad] == 0 {
		t.Fatalf("transfers %v: want prompts stored to the CPU tier and b's prefix loaded back", kinds)
	}
	t.Logf("the twin's transfers: %v", kinds)
}

// The mapping refuses what the engine does not model, each naming its issue: a tier other than
// host DRAM or below it (secondary tiers, #13), another offload connector (#14), and connector
// options other than blocks_per_chunk.
func TestOffloadMappingRefusals(t *testing.T) {
	Register()
	raw, err := os.ReadFile(filepath.Join("testdata", "muse-h100-roce-armD-pp1-sync.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	const tier = "  tiers:\n    - {tier: cpu_dram, bytes: 549755813888}\n"
	for _, c := range []struct{ name, old, new, want string }{
		{"a secondary tier", tier, tier + "    - {tier: nvme_gen4, bytes: 1099511627776}\n", "issues/13"},
		{"a tier not of host DRAM", tier, "  tiers:\n    - {tier: nvme_gen4, bytes: 1099511627776}\n", "issues/13"},
		{"another connector", "  connector: OffloadingConnector\n", "  connector: LMCacheConnectorV1\n", "issues/14"},
		{"a Mooncake option", "  eviction_policy: lru\n", "  eviction_policy: lru\n  connector_options: {load_async: true}\n", "blocks_per_chunk"},
	} {
		t.Run(c.name, func(t *testing.T) {
			s := string(raw)
			if !strings.Contains(s, c.old) {
				t.Fatalf("the fixture has no %q", c.old)
			}
			dir := t.TempDir()
			path := filepath.Join(dir, "s.yaml")
			if err := os.WriteFile(path, []byte(strings.Replace(s, c.old, c.new, 1)), 0o644); err != nil {
				t.Fatal(err)
			}
			r := roots(t, "muse-h100-roce-armD-pp1-sync.yaml")
			r[RootScenario] = path
			_, err := prepare(engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: r})
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("err %v, want a refusal naming %q", err, c.want)
			}
		})
	}
}

// blocks_per_chunk in the connector options is the CPU chunk's GPU blocks, stated to the engine as
// the chunk's tokens (kv_connector_extra_config block_size, offloading/config.py:98-134@5df63a72).
func TestOffloadMappingBlocksPerChunk(t *testing.T) {
	Register()
	raw, err := os.ReadFile(filepath.Join("testdata", "muse-h100-roce-armD-pp1-sync.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	s := strings.Replace(string(raw), "  eviction_policy: lru\n", "  eviction_policy: lru\n  connector_options: {blocks_per_chunk: 4}\n", 1)
	path := filepath.Join(t.TempDir(), "s.yaml")
	if err := os.WriteFile(path, []byte(s), 0o644); err != nil {
		t.Fatal(err)
	}
	r := roots(t, "muse-h100-roce-armD-pp1-sync.yaml")
	r[RootScenario] = path
	o, _ := preparedOffload(t, engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: r})
	if o.BlockSize != 64 {
		t.Fatalf("kv_offload block_size %d, want 4 blocks of 16 tokens", o.BlockSize)
	}
}

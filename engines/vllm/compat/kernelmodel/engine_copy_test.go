// Original to blis-vllm.

package kernelmodel

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-vllm/compat/vllmsettings"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// engineCopyScenarioDirs are the scenario directories the copy is checked over: the kernel's
// vendored scenarios, this repository's, and the dashboard deployments when the workspace that holds them is set
// (BLIS_DASHBOARD_SCENARIOS, a colon-separated list).
func engineCopyScenarioDirs() []string {
	// This repository's own scenarios too: they hold the disaggregated ones, whose pools differ.
	dirs := []string{DefaultScenarios(), filepath.Join("..", "..", "testdata", "scenarios")}
	for _, d := range strings.Split(os.Getenv("BLIS_DASHBOARD_SCENARIOS"), ":") {
		if d != "" {
			dirs = append(dirs, d)
		}
	}
	return dirs
}

// The engine block this package decodes once at open (Model.engine) is the one the kernel was
// built from: field for field the block the kernel's deployment holds, for every pool of every
// scenario that opens. The kernel prices the settings converted from that copy, so the two
// cannot describe different engines, and the simulator's settings (Engine, Settings, the KV
// budget) read what they read before the kernel stopped carrying the block.
func TestTheEngineCopyIsTheKernelsDeploymentEngine(t *testing.T) {
	opened := 0
	for _, dir := range engineCopyScenarioDirs() {
		files, _ := filepath.Glob(filepath.Join(dir, "*.yaml"))
		for _, path := range files {
			r := DefaultRepos()
			r.Scenarios = dir
			for pool := 0; pool < 4; pool++ {
				m, err := OpenPool(filepath.Base(path), r, pool)
				if err != nil {
					break // past the last pool, or a scenario the kernel refuses
				}
				opened++
				// The pool's own block, read from the file: the copy is this pool's, not
				// another pool's that the kernel was handed too.
				_, dep, err := latencykernel.LoadBundle(path)
				if err != nil {
					t.Fatal(err)
				}
				file, err := engineconfig.Decode(mustBlock(t, dep.Pools[pool]))
				if err != nil {
					t.Fatalf("%s pool %d: %v", path, pool, err)
				}
				if !reflect.DeepEqual(m.engine, file) {
					t.Errorf("%s pool %d: the engine copy is not the pool's engine block", path, pool)
				}
				if s := m.k.(*latencykernel.Kernel).Engine(); !reflect.DeepEqual(s, vllmsettings.Of(m.engine)) {
					t.Errorf("%s pool %d: the kernel prices settings other than the copy's", path, pool)
				}
				// The block the kernel's deployment carried is the same block, read the same way.
				kernelBlock, err := engineconfig.Decode(mustBlock(t, m.k.Deployment()))
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(m.engine, kernelBlock) {
					t.Errorf("%s pool %d: the engine copy differs from the kernel's deployment engine:\n"+
						"  copy   %+v\n  kernel %+v", path, pool, m.engine, kernelBlock)
				}
				got, err := m.Engine()
				if err == nil && !reflect.DeepEqual(got, kernelBlock) {
					t.Errorf("%s pool %d: Engine() is not the kernel's deployment engine", path, pool)
				}
			}
		}
	}
	if opened < 50 {
		t.Fatalf("non-vacuity: only %d pools opened", opened)
	}
	t.Logf("%d pools compared", opened)
}

// A disaggregated deployment whose pools state different engine blocks opens each pool with its
// own: the committed P/D scenarios state the same block in both pools, so this one varies the
// decode pool's token budget and sequence cap.
func TestEachPoolsEngineCopyIsItsOwn(t *testing.T) {
	const src = "minimax-m2.5-gb200-1p1d-dp2.yaml"
	body, err := os.ReadFile(filepath.Join("..", "..", "testdata", "scenarios", src))
	if err != nil {
		t.Fatal(err)
	}
	const decode = "  - role: decode\n"
	i := strings.Index(string(body), decode)
	if i < 0 {
		t.Fatalf("%s has no decode pool to vary", src)
	}
	varied := string(body[:i]) + strings.Replace(strings.Replace(string(body[i:]),
		"max_num_batched_tokens: 8192", "max_num_batched_tokens: 4096", 1),
		"max_num_seqs: 256", "max_num_seqs: 128", 1)
	dir := t.TempDir()
	for _, name := range []string{"router"} {
		if err := os.Symlink(filepath.Join(must(filepath.Abs(filepath.Join("..", "..", "testdata", "scenarios"))), name),
			filepath.Join(dir, name)); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "varied.yaml"), []byte(varied), 0o644); err != nil {
		t.Fatal(err)
	}
	r := DefaultRepos()
	r.Scenarios = dir
	for pool, want := range []struct{ tokens, seqs int }{{8192, 256}, {4096, 128}} {
		m, err := OpenPool("varied.yaml", r, pool)
		if err != nil {
			t.Fatalf("pool %d: %v", pool, err)
		}
		e, err := m.Engine()
		if err != nil {
			t.Fatal(err)
		}
		if e.MaxNumBatchedTokens != want.tokens || e.MaxNumSeqs != want.seqs {
			t.Errorf("pool %d: engine copy states max_num_batched_tokens %d and max_num_seqs %d, want its "+
				"own %d and %d", pool, e.MaxNumBatchedTokens, e.MaxNumSeqs, want.tokens, want.seqs)
		}
		if s := m.k.(*latencykernel.Kernel).Engine(); s.MaxNumBatchedTokens != want.tokens {
			t.Errorf("pool %d: the kernel prices max_num_batched_tokens %d, want %d", pool,
				s.MaxNumBatchedTokens, want.tokens)
		}
	}
}

func must(s string, err error) string {
	if err != nil {
		panic(err)
	}
	return s
}

// mustBlock is a pool's engine block.
func mustBlock(t testing.TB, pool deployment.Pool) deployment.EngineBlock {
	t.Helper()
	return pool.Engine
}

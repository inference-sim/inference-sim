// Moved from blis-latency-kernel/loader_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/vllmsettings"
	"github.com/inference-sim/blis-vllm/kerneltests/internal/artifacttest"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/hardware"
	vllmrules "github.com/inference-sim/blis-vllm/rules"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
	"github.com/inference-sim/blis-vllm/rules/pack"
	"github.com/inference-sim/blis-vllm/rules/v0_29"
)

// The catalog and registry are pinned copies under testdata, fetched by
// scripts/fetch-testdata.sh at the commits testdata/upstream.lock names, so these tests read
// real artifacts at a known version. See testdata/README.md for what is fetched and why.
//
// They are relative paths because a Go test runs with its own package directory as the
// working directory, and this package is the repository root. Overridable so a working
// copy can be scored against a live upstream checkout:
//
//	BLIS_CATALOG=../blis-catalog go test ./...
var (
	catalogRoot  = envOr("BLIS_CATALOG", filepath.Join("testdata", "catalog"))
	registryRoot = envOr("BLIS_REGISTRY", filepath.Join("testdata", "registry"))
)

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// fixture builds a kernel from a committed scenario and the real catalog and registry.
//
// Deliberately not a hand-built Inputs struct. A kernel assembled from literals in a
// test proves the arithmetic and nothing about whether the committed artifacts can
// drive it, and the second is where the failures have been.
//
// A MISSING sibling checkout skips, because not every working copy has one. A checkout
// that is PRESENT but unreadable fails, because that is a real incompatibility and
// skipping it is how one hid: under the pseudo-version this repo was pinned to,
// blis-catalog's storage.yaml failed to decode (it already carried v0.2.0's field names),
// so every test here skipped on "catalog unavailable" and 41 of them were passing by not
// running — measured by `go test -v .` at 3cf1ef1: 41 top-level skips, 46 counting
// subtests, 0 failures, suite reporting ok. artifacttest.RequireArtifact draws that line.
func fixtureInputs(t testing.TB, scenario string) Inputs {
	t.Helper()
	// The fixture itself is committed beside this test, so it is never merely absent:
	// a failure to read one is a malformed fixture and always a real failure.
	sc, dep, err := LoadBundle(filepath.Join("testdata", scenario))
	if err != nil {
		t.Fatalf("committed fixture %s: %v", scenario, err)
	}
	graphPath := filepath.Join(catalogRoot, "models", sc.Model, "graph.yaml")
	graph, err := schemas.LoadModelGraph(graphPath)
	artifacttest.RequireArtifact(t, catalogRoot, graphPath, "catalog", err)
	chipPath := filepath.Join(catalogRoot, "hardware", sc.Cluster.Hardware+".yaml")
	chip, err := schemas.LoadChip(chipPath)
	artifacttest.RequireArtifact(t, catalogRoot, chipPath, "catalog", err)
	var fabric *hardware.Fabric
	if sc.Cluster.Fabric != "" {
		fabricPath := filepath.Join(catalogRoot, "networks", sc.Cluster.Fabric+".yaml")
		fabric, err = schemas.LoadFabric(fabricPath)
		artifacttest.RequireArtifact(t, catalogRoot, fabricPath, "catalog", err)
	}
	var sets []*coefficient.Set
	for _, name := range sc.Coefficients {
		setPath := filepath.Join(registryRoot, "coefficients", name+".yaml")
		set, err := schemas.LoadCoefficientSet(setPath)
		artifacttest.RequireArtifact(t, registryRoot, setPath, "registry", err)
		sets = append(sets, set)
	}
	devicesPath := filepath.Join(catalogRoot, "devices", "storage.yaml")
	devices, err := schemas.LoadStorageDevices(devicesPath)
	artifacttest.RequireArtifact(t, catalogRoot, devicesPath, "catalog", err)
	eng, err := vllmsettings.OfPool(dep, 0)
	if err != nil {
		t.Fatalf("committed fixture %s: decoding its engine block: %v", scenario, err)
	}
	return Inputs{
		Scenario: sc, Deployment: dep, PoolIndex: 0,
		Model: graph, Chip: chip, Fabric: fabric,
		Devices: devices, Coefficients: sets, Rules: rulesFor(sc.EngineVersion), Engine: &eng,
	}
}

// rulesFor is the vLLM rules pack for a fixture's declared release, and the 0.29.0 pack for
// a release with none, as these fixtures were written against. The packs come from the
// engine model's rules module, as every caller's do: the kernel looks none up itself.
func rulesFor(version string) *pack.Pack {
	if p := vllmrules.Lookup(version); p != nil {
		return p
	}
	return v0_29.Pack()
}

// withRules gives a Repos the vLLM rules packs and engine-block decoding, as a caller of Open
// does.
func withRules(r Repos) Repos {
	r.Rules = RulesFrom(vllmrules.Lookup)
	r.RulesVersions = vllmrules.Versions
	r.EngineSettings = vllmsettings.OfPool
	return r
}

// withModel replaces a fixture's model graph with the catalog's graph for name, so one
// deployment can price several models.
func withModel(t testing.TB, in *Inputs, name string) {
	t.Helper()
	graphPath := filepath.Join(catalogRoot, "models", name, "graph.yaml")
	graph, err := schemas.LoadModelGraph(graphPath)
	artifacttest.RequireArtifact(t, catalogRoot, graphPath, "catalog", err)
	in.Scenario.Model = name
	in.Model = graph
}

// newKernel is New on inputs a test has edited, with the engine settings decoded from the edited
// pool, as a caller decodes the block it hands the kernel (vllmsettings): a test that rewrites
// in.Deployment.Pools[i].Engine prices what it wrote.
func newKernel(in Inputs) (*Kernel, error) {
	if in.Deployment != nil && in.PoolIndex >= 0 && in.PoolIndex < len(in.Deployment.Pools) {
		eng, err := vllmsettings.OfPool(in.Deployment, in.PoolIndex)
		if err != nil {
			return nil, err
		}
		in.Engine = &eng
	}
	return New(in)
}

// build is New on inputs a test has edited, failing the test on a refusal.
func build(t testing.TB, in Inputs) *Kernel {
	t.Helper()
	k, err := newKernel(in)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return k
}

// fixture builds a kernel from the Inputs a committed scenario implies.
//
// Split from fixtureInputs so a test that needs to PERTURB one field before construction --
// to watch New refuse a deployment that does not fit its cluster, say -- can take the
// documents and assemble them itself, rather than reimplementing the loading.
func fixture(t testing.TB, scenario string) *Kernel {
	t.Helper()
	k, err := newKernel(fixtureInputs(t, scenario))
	if err != nil {
		t.Fatalf("New from committed artifacts: %v", err)
	}
	return k
}

// chipInputs loads one catalog chip.
func chipInputs(t testing.TB, name string) *hardware.Chip {
	t.Helper()
	p := filepath.Join(catalogRoot, "hardware", name+".yaml")
	c, err := schemas.LoadChip(p)
	artifacttest.RequireArtifact(t, catalogRoot, p, "catalog", err)
	return c
}

// editEngine changes the priced pool's engine block the way a caller changes a scenario: decoded
// as vLLM's engine configuration (vllmsettings.Decode), edited, and encoded back
// (engineconfig.Encode). The block is held unopened, so this is the only way a test edits it.
func editEngine(t testing.TB, in *Inputs, edit func(*engineconfig.Engine)) {
	t.Helper()
	editPoolEngine(t, &in.Deployment.Pools[in.PoolIndex], edit)
}

// editPoolEngine is editEngine for any pool.
func editPoolEngine(t testing.TB, pool *deployment.Pool, edit func(*engineconfig.Engine)) {
	t.Helper()
	e, err := vllmsettings.Decode(*pool)
	if err != nil {
		t.Fatal(err)
	}
	edit(&e)
	b, err := engineconfig.Encode(e)
	if err != nil {
		t.Fatal(err)
	}
	pool.Engine = b
}

// poolEngine is the priced pool's engine block, decoded, for a test that reads a stated setting.
func poolEngine(t testing.TB, in Inputs) engineconfig.Engine {
	t.Helper()
	e, err := vllmsettings.Decode(in.Deployment.Pools[in.PoolIndex])
	if err != nil {
		t.Fatal(err)
	}
	return e
}

// engineBlock encodes e as an engine block, for a test that builds a pool from scratch.
func engineBlock(t testing.TB, e engineconfig.Engine) deployment.EngineBlock {
	t.Helper()
	b, err := engineconfig.Encode(e)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// mustDecode is a pool's engine block decoded, for a test that edits several settings and then
// writes them back with setEngine.
func mustDecode(t testing.TB, pool deployment.Pool) engineconfig.Engine {
	t.Helper()
	e, err := vllmsettings.Decode(pool)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

// setEngine writes e as the pool's engine block.
func setEngine(t testing.TB, pool *deployment.Pool, e engineconfig.Engine) {
	t.Helper()
	pool.Engine = engineBlock(t, e)
}

// editInputsEngine is editEngine for an Inputs edit applied without a test handle (a func(*Inputs)
// a table passes to a builder); a block that does not round-trip panics, which fails the test.
func editInputsEngine(in *Inputs, edit func(*engineconfig.Engine)) {
	pool := &in.Deployment.Pools[in.PoolIndex]
	e, err := vllmsettings.Decode(*pool)
	if err != nil {
		panic(err)
	}
	edit(&e)
	b, err := engineconfig.Encode(e)
	if err != nil {
		panic(err)
	}
	pool.Engine = b
}

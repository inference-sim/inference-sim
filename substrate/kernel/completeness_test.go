package latencykernel

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/load"
)

// engineResolution names the assumptions that record an engine decision the cost model makes
// today, not a missing coefficient. The boundary transition moves them to the engine
// (SUB-6); until then they are tracked apart from coefficient fallbacks.
var engineResolution = map[string]bool{
	"cudagraph_mode": true, "engine_version": true, "mamba_cache_mode": true,
	"max_cudagraph_capture_size": true, "modality": true,
}

// pricedEverywhere builds every catalog model on every catalog chip at one standard layout and
// returns, for each value the kernel supplied itself because the registry carries no entry for
// it, the chips it did so on (design SUB-7, VAL-2, VAL-8). A name is the coefficient or
// decision, and the first words of the kernel's own reason say what it fell back to.
func pricedEverywhere(t *testing.T) (built int, coefficient, engine map[string]map[string]bool, refused map[string]int) {
	t.Helper()
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	sets := "[cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory, cost-model-fabric]"
	coefficient, engine, refused = map[string]map[string]bool{}, map[string]map[string]bool{}, map[string]int{}
	for _, m := range d.Models() {
		for _, c := range d.Chips() {
			packaging := "  gpus_per_node: 8\n"
			if ch, err := d.Chip(c); err == nil && ch.GPUsPerNode > 0 {
				packaging = "" // the catalog entry fixes the packaging, and a restatement is refused
			}
			body := fmt.Sprintf("kind: Scenario\nname: p\nmodel: %s\ncoefficients: %s\nengine_version: \"0.32.0\"\ncluster:\n  hardware: %s\n  fabric: ib-400g\n  nodes: 1\n%s---\nkind: Deployment\nname: d\npools:\n  - role: colocated\n    nodes: 1\n    parallel: {tp: 8, pp: 1, dp: 1}\n", m, sets, c, packaging)
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "s.yaml"), []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
			r := Repos{Scenarios: dir, Data: d,
				Rules: func(string) rules.EngineRules { return inertRules{} },
				EngineSettings: func(*deployment.Deployment, int) (enginesettings.Settings, error) {
					return enginesettings.Settings{BlockSize: 16, MaxNumSeqs: 64, MaxNumBatchedTokens: 8192, MaxModelLen: 8192}, nil
				}}
			in, err := OpenInputs("s.yaml", r, 0)
			var k *Kernel
			if err == nil {
				k, err = New(in)
			}
			if err != nil {
				refused[firstWords(err.Error())]++
				continue
			}
			built++
			for _, o := range k.Provenance() {
				if o.Set != KernelAssumptionSet {
					continue
				}
				into := coefficient
				if engineResolution[o.Name] {
					into = engine
				}
				if into[o.Name] == nil {
					into[o.Name] = map[string]bool{}
				}
				into[o.Name][c] = true
			}
		}
	}
	return built, coefficient, engine, refused
}

// stripComments drops the comment lines of a testdata list.
func stripComments(s string) string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		if !strings.HasPrefix(l, "#") {
			out = append(out, l)
		}
	}
	return strings.Join(out, "\n")
}

func firstWords(s string) string {
	if i := strings.IndexAny(s, ":\n"); i > 0 {
		s = s[:i]
	}
	return s
}

func render(m map[string]map[string]bool) string {
	var lines []string
	for name, chips := range m {
		var cs []string
		for c := range chips {
			cs = append(cs, c)
		}
		sort.Strings(cs)
		lines = append(lines, name+" "+strings.Join(cs, ","))
	}
	sort.Strings(lines)
	return strings.Join(lines, "\n") + "\n"
}

// SUB-7 and VAL-2: where the registry carries no entry, the kernel supplies a value itself and
// records that. Each such value is a gap in the registry, and the list of them only shrinks:
// the test fails on a gap not in testdata and on a listed gap that no longer exists, so a fix
// must delete its line and a new fallback cannot arrive unnoticed. Regenerate with
// UPDATE_COMPLETENESS=1 after deciding the change is right.
func TestEveryValueTheKernelSuppliesItselfIsListed(t *testing.T) {
	built, coefficient, engine, _ := pricedEverywhere(t)
	if built < 200 {
		t.Fatalf("only %d model/chip pairs built; the probe is not reaching the kernel", built)
	}
	for file, got := range map[string]string{
		"unstored_coefficients.txt": render(coefficient),
		"engine_resolution.txt":     render(engine),
	} {
		path := filepath.Join("testdata", file)
		if os.Getenv("UPDATE_COMPLETENESS") != "" {
			// Keep the file's comment header; replace only the list under it.
			old, _ := os.ReadFile(path)
			var head []string
			for _, l := range strings.Split(string(old), "\n") {
				if strings.HasPrefix(l, "#") {
					head = append(head, l)
				}
			}
			out := got
			if len(head) > 0 {
				out = strings.Join(head, "\n") + "\n" + got
			}
			if err := os.WriteFile(path, []byte(out), 0o644); err != nil {
				t.Fatal(err)
			}
			continue
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		want := stripComments(string(raw))
		if want != got {
			t.Errorf("%s differs from what the kernel supplies itself.\nwant:\n%s\ngot:\n%s", file, want, got)
		}
	}
}

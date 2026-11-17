package registry

import (
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// VAL-8's evidence ratchet on kernel_impls: a measured or calibrated entry records the kernel
// implementations it is valid for. Only an entry whose name binds a lowered primitive can state
// one: the validator refuses kernel_impls on any other (contracts/spec/coefficient/validate.go,
// "prices no lowered kernel"), so a memory or host-overhead term could never meet the ratchet.
// "Binds a primitive" is coefficient.PrimitiveOf, the function the validator uses.
//
// The count may only fall. Lower it when a fitter starts writing the field; never raise it. The
// measures, fit_range and ci95 ratchets stay in registry/validator/test_coefficient_properties.py.
const maxFittedWithoutKernelImpls = 870

type fittedEntry struct {
	set   string
	entry coefficient.Entry
}

func fittedEntries(t *testing.T) []fittedEntry {
	t.Helper()
	paths, err := SetPaths(filepath.Join("..", "..", "registry"))
	if err != nil {
		t.Fatal(err)
	}
	var out []fittedEntry
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		var s coefficient.Set
		if err := yaml.Unmarshal(raw, &s); err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		for _, e := range s.Coefficients {
			if e.Method == vocab.MethodMeasured || e.Method == vocab.MethodCalibrated {
				out = append(out, fittedEntry{filepath.Base(p), e})
			}
		}
	}
	return out
}

// withoutKernelImpls is the fitted entries that bind a primitive and state no kernel_impls.
func withoutKernelImpls(es []fittedEntry) []fittedEntry {
	var out []fittedEntry
	for _, e := range es {
		if coefficient.PrimitiveOf(e.entry.Name) != "" && len(e.entry.KernelImpls) == 0 {
			out = append(out, e)
		}
	}
	return out
}

func TestFittedEntriesWithoutKernelImplsOnlyGoDown(t *testing.T) {
	missing := withoutKernelImpls(fittedEntries(t))
	if len(missing) > maxFittedWithoutKernelImpls {
		t.Errorf("%d measured or calibrated entries that bind a primitive state no kernel_impls, above the "+
			"%d committed; a new fitted entry records them", len(missing), maxFittedWithoutKernelImpls)
	}
	if len(missing) < maxFittedWithoutKernelImpls {
		t.Errorf("%d fitted primitive-binding entries state no kernel_impls, below the %d committed: lower "+
			"maxFittedWithoutKernelImpls to match", len(missing), maxFittedWithoutKernelImpls)
	}
}

// The ratchet still counts what it is for: a fitted entry that binds a primitive and states no
// kernel_impls raises the count, and one that binds none does not.
func TestTheKernelImplsRatchetCountsAPrimitiveBindingEntry(t *testing.T) {
	es := fittedEntries(t)
	base := len(withoutKernelImpls(es))
	gemm := coefficient.Entry{Name: "gemm_eps_max_bf16", Method: vocab.MethodMeasured}
	if coefficient.PrimitiveOf(gemm.Name) == "" {
		t.Fatalf("%s binds no primitive; the case tests nothing", gemm.Name)
	}
	if got := len(withoutKernelImpls(append(es, fittedEntry{"s.yaml", gemm}))); got != base+1 {
		t.Errorf("a fitted %s with no kernel_impls left the count at %d, want %d", gemm.Name, got, base+1)
	}
	memory := coefficient.Entry{Name: "nontorch_in_budget_bytes_4rank", Method: vocab.MethodMeasured}
	if got := len(withoutKernelImpls(append(es, fittedEntry{"s.yaml", memory}))); got != base {
		t.Errorf("a fitted %s, which binds no primitive, moved the count to %d from %d", memory.Name, got, base)
	}
}

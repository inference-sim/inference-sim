// Moved from inference-sim/blis-schemas/rules/condition_test.go at 70246460a379f8b1898fbf6c36a3091364118950.

package pack

import (
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// ModelSequenceParallel evaluates a condition as data, so its meaning is checked over every
// shape a condition can take, not only the one a registered release uses: each
// requirement, when set, is necessary; when unset, it does not matter; and the
// mega-MoE-or-dp requirement is met by either. Unset widths count as one.
func TestModelSequenceParallelEvaluatesEveryConditionShape(t *testing.T) {
	const arch = "Arch"
	mega := "mega"
	for shape := 0; shape < 16; shape++ {
		c := SequenceParallelCondition{
			PipelineParallelOne: shape&1 != 0, ExpertParallel: shape&2 != 0,
			TensorParallelAboveOne: shape&4 != 0, MegaMoEOrDataParallelAboveOne: shape&8 != 0,
		}
		p := &Pack{ModelSequenceParallelRules: map[string]SequenceParallelCondition{arch: c},
			MegaMoEBackends: map[string]bool{mega: true}}
		for _, tp := range []int{0, 1, 4} {
			for _, pp := range []int{0, 1, 2} {
				for _, dp := range []int{0, 1, 2} {
					for _, ep := range []bool{false, true} {
						for _, moe := range []string{"", mega} {
							pl := deployment.Parallelism{TP: tp, PP: pp, DP: dp, EnableExpertParallel: ep}
							want := true
							if c.PipelineParallelOne && pp > 1 {
								want = false
							}
							if c.ExpertParallel && !ep {
								want = false
							}
							if c.TensorParallelAboveOne && tp <= 1 {
								want = false
							}
							if c.MegaMoEOrDataParallelAboveOne && moe != mega && dp <= 1 {
								want = false
							}
							if got := p.ModelSequenceParallel(arch, pl, moe); got != want {
								t.Fatalf("condition %+v, %+v, moe %q: got %v, want %v", c, pl, moe, got, want)
							}
							if p.ModelSequenceParallel("Other", pl, moe) {
								t.Fatalf("an architecture with no condition is on")
							}
						}
					}
				}
			}
		}
	}
}

// SequenceParallelMoE treats an unstated backend as the pack's default, whatever that is.
func TestSequenceParallelMoEDefaultIsThePacks(t *testing.T) {
	for _, def := range []string{"in_set", "out_of_set"} {
		p := &Pack{DefaultAll2AllBackend: def, SPMoEBackends: map[string]bool{"in_set": true}}
		if p.SequenceParallelMoE("", true, 4, 2) != p.SequenceParallelMoE(def, true, 4, 2) {
			t.Errorf("default %q: an unstated backend differs from it", def)
		}
	}
}

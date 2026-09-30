package main

import (
	"fmt"
	"os"
	"sort"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
)

func main() {
	dir := "/Users/sri/Documents/Projects/blis-latency-kernel/testdata/aisimulate"
	ents, _ := os.ReadDir(dir)
	var names []string
	for _, e := range ents {
		names = append(names, e.Name())
	}
	sort.Strings(names)
	var ok, refused, suspect int
	for _, n := range names {
		m, err := kernelmodel.Open(n, kernelmodel.Repos{
			Scenarios: dir,
			Catalog:   "/Users/sri/Documents/Projects/blis-catalog",
			Registry:  "/Users/sri/Documents/Projects/blis-registry",
		})
		if err != nil {
			fmt.Printf("OPENFAIL %-44s %v\n", n, err)
			continue
		}
		epw := m.Kernel().Resolved().ExpertParallelWidth
		b, err := m.KVBudget()
		if err != nil {
			refused++
			fmt.Printf("REFUSED  %-44s epw=%d\n", n, epw)
			continue
		}
		// "Wrong but positive": EP off means the expert term was undivided, so the
		// fixed figure is too large and the budget too small -- but still positive.
		tag := "OK      "
		if epw <= 1 {
			tag = "SUSPECT "
			suspect++
		} else {
			ok++
		}
		fmt.Printf("%s %-44s epw=%d blocks=%8d fixed=%6.1fGiB\n",
			tag, n, epw, b.TotalBlocks, float64(b.FixedBytes)/(1<<30))
	}
	fmt.Printf("\n%d refused, %d suspect (EP off => undivided experts), %d unaffected\n",
		refused, suspect, ok)
}

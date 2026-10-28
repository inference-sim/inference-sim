// Moved from blis-latency-kernel/cmd/overlap-probe/main.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

// Command overlap-probe answers one question: does the per-layer overlap term change the
// SHAPE of the concurrency response, or only its level?
//
// It matters because cmd/shape scores a ratio of step times normalised to a sweep's lowest
// concurrency. Any term that scales proportionally with batch cancels in that ratio. If
// overlap is such a term, then the shape comparison is structurally blind to it, and a shape
// result near parity says nothing about whether resource overlap improves absolute accuracy.
//
// The probe prints Overlap and NoOverlap side by side, each normalised to its own value at the
// lowest concurrency. If the two normalised columns track each other, the ratio cannot
// distinguish them and cmd/shape cannot reward the overlap model.
//
// Usage:
//
//	go run ./cmd/overlap-probe testdata/aisimulate/glm-5-h200-fp8-sglang-tp8.yaml
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/inference-sim/blis-vllm/kerneltests/internal/harness"
)

func main() {
	catalog := flag.String("catalog", harness.DefaultCatalog(), "")
	registry := flag.String("registry", harness.DefaultRegistry(), "")
	context := flag.Int("context", 1024, "context tokens per request")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: overlap-probe <scenario.yaml>")
		os.Exit(2)
	}

	k, err := harness.Open(flag.Arg(0), harness.Repos{
		Catalog: *catalog, Registry: *registry,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	fmt.Printf("%-6s %12s %12s   %10s %10s %9s\n",
		"conc", "overlap_us", "noover_us", "ovl_rel", "noovl_rel", "spread")
	var o0, n0 float64
	for _, c := range []int{4, 8, 16, 32, 64, 128, 256} {
		e := k.StepTime(harness.DecodeBatch(c, *context))
		o := e.Overlap.Seconds() * 1e6
		n := e.NoOverlap.Seconds() * 1e6
		if o0 == 0 {
			o0, n0 = o, n
		}
		fmt.Printf("%-6d %12.1f %12.1f   %10.4f %10.4f %8.2f%%\n",
			c, o, n, o/o0, n/n0, 100*((o/o0)/(n/n0)-1))
	}
}

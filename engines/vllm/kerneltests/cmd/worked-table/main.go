// Moved from blis-latency-kernel/cmd/worked-table/main.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

// Command worked-table prints a worked step-time table, computed by the kernel itself from
// the committed catalog and registry.
//
// It exists so a document that quotes the table is checked against the implementation
// rather than against a second implementation of the same model. A Python replica of the kernel was the
// obvious way to verify the table and the wrong one: two implementations of one cost
// model drift, and every difference then has to be adjudicated even when neither side is
// wrong. There is one implementation, and this prints what it says.
//
// Usage:
//
//	go run ./cmd/worked-table
//	go run ./cmd/worked-table -format=markdown
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/inference-sim/blis-schemas/kernel"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-vllm/kerneltests/internal/harness"
)

// row is one line of the table: a batch shape and the scenario that prices it.
type row struct {
	label    string
	scenario string
	requests int
	queryLen int
	context  int
}

// Every row is a deployment the InferenceX vLLM corpus actually measured, named by
// its own scenario file under testdata/direct. An earlier version of this table priced
// a withheld preview model at EP=16 and EP=72, which is not an InferenceX deployment: a worked
// table whose configuration nothing measured cannot be checked against a measurement.
//
// The selection spans what the corpus varies: both chip families (Hopper runs FA3,
// vllm/v1/attention/backends/fa_utils.py:99-101; SM100 Blackwell prefers FlashInfer for
// causal non-MLA attention, vllm/platforms/cuda.py:157-166), pure
// tensor parallelism against expert parallelism, and both regimes, since prefill and
// decode dispatch differently (a prefill step's token count exceeds the cudagraph
// capture ceiling on 88.9% of corpus cells and so runs eager, while a decode step
// never does).
var rows = []row{
	{"1 req × 1024-token prefill, deepseek-v4-pro b200 `TP=8`",
		"deepseek-v4-pro-b200-fp4-vllm-tp8-cfg1.yaml", 1, 1024, 1024},
	{"1 req × 8192-token prefill, same deployment",
		"deepseek-v4-pro-b200-fp4-vllm-tp8-cfg1.yaml", 1, 8192, 8192},
	{"1 req × 1024-token prefill, minimax-m3 h200 `TP=8 EP=8`",
		"minimax-m3-h200-fp8-vllm-tp8-ep8-cfg1.yaml", 1, 1024, 1024},
	{"32 req × `q=1` decode, ctx 1k, minimax-m3 h200 `TP=8 EP=8`",
		"minimax-m3-h200-fp8-vllm-tp8-ep8-cfg1.yaml", 32, 1, 1024},
	{"256 req × `q=1` decode, ctx 1k, same deployment",
		"minimax-m3-h200-fp8-vllm-tp8-ep8-cfg1.yaml", 256, 1, 1024},
	{"256 req × `q=1` decode, ctx 1k, minimax-m3 b200 `TP=8 EP=8`",
		"minimax-m3-b200-fp4-vllm-tp8-ep8-cfg1.yaml", 256, 1, 1024},
}

func main() {
	catalog := flag.String("catalog", harness.DefaultCatalog(),
		"blis-catalog checkout")
	registry := flag.String("registry", harness.DefaultRegistry(),
		"blis-registry checkout")
	testdata := flag.String("testdata", "testdata/direct",
		"scenario fixtures; the InferenceX vLLM deployments live under testdata/direct")
	format := flag.String("format", "table", "table or markdown")
	flag.Parse()

	cache := map[string]*latencykernel.Kernel{}
	for _, r := range rows {
		if _, ok := cache[r.scenario]; ok {
			continue
		}
		k, err := harness.Open(r.scenario, harness.Repos{Scenarios: *testdata, Catalog: *catalog, Registry: *registry})
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", r.scenario, err)
			os.Exit(1)
		}
		cache[r.scenario] = k
	}

	if *format == "markdown" {
		fmt.Println("| Batch | HBM | SM | NVLink | `Overlap` | `Bottleneck` | Band |")
		fmt.Println("| --- | --- | --- | --- | --- | --- | --- |")
	} else {
		fmt.Printf("%-38s %8s %8s %8s %9s %-10s %6s\n",
			"batch", "HBM", "SM", "link", "Overlap", "bottleneck", "band")
	}
	for _, r := range rows {
		e := cache[r.scenario].StepTime(batch(r))
		ms := func(res kernel.Resource) float64 {
			return float64(e.PerResource[res].Microseconds()) / 1000
		}
		// NVLink and NIC are distinct resources that overlap with each other, so the
		// column reports the larger rather than their sum — summing them would exceed
		// the Overlap figure beside it and contradict the max-over-resources rule.
		link := ms(kernel.ResourceNVLink)
		if n := ms(kernel.ResourceNIC); n > link {
			link = n
		}
		overlap := float64(e.Overlap.Microseconds()) / 1000
		band := float64(e.NoOverlap) / float64(e.Overlap)
		if *format == "markdown" {
			bottleneck := string(e.Bottleneck)
			if e.Bottleneck == kernel.ResourceHBM {
				bottleneck = "**HBM**"
			}
			fmt.Printf("| %s | %.2f ms | %.2f ms | %.2f ms | **%.2f ms** | %s | %.2f× |\n",
				r.label, ms(kernel.ResourceHBM), ms(kernel.ResourceSM),
				link, overlap, bottleneck, band)
			continue
		}
		fmt.Printf("%-38s %8.2f %8.2f %8.2f %9.2f %-10s %5.2fx\n",
			r.label, ms(kernel.ResourceHBM), ms(kernel.ResourceSM),
			link, overlap, e.Bottleneck, band)
	}
}

func batch(r row) kernel.Batch {
	b := kernel.Batch{Reqs: make([]kernel.ReqShape, r.requests), DecodeThreshold: 8}
	for i := range b.Reqs {
		b.Reqs[i] = kernel.ReqShape{
			Scheduled: r.queryLen,
			Computed:  r.context - r.queryLen,
			PromptLen: r.context,
		}
	}
	return b
}

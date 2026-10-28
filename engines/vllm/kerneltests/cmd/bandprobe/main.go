// Moved from blis-latency-kernel/cmd/bandprobe/main.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

// Command bandprobe prices one step per line of a CSV and prints both band edges.
//
// It exists so a fitter outside this repository can compare both edges, `Overlap` and
// `NoOverlap`, with measured step times without reimplementing step composition. The kernel
// prices the step; this command only moves rows in and out. A second implementation of the
// cost model would be a second answer to the same modelling question, maintained apart.
//
// Input on stdin, one step per line, no header:
//
//	batch,prefill_tokens,context_tokens_per_sequence
//
// A prefill_tokens of 0 is a decode step: every request schedules one token. Otherwise
// each request schedules prefill_tokens and carries context_tokens_per_sequence of
// context, which is the mixed prefill+decode shape FPM measures.
//
// Output adds the two edges in microseconds and the bottleneck:
//
//	batch,prefill_tokens,context,overlap_us,nooverlap_us,bottleneck
//
// Usage:
//
//	go run ./cmd/bandprobe -scenario NAME.yaml -testdata DIR < steps.csv
package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/inference-sim/blis-schemas/kernel"

	"github.com/inference-sim/blis-vllm/kerneltests/internal/harness"
)

func main() {
	scen := flag.String("scenario", "", "scenario file name")
	td := flag.String("testdata", "testdata", "directory holding the scenario")
	cat := flag.String("catalog", harness.DefaultCatalog(), "")
	reg := flag.String("registry", harness.DefaultRegistry(), "")
	flag.Parse()

	k, err := harness.Open(*scen, harness.Repos{
		Scenarios: *td, Catalog: *cat, Registry: *reg,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	in := csv.NewReader(bufio.NewReader(os.Stdin))
	in.FieldsPerRecord = -1
	out := csv.NewWriter(os.Stdout)
	defer out.Flush()
	_ = out.Write([]string{
		"batch", "prefill_tokens", "context", "overlap_us", "nooverlap_us", "bottleneck",
	})

	for {
		rec, err := in.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if len(rec) < 3 {
			continue
		}
		batch, e1 := strconv.Atoi(rec[0])
		prefill, e2 := strconv.Atoi(rec[1])
		ctx, e3 := strconv.Atoi(rec[2])
		if e1 != nil || e2 != nil || e3 != nil || batch <= 0 {
			continue
		}
		var b kernel.Batch
		if prefill > 0 {
			b = prefillBatch(batch, prefill, ctx)
		} else {
			b = decodeBatch(batch, ctx)
		}
		e := k.StepTime(b)
		_ = out.Write([]string{
			rec[0], rec[1], rec[2],
			strconv.FormatFloat(e.Overlap.Seconds()*1e6, 'f', 3, 64),
			strconv.FormatFloat(e.NoOverlap.Seconds()*1e6, 'f', 3, 64),
			fmt.Sprint(e.Bottleneck),
		})
	}
}

// decodeBatch builds the same shape harness.DecodeBatch does, and a test pins it there.
// Computed is context-1 because the context a decode step reads includes the token it is
// about to produce, and DecodeThreshold selects the attention form per request.
func decodeBatch(batch, ctx int) kernel.Batch {
	return harness.DecodeBatch(batch, ctx)
}

// prefillBatch builds a chunked-prefill step: each request schedules `prefill` tokens on
// top of `ctx` already computed, out of a prompt that is at least their sum. This is the
// mixed prefill-plus-context shape FPM measures.
func prefillBatch(batch, prefill, ctx int) kernel.Batch {
	reqs := make([]kernel.ReqShape, batch)
	for i := range reqs {
		reqs[i] = kernel.ReqShape{
			Scheduled: prefill, Computed: ctx, PromptLen: ctx + prefill,
		}
	}
	return kernel.Batch{Reqs: reqs, DecodeThreshold: harness.DecodeThreshold}
}

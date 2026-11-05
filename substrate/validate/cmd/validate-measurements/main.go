// Command validate-measurements checks a measurements directory: the licence registry in
// sources.yaml, and every measured run under <license_ref>/<name>.yaml against it.
//
// validate-catalog runs the same check over a catalog's measurements/. This command is
// for an adapter's output before it is committed, and for output that must stay local:
// with -local, a run whose licence permits scoring but not redistribution passes.
//
// A run's verdict is recomputed from its diff, provenance rows and licence, and a stated
// verdict that differs fails. Exit 0 when everything validates, 1 when anything fails,
// 2 on misuse.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/inference-sim/blis-substrate/validate/measurements"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("validate-measurements", flag.ContinueOnError)
	fs.SetOutput(stderr)
	local := fs.Bool("local", false, "allow runs whose licence does not permit redistribution")
	if err := fs.Parse(args); err != nil || fs.NArg() != 1 {
		fmt.Fprintln(stderr, "usage: validate-measurements [-local] <measurements dir>")
		return 2
	}
	results, err := measurements.Check(fs.Arg(0), !*local)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	failed := 0
	for _, r := range results {
		if r.Err != nil {
			fmt.Fprintf(stderr, "%s: %v\n", r.Name, r.Err)
			fmt.Fprintf(stdout, "%-60s FAILED (load error)\n", r.Name)
			failed++
			continue
		}
		for _, p := range r.Problems.All() {
			fmt.Fprintf(stderr, "%s: %s\n", r.Name, p)
		}
		if !r.Problems.OK() {
			fmt.Fprintf(stdout, "%-60s FAILED (%d problem(s))\n", r.Name, len(r.Problems.Errors()))
			failed++
			continue
		}
		fmt.Fprintf(stdout, "%-60s %s\n", r.Name, r.Summary)
	}
	if failed > 0 {
		fmt.Fprintf(stderr, "\n%d of %d file(s) failed\n", failed, len(results))
		return 1
	}
	fmt.Fprintf(stdout, "\nall %d file(s) validate\n", len(results))
	return 0
}

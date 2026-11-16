// Command unselected-entries lists the coefficient registry entries that meet neither condition
// of design VAL-8: no lookup on the conformance scenarios selected the entry, and no engine
// release's kernel-implementation table lists a kernel implementation the entry is valid for.
// Nothing the project runs reads such an entry, and the next substrate release removes it.
//
// Usage:
//
//	unselected-entries [-selected RESULT]... [-kernel-impls TABLE]... <registry root>
//
// Each -selected file is a Result (JSON or YAML) of a conformance scenario's run, whose
// selected_entries names the entries its lookups returned. Each -kernel-impls file is the
// kernel_impls.json of an engine release that some supported simulator release pins. Both
// flags repeat. The command writes one line per entry it lists to stdout and a summary to
// stderr. It exits 0 when it lists none, 1 when it lists some, and 2 on misuse or a file that
// does not load.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/kernelimpls"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
	"github.com/inference-sim/blis-substrate/validate/dataset"
	"github.com/inference-sim/blis-substrate/validate/registry"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

type files []string

func (f *files) String() string     { return strings.Join(*f, ",") }
func (f *files) Set(s string) error { *f = append(*f, s); return nil }

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("unselected-entries", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var selected, tables files
	fs.Var(&selected, "selected", "a Result of a conformance scenario's run (repeatable)")
	fs.Var(&tables, "kernel-impls", "an engine release's kernel_impls.json (repeatable)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "usage: unselected-entries [-selected RESULT]... [-kernel-impls TABLE]... <registry root>")
		return 2
	}
	root := fs.Arg(0)
	paths, err := registry.SetPaths(root)
	if err != nil {
		fmt.Fprintf(stderr, "scanning %s: %v\n", root, err)
		return 2
	}
	if len(paths) == 0 {
		fmt.Fprintf(stderr, "%s: no coefficient sets found under coefficients/\n", root)
		return 2
	}
	sets := map[string]*coefficient.Set{}
	for _, p := range paths {
		s, err := contracts.LoadCoefficientSet(p)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 2
		}
		sets[dataset.Stem(p)] = s
	}
	var reports []result.SelectedEntries
	for _, p := range selected {
		r, err := contracts.LoadResult(p)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 2
		}
		reports = append(reports, r.SelectedEntries)
	}
	var tabs []*kernelimpls.Table
	for _, p := range tables {
		t, err := contracts.LoadKernelImplTable(p)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 2
		}
		tabs = append(tabs, t)
	}
	found := dataset.FindUnselected(sets, reports, tabs)
	for _, u := range found {
		fmt.Fprintln(stdout, u)
	}
	total := 0
	for _, s := range sets {
		total += len(s.Coefficients)
	}
	fmt.Fprintf(stderr, "%d of %d entries are neither selected by the %d run(s) given nor published by the %d table(s) given (VAL-8)\n",
		len(found), total, len(reports), len(tabs))
	if len(found) > 0 {
		return 1
	}
	return 0
}
